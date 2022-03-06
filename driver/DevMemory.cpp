#include "DeviceMemory.h"

// ======================================= roc::Memory ============================================
DevMemory::DevMemory(const amd::Device& dev, amd::Memory& owner)
    : flags_(0)
    , owner_(nullptr)
    , version_(0)
    , mapMemory_(nullptr)
    , indirectMapCount_(0)
    , size_(owner.getSize())
    , dev_(dev)
    , deviceMemory_(nullptr)
    ,  kind_(MEMORY_KIND_NORMAL)
    // ,  amdImageDesc_(nullptr),
    ,  persistent_host_ptr_(nullptr)
    ,  pinnedMemory_(nullptr) {}

DevMemory::DevMemory(const roc::Device& dev, size_t size)
    : flags_(0)
    , owner_(nullptr)
    , version_(0)
    , mapMemory_(nullptr)
    , indirectMapCount_(0)
    , size_(size) {}
    , dev_(dev)
    , deviceMemory_(nullptr)
    , kind_(MEMORY_KIND_NORMAL)
    // , amdImageDesc_(nullptr),
    , persistent_host_ptr_(nullptr)
      pinnedMemory_(nullptr) {}

DevMemory::~DevMemory() {
  // Destory pinned memory
  if (flags_ & PinnedMemoryAlloced) {
    pinnedMemory_->release();
  }

  dev().removeVACache(this);
  if (nullptr != mapMemory_) {
    mapMemory_->release();
  }
}

bool Memory::pinSystemMemory(void* hostPtr, size_t size) {
  size_t pinAllocSize;
  const static bool SysMem = true;
  amd::Memory* amdMemory = nullptr;
  amd::Memory* amdParent = owner()->parent();

  // If memory has a direct access already, then skip the host memory pinning
  if (isHostMemDirectAccess()) {
    return true;
  }

  // Memory was pinned already
  if (flags_ & PinnedMemoryAlloced) {
    return true;
  }

  // Check if runtime allocates a parent object
  if (amdParent != nullptr) {
    Memory* parent = dev().getDevMemory(amdParent);
    amd::Memory* amdPinned = parent->pinnedMemory_;
    if (amdPinned != nullptr) {
      // Create view on the parent's pinned memory
      amdMemory = new (amdPinned->getContext())
          amd::Buffer(*amdPinned, 0, owner()->getOrigin(), owner()->getSize());
      if ((amdMemory != nullptr) && !amdMemory->create()) {
        amdMemory->release();
        amdMemory = nullptr;
      }
    }
  }

  if (amdMemory == nullptr) {
    amdMemory = new (dev().context()) amd::Buffer(dev().context(), CL_MEM_USE_HOST_PTR, size);
    if ((amdMemory != nullptr) && !amdMemory->create(hostPtr, SysMem)) {
      amdMemory->release();
      return false;
    }
  }

  // Get device memory for this virtual device
  // @note: This will force real memory pinning
  Memory* srcMemory = dev().getDevMemory(amdMemory);

  if (srcMemory == nullptr) {
    // Release memory
    amdMemory->release();
    return false;
  } else {
    pinnedMemory_ = amdMemory;
    flags_ |= PinnedMemoryAlloced;
  }

  return true;
}
#if 0
void Memory::syncCacheFromHost(VirtualGPU& gpu, device::Memory::SyncFlags syncFlags) {
  // If the last writer was another GPU, then make a writeback
  if (!isHostMemDirectAccess() && (owner()->getLastWriter() != nullptr) &&
      (&dev() != owner()->getLastWriter())) {
    // Make sure GPU finished operation before synchronization with the backing store
    gpu.releaseGpuMemoryFence();
    mgpuCacheWriteBack();
  }

  // If host memory doesn't have direct access, then we have to synchronize
  if (!isHostMemDirectAccess() && (nullptr != owner()->getHostMem())) {
    bool hasUpdates = true;
    amd::Memory* amdParent = owner()->parent();

    // Make sure the parent of subbuffer is up to date
    if (!syncFlags.skipParent_ && (amdParent != nullptr)) {
      Memory* gpuMemory = dev().getRocMemory(amdParent);

      //! \note: Skipping the sync for a view doesn't reflect the parent settings,
      //! since a view is a small portion of parent
      device::Memory::SyncFlags syncFlagsTmp;

      // Sync parent from a view, so views have to be skipped
      syncFlagsTmp.skipViews_ = true;

      // Make sure the parent sync is an unique operation.
      // If the app uses multiple subbuffers from multiple queues,
      // then the parent sync can be called from multiple threads
      amd::ScopedLock lock(owner()->parent()->lockMemoryOps());
      gpuMemory->syncCacheFromHost(gpu, syncFlagsTmp);
      //! \note Don't do early exit here, since we still have to sync
      //! this view, if the parent sync operation was a NOP.
      //! If parent was synchronized, then this view sync will be a NOP
    }

    // Is this a NOP?
    if ((version_ == owner()->getVersion()) || (&dev() == owner()->getLastWriter())) {
      hasUpdates = false;
    }

    // Update all available views, since we sync the parent
    if ((owner()->subBuffers().size() != 0) && (hasUpdates || !syncFlags.skipViews_)) {
      device::Memory::SyncFlags syncFlagsTmp;

      // Sync views from parent, so parent has to be skipped
      syncFlagsTmp.skipParent_ = true;

      if (hasUpdates) {
        // Parent will be synced so update all views with a skip
        syncFlagsTmp.skipEntire_ = true;
      } else {
        // Passthrough the skip entire flag to the views, since
        // any view is a submemory of the parent
        syncFlagsTmp.skipEntire_ = syncFlags.skipEntire_;
      }

      amd::ScopedLock lock(owner()->lockMemoryOps());
      for (auto& sub : owner()->subBuffers()) {
        //! \note Don't allow subbuffer's allocation in the worker thread.
        //! It may cause a system lock, because possible resource
        //! destruction, heap reallocation or subbuffer allocation
        static const bool AllocSubBuffer = false;
        device::Memory* devSub = sub->getDeviceMemory(dev(), AllocSubBuffer);
        if (nullptr != devSub) {
          Memory* gpuSub = reinterpret_cast<Memory*>(devSub);
          gpuSub->syncCacheFromHost(gpu, syncFlagsTmp);
        }
      }
    }

    // Make sure we didn't have a NOP,
    // because this GPU device was the last writer
    if (&dev() != owner()->getLastWriter()) {
      // Update the latest version
      version_ = owner()->getVersion();
    }

    // Exit if sync is a NOP or sync can be skipped
    if (!hasUpdates || syncFlags.skipEntire_) {
      return;
    }

    bool result = false;
    static const bool Entire = true;
    amd::Coord3D origin(0, 0, 0);

    // If host memory was pinned then make a transfer
    if (flags_ & PinnedMemoryAlloced) {
      Memory& pinned = *dev().getRocMemory(pinnedMemory_);
      if (owner()->getType() == CL_MEM_OBJECT_BUFFER) {
        amd::Coord3D region(owner()->getSize());
        result = gpu.blitMgr().copyBuffer(pinned, *this, origin, origin, region, Entire);
      } else {
        amd::Image& image = static_cast<amd::Image&>(*owner());
        result =
            gpu.blitMgr().copyBufferToImage(pinned, *this, origin, origin, image.getRegion(),
                                            Entire, image.getRowPitch(), image.getSlicePitch());
      }
    }

    if (!result) {
      if (owner()->getType() == CL_MEM_OBJECT_BUFFER) {
        amd::Coord3D region(owner()->getSize());
        result = gpu.blitMgr().writeBuffer(owner()->getHostMem(), *this, origin, region, Entire);
      } else {
        amd::Image& image = static_cast<amd::Image&>(*owner());
        result = gpu.blitMgr().writeImage(owner()->getHostMem(), *this, origin, image.getRegion(),
                                          image.getRowPitch(), image.getSlicePitch(), Entire);
      }
    }

    // Should never fail
    assert(result && "Memory synchronization failed!");
  }
}

void Memory::syncHostFromCache(device::Memory::SyncFlags syncFlags) {
  // Sanity checks
  assert(owner() != nullptr);

  // If host memory doesn't have direct access, then we have to synchronize
  if (!isHostMemDirectAccess()) {
    bool hasUpdates = true;
    amd::Memory* amdParent = owner()->parent();

    // Make sure the parent of subbuffer is up to date
    if (!syncFlags.skipParent_ && (amdParent != nullptr)) {
      device::Memory* m = dev().getRocMemory(amdParent);

      //! \note: Skipping the sync for a view doesn't reflect the parent settings,
      //! since a view is a small portion of parent
      device::Memory::SyncFlags syncFlagsTmp;

      // Sync parent from a view, so views have to be skipped
      syncFlagsTmp.skipViews_ = true;

      // Make sure the parent sync is an unique operation.
      // If the app uses multiple subbuffers from multiple queues,
      // then the parent sync can be called from multiple threads
      amd::ScopedLock lock(owner()->parent()->lockMemoryOps());
      m->syncHostFromCache(syncFlagsTmp);
      //! \note Don't do early exit here, since we still have to sync
      //! this view, if the parent sync operation was a NOP.
      //! If parent was synchronized, then this view sync will be a NOP
    }

    // Is this a NOP?
    if ((nullptr == owner()->getLastWriter()) || (version_ == owner()->getVersion())) {
      hasUpdates = false;
    }

    // Update all available views, since we sync the parent
    if ((owner()->subBuffers().size() != 0) && (hasUpdates || !syncFlags.skipViews_)) {
      device::Memory::SyncFlags syncFlagsTmp;

      // Sync views from parent, so parent has to be skipped
      syncFlagsTmp.skipParent_ = true;

      if (hasUpdates) {
        // Parent will be synced so update all views with a skip
        syncFlagsTmp.skipEntire_ = true;
      } else {
        // Passthrough the skip entire flag to the views, since
        // any view is a submemory of the parent
        syncFlagsTmp.skipEntire_ = syncFlags.skipEntire_;
      }

      amd::ScopedLock lock(owner()->lockMemoryOps());
      for (auto& sub : owner()->subBuffers()) {
        //! \note Don't allow subbuffer's allocation in the worker thread.
        //! It may cause a system lock, because possible resource
        //! destruction, heap reallocation or subbuffer allocation
        static const bool AllocSubBuffer = false;
        device::Memory* devSub = sub->getDeviceMemory(dev(), AllocSubBuffer);
        if (nullptr != devSub) {
          Memory* gpuSub = reinterpret_cast<Memory*>(devSub);
          gpuSub->syncHostFromCache(syncFlagsTmp);
        }
      }
    }

    // Make sure we didn't have a NOP,
    // because CPU was the last writer
    if (nullptr != owner()->getLastWriter()) {
      // Mark parent as up to date, set our version accordingly
      version_ = owner()->getVersion();
    }

    // Exit if sync is a NOP or sync can be skipped
    if (!hasUpdates || syncFlags.skipEntire_) {
      return;
    }

    bool result = false;
    static const bool Entire = true;
    amd::Coord3D origin(0, 0, 0);

    // If backing store was pinned then make a transfer
    if (flags_ & PinnedMemoryAlloced) {
      Memory& pinned = *dev().getRocMemory(pinnedMemory_);
      if (owner()->getType() == CL_MEM_OBJECT_BUFFER) {
        amd::Coord3D region(owner()->getSize());
        result = dev().xferMgr().copyBuffer(*this, pinned, origin, origin, region, Entire);
      } else {
        amd::Image& image = static_cast<amd::Image&>(*owner());
        result =
            dev().xferMgr().copyImageToBuffer(*this, pinned, origin, origin, image.getRegion(),
                                              Entire, image.getRowPitch(), image.getSlicePitch());
      }
    }

    // Just do a basic host read
    if (!result) {
      if (owner()->getType() == CL_MEM_OBJECT_BUFFER) {
        amd::Coord3D region(owner()->getSize());
        result = dev().xferMgr().readBuffer(*this, owner()->getHostMem(), origin, region, Entire);
      } else {
        amd::Image& image = static_cast<amd::Image&>(*owner());
        result = dev().xferMgr().readImage(*this, owner()->getHostMem(), origin, image.getRegion(),
                                           image.getRowPitch(), image.getSlicePitch(), Entire);
      }
    }

    // Should never fail
    assert(result && "Memory synchronization failed!");
  }
}
void Memory::mgpuCacheWriteBack() {
  // Lock memory object, so only one write back can occur
  amd::ScopedLock lock(owner()->lockMemoryOps());

  // Attempt to allocate a staging buffer if don't have any
  if (owner()->getHostMem() == nullptr) {
    if (nullptr != owner()->getSvmPtr()) {
      owner()->commitSvmMemory();
      owner()->setHostMem(owner()->getSvmPtr());
    } else {
      static const bool forceAllocHostMem = true;
      owner()->allocHostMemory(nullptr, forceAllocHostMem);
    }
  }

  // Make synchronization
  if (owner()->getHostMem() != nullptr) {
    //! \note Ignore pinning result
    bool ok = pinSystemMemory(owner()->getHostMem(), owner()->getSize());
    owner()->cacheWriteBack();
  }
}

void* Memory::cpuMap(uint flags, uint startLayer, uint numLayers,
                     size_t* rowPitch, size_t* slicePitch) {
#endif

Buffer::Buffer(const Device& dev, amd::Memory& owner) : DevMemory(dev, owner) {}

Buffer::Buffer(const Device& dev, size_t size) : DevMemory(dev, size) {}

Buffer::~Buffer() {
  if (owner() == nullptr) {
    dev().hostFree(deviceMemory_, size());
  } else {
    destroy();
  }
}

void Buffer::destroy() {
  if (owner()->parent() != nullptr) {
    return;
  }
/*
  if (kind_ == MEMORY_KIND_INTEROP) {
    destroyInteropBuffer();
    return;
  }

  cl_mem_flags memFlags = owner()->getMemFlags();

  if (owner()->getSvmPtr() != nullptr) {
    if (dev().forceFineGrain(owner()) || dev().isFineGrainedSystem(true)) {
      memFlags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
    }
    const bool isFineGrain = memFlags & CL_MEM_SVM_FINE_GRAIN_BUFFER;

    if (kind_ != MEMORY_KIND_PTRGIVEN) {
      if (isFineGrain) {
        if (memFlags & CL_MEM_ALLOC_HOST_PTR) {
          if (dev().info().hmmSupported_) {
            // AMD HMM path. Destroy system memory
            amd::Os::uncommitMemory(deviceMemory_, size());
            amd::Os::releaseMemory(deviceMemory_, size());
          } else {
            dev().hostFree(deviceMemory_, size());
          }
        } else if (memFlags & ROCCLR_MEM_HSA_SIGNAL_MEMORY) {
          if (HSA_STATUS_SUCCESS != hsa_signal_destroy(signal_)) {
            ClPrint(amd::LOG_DEBUG, amd::LOG_MEM,
                    "[ROCClr] ROCCLR_MEM_HSA_SIGNAL_MEMORY signal destroy failed \n");
          }
          deviceMemory_ = nullptr;
        } else {
          dev().hostFree(deviceMemory_, size());
        }
      } else {
        dev().memFree(deviceMemory_, size());
      }
    }

    if ((deviceMemory_ != nullptr) &&
        (dev().settings().apuSystem_ || !isFineGrain)) {
      const_cast<Device&>(dev()).updateFreeMemory(size(), true);
    }

    return;
  }

#ifdef WITH_AMDGPU_PRO
  if ((memFlags & CL_MEM_USE_PERSISTENT_MEM_AMD) && dev().ProEna()) {
    dev().iPro().FreeDmaBuffer(deviceMemory_);
    return;
  }
#endif
  if (deviceMemory_ != nullptr) {
    if (deviceMemory_ != owner()->getHostMem()) {
      // if they are identical, the host pointer will be
      // deallocated later on => avoid double deallocation
      if (isHostMemDirectAccess()) {
        if (memFlags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR)) {
          if (dev().agent_profile() != HSA_PROFILE_FULL) {
            hsa_amd_memory_unlock(owner()->getHostMem());
          }
        }
      } else {
        dev().memFree(deviceMemory_, size());
        const_cast<Device&>(dev()).updateFreeMemory(size(), true);
      }
    }
    else {
      if (!(memFlags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR))) {
        dev().memFree(deviceMemory_, size());
        if (dev().settings().apuSystem_) {
          const_cast<Device&>(dev()).updateFreeMemory(size(), true);
        }
      }
    }
  }

  if (memFlags & CL_MEM_USE_HOST_PTR) {
    if (dev().agent_profile() == HSA_PROFILE_FULL) {
      hsa_memory_deregister(owner()->getHostMem(), size());
    }
  }
  */
}

// ================================================================================================
bool Buffer::create() {
  if (owner() == nullptr) {
    deviceMemory_ = dev().hostAlloc(size(), 1, Device::MemorySegment::kNoAtomics);
    if (deviceMemory_ != nullptr) {
      flags_ |= HostMemoryDirectAccess;
      return true;
    }
    return false;
  }

  // Allocate backing storage in device local memory unless UHP or AHP are set
  cl_mem_flags memFlags = owner()->getMemFlags();

  if ((owner()->parent() == nullptr) &&
      (owner()->getSvmPtr() != nullptr)) {
    if (dev().forceFineGrain(owner()) || dev().isFineGrainedSystem(true)) {
      memFlags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
    }
    const bool isFineGrain = memFlags & CL_MEM_SVM_FINE_GRAIN_BUFFER;

    if (isFineGrain) {
      // Use CPU direct access for the fine grain buffer
      flags_ |= HostMemoryDirectAccess;
    }

    if (owner()->getSvmPtr() == reinterpret_cast<void*>(amd::Memory::MemoryType::kSvmMemoryPtr)) {
      if (isFineGrain) {
        if (memFlags & CL_MEM_ALLOC_HOST_PTR) {
          if (dev().info().hmmSupported_) {
            // AMD HMM path. Just allocate system memory and KFD will manage it
            deviceMemory_ =  amd::Os::reserveMemory(
                0, size(), amd::Os::pageSize(), amd::Os::MEM_PROT_RW);
            amd::Os::commitMemory(deviceMemory_, size(), amd::Os::MEM_PROT_RW);
            // Currently HMM requires cirtain initial calls to mark sysmem allocation as
            // GPU accessible or prefetch memory into GPU
            if (!dev().SvmAllocInit(deviceMemory_, size())) {
              ClPrint(amd::LOG_ERROR, amd::LOG_MEM, "SVM init in ROCr failed!");
              return false;
            }
          } else {
            deviceMemory_ = dev().hostAlloc(size(), 1, Device::MemorySegment::kNoAtomics);
          }
        } else if (memFlags & CL_MEM_FOLLOW_USER_NUMA_POLICY) {
          deviceMemory_ = dev().hostNumaAlloc(size(), 1, (memFlags & CL_MEM_SVM_ATOMICS) != 0);
        } else if (memFlags & ROCCLR_MEM_HSA_SIGNAL_MEMORY) {
          // TODO: ROCr will introduce a new attribute enum that implies a non-blocking signal,
          // replace "HSA_AMD_SIGNAL_AMD_GPU_ONLY" with this new enum when it is ready.
          if (HSA_STATUS_SUCCESS !=
              hsa_amd_signal_create(kInitSignalValueOne, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY,
                                    &signal_)) {
            ClPrint(amd::LOG_ERROR, amd::LOG_MEM,
                    "[ROCclr] ROCCLR_MEM_HSA_SIGNAL_MEMORY signal creation failed");
            return false;
          }
          volatile hsa_signal_value_t* signalValuePtr = nullptr;
          if (HSA_STATUS_SUCCESS != hsa_amd_signal_value_pointer(signal_, &signalValuePtr)) {
            ClPrint(amd::LOG_ERROR, amd::LOG_MEM,
                    "[ROCclr] ROCCLR_MEM_HSA_SIGNAL_MEMORY pointer query failed");
            return false;
          }

          deviceMemory_ = const_cast<long int*>(signalValuePtr);  // conversion to void * is
                                                                  // implicit

          // Disable host access to force blit path for memeory writes.
          flags_ &= ~HostMemoryDirectAccess;
        } else {
          deviceMemory_ = dev().hostAlloc(size(), 1, ((memFlags & CL_MEM_SVM_ATOMICS) != 0)
                                                       ? Device::MemorySegment::kAtomics
                                                       : Device::MemorySegment::kNoAtomics);
        }
      } else {
        assert(!isHostMemDirectAccess() && "Runtime doesn't support direct access to GPU memory!");
        deviceMemory_ = dev().deviceLocalAlloc(size(), (memFlags & CL_MEM_SVM_ATOMICS) != 0);
      }
      owner()->setSvmPtr(deviceMemory_);
    } else {
      deviceMemory_ = owner()->getSvmPtr();
      if (owner()->getSvmPtr() == reinterpret_cast<void*>(amd::Memory::MemoryType
                                                          ::kArenaMemoryPtr)) {
        kind_ = MEMORY_KIND_ARENA;
      } else {
        kind_ = MEMORY_KIND_PTRGIVEN;
      }
    }

    if ((deviceMemory_ != nullptr) && (dev().settings().apuSystem_ || !isFineGrain)
                                   && (kind_ != MEMORY_KIND_ARENA)) {
      const_cast<Device&>(dev()).updateFreeMemory(size(), false);
    }

    return deviceMemory_ != nullptr;
  }

  // Interop buffer
  if (owner()->isInterop()) {
    amd::InteropObject* interop = owner()->getInteropObj();
    amd::VkObject* vkObject = interop->asVkObject();
    amd::GLObject* glObject = interop->asGLObject();
    if (vkObject != nullptr) {
      hsa_status_t status = interopMapBuffer(vkObject->getVkSharedHandle());
      if (status != HSA_STATUS_SUCCESS) return false;
      return true;
    } else if (glObject != nullptr) {
      return createInteropBuffer(GL_ARRAY_BUFFER,0);
    }
  }
  if (nullptr != owner()->parent()) {
    amd::Memory& parent = *owner()->parent();
    // Sub-Buffer creation.
    roc::Memory* parentBuffer = static_cast<roc::Memory*>(parent.getDeviceMemory(dev_));

    if (parentBuffer == nullptr) {
      LogError("[OCL] Fail to allocate parent buffer");
      return false;
    }

    const size_t offset = owner()->getOrigin();
    deviceMemory_ = parentBuffer->getDeviceMemory() + offset;

    flags_ |= parentBuffer->isHostMemDirectAccess() ? HostMemoryDirectAccess : 0;
    flags_ |= parentBuffer->isCpuUncached() ? MemoryCpuUncached : 0;

    // Explicitly set the host memory location,
    // because the parent location could change after reallocation
    if (nullptr != parent.getHostMem()) {
      owner()->setHostMem(reinterpret_cast<char*>(parent.getHostMem()) + offset);
    } else {
      owner()->setHostMem(nullptr);
    }

    return true;
  }

#ifdef WITH_AMDGPU_PRO
  if ((memFlags & CL_MEM_USE_PERSISTENT_MEM_AMD) && dev().ProEna()) {
    void* host_ptr = nullptr;
    deviceMemory_ = dev().iPro().AllocDmaBuffer(dev().getBackendDevice(), size(), &host_ptr);
    if (deviceMemory_ == nullptr) {
      return false;
    }
    persistent_host_ptr_ = host_ptr;
    return true;
  }
#endif

  if (!(memFlags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR))) {
    deviceMemory_ = dev().deviceLocalAlloc(size());

    if (deviceMemory_ == nullptr) {
      // TODO: device memory is not enabled yet.
      // Fallback to system memory if exist.
      flags_ |= HostMemoryDirectAccess;
      if (dev().agent_profile() == HSA_PROFILE_FULL && owner()->getHostMem() != nullptr) {
        deviceMemory_ = owner()->getHostMem();
        assert(
            amd::isMultipleOf(deviceMemory_, static_cast<size_t>(dev().info().memBaseAddrAlign_)));
        return true;
      }

      deviceMemory_ = dev().hostAlloc(size(), 1, Device::MemorySegment::kNoAtomics);
      owner()->setHostMem(deviceMemory_);

      if ((deviceMemory_ != nullptr) && dev().settings().apuSystem_) {
        const_cast<Device&>(dev()).updateFreeMemory(size(), false);
      }
    }
    else {
      const_cast<Device&>(dev()).updateFreeMemory(size(), false);
    }

    assert(amd::isMultipleOf(deviceMemory_, static_cast<size_t>(dev().info().memBaseAddrAlign_)));

    // Transfer data only if OCL context has one device.
    // Cache coherency layer will update data for multiple devices
    if (deviceMemory_ && (memFlags & CL_MEM_COPY_HOST_PTR) &&
        (owner()->getContext().devices().size() == 1)) {
      // To avoid recurssive call to Device::createMemory, we perform
      // data transfer to the view of the buffer.
      amd::Buffer* bufferView = new (owner()->getContext())
          amd::Buffer(*owner(), 0, owner()->getOrigin(), owner()->getSize());
      bufferView->create(nullptr, false, true);

      roc::Buffer* devBufferView = new roc::Buffer(dev_, *bufferView);
      devBufferView->deviceMemory_ = deviceMemory_;

      bufferView->replaceDeviceMemory(&dev_, devBufferView);

      bool ret = dev().xferMgr().writeBuffer(owner()->getHostMem(), *devBufferView, amd::Coord3D(0),
                                             amd::Coord3D(size()), true);

      // Release host memory, since runtime copied data
      owner()->setHostMem(nullptr);
      bufferView->release();
      return ret;
    }

    return deviceMemory_ != nullptr;
  }
  assert(owner()->getHostMem() != nullptr);

  flags_ |= HostMemoryDirectAccess;

  if (dev().agent_profile() == HSA_PROFILE_FULL) {
    deviceMemory_ = owner()->getHostMem();

    if (memFlags & CL_MEM_USE_HOST_PTR) {
      hsa_memory_register(deviceMemory_, size());
    }

    return deviceMemory_ != nullptr;
  }

  if (owner()->getSvmPtr() != owner()->getHostMem()) {
    if (memFlags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR)) {
      hsa_amd_memory_pool_t pool = (memFlags & CL_MEM_SVM_ATOMICS) ?
                                    dev().SystemSegment() :
                                    (dev().SystemCoarseSegment().handle != 0 ?
                                        dev().SystemCoarseSegment() : dev().SystemSegment());
      hsa_status_t status = hsa_amd_memory_lock_to_pool(owner()->getHostMem(),
          owner()->getSize(), nullptr, 0, pool, 0, &deviceMemory_);
      ClPrint(amd::LOG_DEBUG, amd::LOG_MEM, "Locking to pool %p, size 0x%zx, HostPtr = %p,"
              " DevPtr = %p", pool, owner()->getSize(), owner()->getHostMem(), deviceMemory_ );
      if (status != HSA_STATUS_SUCCESS) {
        DevLogPrintfError("Failed to lock memory to pool, failed with hsa_status: %d \n", status);
        deviceMemory_ = nullptr;
      }
    } else {
      deviceMemory_ = owner()->getHostMem();
    }
  } else {
    deviceMemory_ = owner()->getHostMem();
  }

  return deviceMemory_ != nullptr;
}


