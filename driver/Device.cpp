#include "Device.h"

namespace drv {

bool roc::Device::isHsaInitialized_ = false;
std::vector<hsa_agent_t> roc::Device::gpu_agents_;
std::vector<AgentInfo> roc::Device::cpu_agents_;
address Device::mg_sync_ = nullptr;

Device::Device(hsa_agent_t bkendDevice)
    : mapCacheOps_(nullptr)
    , mapCache_(nullptr)
    , _bkendDevice(bkendDevice)
    , pciDeviceId_(0)
    , gpuvm_segment_max_alloc_(0)
    , alloc_granularity_(0)
    , xferQueue_(nullptr)
    , xferRead_(nullptr)
    , xferWrite_(nullptr)
    , pro_device_(nullptr)
    , pro_ena_(false)
    , freeMem_(0)
    , vgpusAccess_("Virtual GPU List Ops Lock", true)
    , hsa_exclusive_gpu_access_(false)
    , queuePool_(QueuePriority::Total)
    , coopHostcallBuffer_(nullptr)
    , queueWithCUMaskPool_(QueuePriority::Total)
    , numOfVgpus_(0)
    , preferred_numa_node_(0) {
  group_segment_.handle = 0;
  system_segment_.handle = 0;
  system_coarse_segment_.handle = 0;
  system_kernarg_segment_.handle = 0;
  gpuvm_segment_.handle = 0;
  gpu_fine_grained_segment_.handle = 0;
  prefetch_signal_.handle = 0;
}

void Device::setupCpuAgent() {
  int32_t numaDistance = std::numeric_limits<int32_t>::max();
  uint32_t index = 0; // 0 as default
  auto size = cpu_agents_.size();
  for (uint32_t i = 0; i < size; i++) {
    std::vector<drv::Device::LinkAttrType> link_attrs;
    link_attrs.push_back(std::make_pair(LinkAttribute::kLinkDistance, 0));
    if (findLinkInfo(cpu_agents_[i].fine_grain_pool, &link_attrs)) {
      if (link_attrs[0].second < numaDistance) {
        numaDistance = link_attrs[0].second;
        index = i;
      }
    }
  }
  preferred_numa_node_ = index;
  cpu_agent_ = cpu_agents_[index].agent;
  system_segment_ = cpu_agents_[index].fine_grain_pool;
  system_coarse_segment_ = cpu_agents_[index].coarse_grain_pool;
  system_kernarg_segment_ = cpu_agents_[index].kern_arg_pool;
  ClPrint(drv::LOG_INFO, drv::LOG_INIT, "Numa selects cpu agent[%zu]=0x%zx(fine=0x%zx,"
          "coarse=0x%zx) for gpu agent=0x%zx", index, cpu_agent_.handle,
          system_segment_.handle, system_coarse_segment_.handle, _bkendDevice.handle);
}

Device::~Device() {
#ifdef WITH_AMDGPU_PRO
  delete pro_device_;
#endif

  // Release cached map targets
  for (uint i = 0; mapCache_ != nullptr && i < mapCache_->size(); ++i) {
    if ((*mapCache_)[i] != nullptr) {
      (*mapCache_)[i]->release();
    }
  }
  delete mapCache_;
  delete mapCacheOps_;

  if (nullptr != p2p_stage_) {
    p2p_stage_->release();
    p2p_stage_ = nullptr;
  }
  if (nullptr != mg_sync_) {
    drv::SvmBuffer::free(GlbCtx(), mg_sync_);
    mg_sync_ = nullptr;
  }
  if (glb_ctx_ != nullptr) {
      glb_ctx_->release();
      glb_ctx_ = nullptr;
  }

  // Destroy temporary buffers for read/write
  delete xferRead_;
  delete xferWrite_;

  // Destroy transfer queue
  delete xferQueue_;

  delete blitProgram_;

  if (context_ != nullptr) {
    context_->release();
  }

  delete[] p2p_agents_list_;

  if (coopHostcallBuffer_) {
    disableHostcalls(coopHostcallBuffer_);
    context().svmFree(coopHostcallBuffer_);
    coopHostcallBuffer_ = nullptr;
  }

  if (0 != prefetch_signal_.handle) {
    hsa_signal_destroy(prefetch_signal_);
  }
}

hsa_status_t Device::iterateAgentCallback(hsa_agent_t agent, void* data) {
  hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

  hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }

  if (dev_type == HSA_DEVICE_TYPE_CPU) {
    AgentInfo info = { agent, { 0 }, { 0 }, { 0 }};
    stat = hsa_amd_agent_iterate_memory_pools(agent, Device::iterateCpuMemoryPoolCallback,
                                              reinterpret_cast<void*>(&info));
    if (stat == HSA_STATUS_SUCCESS) {
      cpu_agents_.push_back(info);
    }
  } else if (dev_type == HSA_DEVICE_TYPE_GPU) {
    gpu_agents_.push_back(agent);
  }

  return stat;
}

hsa_status_t Device::loaderQueryHostAddress(const void* device, const void** host) {
  return amd_loader_ext_table.hsa_ven_amd_loader_query_host_address
      ? amd_loader_ext_table.hsa_ven_amd_loader_query_host_address(device, host)
      : HSA_STATUS_ERROR;
}

Device::XferBuffers::~XferBuffers() {
  // Destroy temporary buffer for reads
  for (const auto& buf : freeBuffers_) {
    delete buf;
  }
  freeBuffers_.clear();
}

bool Device::XferBuffers::create() {
  Memory* xferBuf = nullptr;
  bool result = false;

  // Create a buffer object
  xferBuf = new Buffer(dev(), bufSize_);

  // Try to allocate memory for the transfer buffer
  if ((nullptr == xferBuf) || !xferBuf->create()) {
    delete xferBuf;
    xferBuf = nullptr;
    LogError("Couldn't allocate a transfer buffer!");
  } else {
    result = true;
    freeBuffers_.push_back(xferBuf);
  }

  return result;
}

Memory& Device::XferBuffers::acquire() {
  Memory* xferBuf = nullptr;
  size_t listSize;

  // Lock the operations with the staged buffer list
  drv::ScopedLock l(lock_);
  listSize = freeBuffers_.size();

  // If the list is empty, then attempt to allocate a staged buffer
  if (listSize == 0) {
    // Allocate memory
    xferBuf = new Buffer(dev(), bufSize_);

    // Allocate memory for the transfer buffer
    if ((nullptr == xferBuf) || !xferBuf->create()) {
      delete xferBuf;
      xferBuf = nullptr;
      LogError("Couldn't allocate a transfer buffer!");
    } else {
      ++acquiredCnt_;
    }
  }

  if (xferBuf == nullptr) {
    xferBuf = *(freeBuffers_.begin());
    freeBuffers_.erase(freeBuffers_.begin());
    ++acquiredCnt_;
  }

  return *xferBuf;
}

void Device::XferBuffers::release(VirtualGPU& gpu, Memory& buffer) {
  // Make sure buffer isn't busy on the current VirtualGPU, because
  // the next aquire can come from different queue
  //    buffer.wait(gpu);
  // Lock the operations with the staged buffer list
  drv::ScopedLock l(lock_);
  freeBuffers_.push_back(&buffer);
  --acquiredCnt_;
}

bool Device::init() {
  ClPrint(drv::LOG_INFO, drv::LOG_INIT, "Initializing HSA stack.");

  // Initialize the compiler
  if (!initCompiler(offlineDevice_)) {
    return false;
  }

  if (HSA_STATUS_SUCCESS != hsa_init()) {
    LogError("hsa_init failed.");
    return false;
  }

  hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(amd_loader_ext_table),
                                       &amd_loader_ext_table);

  if (HSA_STATUS_SUCCESS != hsa_iterate_agents(iterateAgentCallback, nullptr)) {
    return false;
  }

  std::string ordinals = drv::IS_HIP ? ((HIP_VISIBLE_DEVICES[0] != '\0') ?
                         HIP_VISIBLE_DEVICES : CUDA_VISIBLE_DEVICES)
                         : GPU_DEVICE_ORDINAL;
  if (ordinals[0] != '\0') {
    size_t pos = 0;
    std::vector<hsa_agent_t> valid_agents;
    std::set<size_t> valid_indexes;
    do {
      size_t end;
      bool deviceIdValid = true;
      end = ordinals.find_first_of(',', pos);
      if (end == std::string::npos) {
        end = ordinals.size();
      }
      std::string strIndex = ordinals.substr(pos, end - pos);
      int index = atoi(strIndex.c_str());
      if (index < 0 ||
          static_cast<size_t>(index) >= gpu_agents_.size() ||
          strIndex != std::to_string(index)) {
        deviceIdValid = false;
      }

      if (!deviceIdValid) {
        // Exit the loop as anything to the right of invalid deviceId
        // has to be discarded
        break;
      } else {
        if (valid_indexes.find(index) == valid_indexes.end()) {
          valid_agents.push_back(gpu_agents_[index]);
          valid_indexes.insert(index);
        }
      }
      pos = end + 1;
    } while (pos < ordinals.size());
    gpu_agents_ = valid_agents;
  }

  for (auto agent : gpu_agents_) {
    std::unique_ptr<Device> roc_device(new Device(agent));
    if (!roc_device) {
      LogError("Error creating new instance of Device on then heap.");
      continue;
    }

    if (!roc_device->create()) {
      LogError("Error creating new instance of Device.");
      continue;
    }

    // Setup System Memory to be Non-Coherent per user
    // request via environment variable. By default the
    // System Memory is setup to be Coherent
    if (roc_device->settings().enableNCMode_) {
      hsa_status_t err = hsa_amd_coherency_set_type(agent, HSA_AMD_COHERENCY_TYPE_NONCOHERENT);
      if (err != HSA_STATUS_SUCCESS) {
        LogError("Unable to set NC memory policy!");
        continue;
      }
    }

    // Check to see if a global CU mask is requested
    if (drv::IS_HIP && ROC_GLOBAL_CU_MASK[0] != '\0') {
      roc_device->getGlobalCUMask(ROC_GLOBAL_CU_MASK);
    }

    roc_device.release()->registerDevice();
  }

  if (0 != Device::numDevices(CL_DEVICE_TYPE_GPU, false)) {
    // Loop through all available devices
    for (auto device1: Device::devices()) {
      // Find all agents that can have access to the current device
      for (auto agent: static_cast<Device*>(device1)->p2pAgents()) {
        // Find cl_device_id associated with the current agent
        for (auto device2: Device::devices()) {
          if (agent.handle == static_cast<Device*>(device2)->getBackendDevice().handle) {
            // Device2 can have access to device1
            device2->p2pDevices_.push_back(as_cl(device1));
            device1->p2p_access_devices_.push_back(device2);
          }
        }
      }
    }
  }

  return true;
}

extern const char* SchedulerSourceCode;
extern const char* GwsInitSourceCode;
extern const char* rocBlitLinearSourceCode;

void Device::tearDown() {
  NullDevice::tearDown();
  hsa_shut_down();
}

bool Device::create() {
  char agent_name[64] = {0};
  if (HSA_STATUS_SUCCESS != hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_NAME, agent_name)) {
    LogError("Unable to get HSA device name");
    return false;
  }

  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CHIP_ID,
                         &pciDeviceId_)) {
    LogPrintfError("Unable to get PCI ID of HSA device %s", agent_name);
    return false;
  }

  struct agent_isas_t {
    uint count;
    hsa_isa_t first_isa;
  } agent_isas = {0, {0}};
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_iterate_isas(_bkendDevice,
                             [](hsa_isa_t isa, void* data) {
                               agent_isas_t* agent_isas = static_cast<agent_isas_t*>(data);
                               if (agent_isas->count++ == 0) {
                                 agent_isas->first_isa = isa;
                               }
                               return HSA_STATUS_SUCCESS;
                             },
                             &agent_isas)) {
    LogPrintfError("Unable to iterate supported ISAs for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }
  if (agent_isas.count != 1) {
    LogPrintfError("HSA device %s (PCI ID %x) has %u ISAs but can only support a single ISA",
                   agent_name, pciDeviceId_, agent_isas.count);
    return false;
  }

  uint32_t isa_name_length = 0;
  if (HSA_STATUS_SUCCESS !=
      hsa_isa_get_info_alt(agent_isas.first_isa, (hsa_isa_info_t)HSA_ISA_INFO_NAME_LENGTH,
                           &isa_name_length)) {
    LogPrintfError("Unable to get ISA name length for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }

  std::vector<char> isa_name(isa_name_length + 1, '\0');
  if (HSA_STATUS_SUCCESS !=
      hsa_isa_get_info_alt(agent_isas.first_isa, (hsa_isa_info_t)HSA_ISA_INFO_NAME,
                           isa_name.data())) {
    LogPrintfError("Unable to get ISA name for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }

  const drv::Isa *isa = drv::Isa::findIsa(isa_name.data());
  if (!isa || !isa->runtimeRocSupported()) {
    LogPrintfError("Unsupported HSA device %s (PCI ID %x) for ISA %s", agent_name, pciDeviceId_,
                   isa_name.data());
    return false;
  }

  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_PROFILE, &agent_profile_)) {
    LogPrintfError("Unable to get profile for HSA device %s (PCI ID %x)", agent_name, pciDeviceId_);
    return false;
  }

  uint32_t coop_groups = 0;
  // Check cooperative groups for HIP only
  if (drv::IS_HIP &&
      (HSA_STATUS_SUCCESS !=
       hsa_agent_get_info(_bkendDevice,
                          static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES),
                          &coop_groups))) {
    LogPrintfError(
        "Unable to determine if cooperative queues are supported for HSA device %s (PCI ID %x)",
        agent_name, pciDeviceId_);
    return false;
  }

  // Create HSA settings
  assert(!settings_);
  roc::Settings* hsaSettings = new roc::Settings();
  settings_ = hsaSettings;
  if (!hsaSettings ||
      !hsaSettings->create((agent_profile_ == HSA_PROFILE_FULL), isa->versionMajor(),
                           isa->versionMinor(), isa->versionStepping(),
                           isa->xnack() == drv::Isa::Feature::Enabled,
                           coop_groups)) {
    LogPrintfError("Unable to create settings for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }

  if (!ValidateComgr()) {
    LogPrintfError("Code object manager initialization failed for HSA device %s (PCI ID %x)",
                   agent_name, pciDeviceId_);
    return false;
  }

  if (!drv::Device::create(*isa)) {
    LogPrintfError("Unable to setup device for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }

  uint32_t hsa_bdf_id = 0;
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice,
        static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID), &hsa_bdf_id)) {
    LogPrintfError("Unable to determine BFD ID for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }

  info_.deviceTopology_.pcie.type = CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD;
  info_.deviceTopology_.pcie.bus = (hsa_bdf_id & (0xFF << 8)) >> 8;
  info_.deviceTopology_.pcie.device = (hsa_bdf_id & (0x1F << 3)) >> 3;
  info_.deviceTopology_.pcie.function = (hsa_bdf_id & 0x07);
  uint32_t pci_domain_id = 0;
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice,
        static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN), &pci_domain_id)) {
    LogPrintfError("Unable to determine domain ID for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }
  info_.pciDomainID = pci_domain_id;

#ifdef WITH_AMDGPU_PRO
  // Create amdgpu-pro device interface for SSG support
  pro_device_ = IProDevice::Init(
      info_.deviceTopology_.pcie.bus,
      info_.deviceTopology_.pcie.device,
      info_.deviceTopology_.pcie.function);
  if (pro_device_ != nullptr) {
    pro_ena_ = true;
    settings_->enableExtension(ClAMDLiquidFlash);
    pro_device_->GetAsicIdAndRevisionId(&info_.pcieDeviceId_, &info_.pcieRevisionId_);
  }
#endif

  // Get Agent HDP Flush Register Memory
  hsa_amd_hdp_flush_t hdpInfo;
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice,
        static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_HDP_FLUSH), &hdpInfo)) {
    LogPrintfError("Unable to determine HDP flush info for HSA device %s", agent_name);
    return false;
  }
  info_.hdpMemFlushCntl = hdpInfo.HDP_MEM_FLUSH_CNTL;
  info_.hdpRegFlushCntl = hdpInfo.HDP_REG_FLUSH_CNTL;

  if (populateOCLDeviceConstants() == false) {
    LogPrintfError("populateOCLDeviceConstants failed for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }

  drv::Context::Info info = {0};
  std::vector<drv::Device*> devices;
  devices.push_back(this);

  // Create a dummy context
  context_ = new drv::Context(devices, info);
  if (context_ == nullptr) {
    return false;
  }

  mapCacheOps_ = new drv::Monitor("Map Cache Lock", true);
  if (nullptr == mapCacheOps_) {
    return false;
  }

  mapCache_ = new std::vector<drv::Memory*>();
  if (mapCache_ == nullptr) {
    return false;
  }
  // Use just 1 entry by default for the map cache
  mapCache_->push_back(nullptr);

  if ((glb_ctx_ == nullptr) && (gpu_agents_.size() >= 1) &&
      // Allow creation for the last device in the list.
      (gpu_agents_[gpu_agents_.size() - 1].handle == _bkendDevice.handle)) {
    std::vector<drv::Device*> devices;
    uint32_t numDevices = drv::Device::numDevices(CL_DEVICE_TYPE_GPU, false);
    // Add all PAL devices
    for (uint32_t i = 0; i < numDevices; ++i) {
      devices.push_back(drv::Device::devices()[i]);
    }
    // Add current
    devices.push_back(this);
    // Create a dummy context
    glb_ctx_ = new drv::Context(devices, info);
    if (glb_ctx_ == nullptr) {
      return false;
    }

    if ((p2p_agents_.size() < (devices.size()-1)) && (devices.size() > 1)) {
      drv::Buffer* buf = new (GlbCtx()) drv::Buffer(GlbCtx(), CL_MEM_ALLOC_HOST_PTR, kP2PStagingSize);
      if ((buf != nullptr) && buf->create()) {
        p2p_stage_ = buf;
      }
      else {
        delete buf;
        return false;
      }
    }
    // Check if sync buffer wasn't allocated yet
    if (drv::IS_HIP && mg_sync_ == nullptr) {
      mg_sync_ = reinterpret_cast<address>(drv::SvmBuffer::malloc(
          GlbCtx(), (CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS),
          kMGInfoSizePerDevice * GlbCtx().devices().size(), kMGInfoSizePerDevice));
      if (mg_sync_ == nullptr) {
        return false;
      }
    }
  }

  if (settings().stagedXferSize_ != 0) {
    // Initialize staged write buffers
    if (settings().stagedXferWrite_) {
      xferWrite_ = new XferBuffers(*this, drv::alignUp(settings().stagedXferSize_, 4 * Ki));
      if ((xferWrite_ == nullptr) || !xferWrite_->create()) {
        LogError("Couldn't allocate transfer buffer objects for read");
        return false;
      }
    }

    // Initialize staged read buffers
    if (settings().stagedXferRead_) {
      xferRead_ = new XferBuffers(*this, drv::alignUp(settings().stagedXferSize_, 4 * Ki));
      if ((xferRead_ == nullptr) || !xferRead_->create()) {
        LogError("Couldn't allocate transfer buffer objects for write");
        return false;
      }
    }
  }

  // Create signal for HMM prefetch operation on device
  if (HSA_STATUS_SUCCESS != hsa_signal_create(kInitSignalValueOne, 0, nullptr, &prefetch_signal_)) {
    return false;
  }

  // only create arena_mem_object if CPU memory is accessible.
  if (info_.hmmCpuMemoryAccessible_) {
    arena_mem_obj_ = new (context()) drv::ArenaMemory(context());
    if (!arena_mem_obj_->create(nullptr)) {
      LogError("Arena Memory Creation failed!");
      arena_mem_obj_->release();
      arena_mem_obj_ = nullptr;
    }
  }

  return true;
}

bool Device::AcquireExclusiveGpuAccess() {
  // Lock the virtual GPU list
  vgpusAccess().lock();

  // Find all available virtual GPUs and lock them
  // from the execution of commands
  for (uint idx = 0; idx < vgpus().size(); ++idx) {
    vgpus()[idx]->execution().lock();
    // Make sure a wait is done
    vgpus()[idx]->releaseGpuMemoryFence();
  }
  if (!hsa_exclusive_gpu_access_) {
    // @todo call rocr
    hsa_exclusive_gpu_access_ = true;
  }
  return true;
}

void Device::ReleaseExclusiveGpuAccess(VirtualGPU& vgpu) const {
  // Make sure the operation is done
  vgpu.releaseGpuMemoryFence();

  // Find all available virtual GPUs and unlock them
  // for the execution of commands
  for (uint idx = 0; idx < vgpus().size(); ++idx) {
    vgpus()[idx]->execution().unlock();
  }

  // Unock the virtual GPU list
  vgpusAccess().unlock();
}

bool Device::createBlitProgram() {
  bool result = true;
  std::string extraKernel;

#if defined(USE_COMGR_LIBRARY)
  if (settings().useLightning_) {
    if (drv::IS_HIP) {
      extraKernel = rocBlitLinearSourceCode;
      if (info().cooperativeGroups_) {
        extraKernel.append(GwsInitSourceCode);
      }
    }
    else {
      extraKernel = SchedulerSourceCode;
    }

  }
#endif  // USE_COMGR_LIBRARY

  blitProgram_ = new BlitProgram(context_);
  // Create blit programs
  if (blitProgram_ == nullptr || !blitProgram_->create(this, extraKernel, "")) {
    delete blitProgram_;
    blitProgram_ = nullptr;
    LogError("Couldn't create blit kernels!");
    return false;
  }

  return result;
}

device::Program* Device::createProgram(drv::Program& owner, drv::option::Options* options) {
  device::Program* program;
  if (settings().useLightning_) {
    program = new LightningProgram(*this, owner);
  } else {
    program = new HSAILProgram(*this, owner);
  }

  if (program == nullptr) {
    LogError("Memory allocation has failed!");
  }

  return program;
}

hsa_status_t Device::iterateGpuMemoryPoolCallback(hsa_amd_memory_pool_t pool, void* data) {
  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_region_segment_t segment_type = (hsa_region_segment_t)0;
  hsa_status_t stat =
      hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment_type);
  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }

  // TODO: system and device local segment
  Device* dev = reinterpret_cast<Device*>(data);
  switch (segment_type) {
    case HSA_REGION_SEGMENT_GLOBAL: {
      if (dev->settings().enableLocalMemory_) {
        uint32_t global_flag = 0;
        hsa_status_t stat =
            hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flag);
        if (stat != HSA_STATUS_SUCCESS) {
          return stat;
        }

        if ((global_flag & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0) {
          dev->gpu_fine_grained_segment_ = pool;
        } else if ((global_flag & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0) {
          dev->gpuvm_segment_ = pool;

          // If cpu agent cannot access this pool, the device does not support large bar.
          hsa_amd_memory_pool_access_t tmp{};
          hsa_amd_agent_memory_pool_get_info(
            dev->cpu_agent_,
            pool,
            HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
            &tmp);

          if (tmp == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
            dev->info_.largeBar_ = false;
          } else {
            dev->info_.largeBar_ = ROC_ENABLE_LARGE_BAR;
          }
        }

        if (dev->gpuvm_segment_.handle == 0) {
          dev->gpuvm_segment_ = pool;
        }
      }
      break;
    }
    case HSA_REGION_SEGMENT_GROUP:
      dev->group_segment_ = pool;
      break;
    default:
      break;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t Device::iterateCpuMemoryPoolCallback(hsa_amd_memory_pool_t pool, void* data) {
  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_region_segment_t segment_type = (hsa_region_segment_t)0;
  hsa_status_t stat =
      hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment_type);
  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }
  AgentInfo* agentInfo = reinterpret_cast<AgentInfo*>(data);

  switch (segment_type) {
    case HSA_REGION_SEGMENT_GLOBAL: {
      uint32_t global_flag = 0;
      stat = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                          &global_flag);
      if (stat != HSA_STATUS_SUCCESS) {
        break;
      }

      if ((global_flag & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0) {
        if (agentInfo->fine_grain_pool.handle == 0) {
          agentInfo->fine_grain_pool = pool;
        } else if ((global_flag & HSA_REGION_GLOBAL_FLAG_KERNARG) == 0) {
          // If the fine_grain_pool was already filled, but kern_args flag was not set over-write.
          // This means this is region-1(fine_grain only), so over-write this with memory pool set
          // from "fine_grain and kern_args".
          agentInfo->fine_grain_pool = pool;
        }
        guarantee(((global_flag & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) == 0)
                  && ("Memory Segment cannot be both coarse and fine grained"));
      } else {
        // If the flag is not set to fine grained, then it is coarse_grained by default.
        agentInfo->coarse_grain_pool = pool;
        guarantee(((global_flag & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0)
                  && ("Memory Segments that are not fine grained has to be coarse grained"));
        guarantee(((global_flag & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) == 0)
                  && ("Memory Segment cannot be both coarse and fine grained"));
        guarantee(((global_flag & HSA_REGION_GLOBAL_FLAG_KERNARG) == 0)
                  && ("Coarse grained memory segment cannot have kern_args tag"));
      }

      if ((global_flag & HSA_REGION_GLOBAL_FLAG_KERNARG) != 0) {
        agentInfo->kern_arg_pool = pool;
        guarantee(((global_flag & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) == 0)
                  && ("Coarse grained memory segment cannot have kern_args tag"));
      }

      break;
    }
    default:
      break;
  }

  return stat;
}

Memory* Device::getGpuMemory(drv::Memory* mem) const {
  return static_cast<roc::Memory*>(mem->getDeviceMemory(*this));
}

Memory* Device::getRocMemory(drv::Memory* mem) const {
  return static_cast<roc::Memory*>(mem->getDeviceMemory(*this));
}


bool Device::globalFreeMemory(size_t* freeMemory) const {
  const uint TotalFreeMemory = 0;
  const uint LargestFreeBlock = 1;

  freeMemory[TotalFreeMemory] = freeMem_ / Ki;
  freeMemory[TotalFreeMemory] -= (freeMemory[TotalFreeMemory] > HIP_HIDDEN_FREE_MEM * Ki) ?
                                  HIP_HIDDEN_FREE_MEM * Ki : 0;
  // since there is no memory heap on ROCm, the biggest free block is
  // equal to total free local memory
  freeMemory[LargestFreeBlock] = freeMemory[TotalFreeMemory];

  return true;
}
DevMemory* Device::createMemory(drv::Memory& owner) const {
  DevMemory* memory = nullptr;
  if (owner.asBuffer()) {
    memory = new Buffer(*this, owner);
  } else if (owner.asImage()) {
    memory = new Image(*this, owner);
  } else {
    LogError("Unknown memory type");
  }

  if (memory == nullptr) {
    return nullptr;
  }

  bool result = memory->create();

  if (!result) {
    LogError("Failed creating memory");
    delete memory;
    return nullptr;
  }

  if (isP2pEnabled()) {
    memory->setAllowedPeerAccess(true);
  }
  // Initialize if the memory is a pipe object
  if (owner.getType() == CL_MEM_OBJECT_PIPE) {
    // Pipe initialize in order read_idx, write_idx, end_idx. Refer clk_pipe_t structure.
    // Init with 3 DWORDS for 32bit addressing and 6 DWORDS for 64bit
    size_t pipeInit[3] = { 0, 0, owner.asPipe()->getMaxNumPackets() };
    xferMgr().writeBuffer(pipeInit, *memory, drv::Coord3D(0), drv::Coord3D(sizeof(pipeInit)));
  }

  // Transfer data only if OCL context has one device.
  // Cache coherency layer will update data for multiple devices
  if (!memory->isHostMemDirectAccess() && owner.asImage() && (owner.parent() == nullptr) &&
      (owner.getMemFlags() & CL_MEM_COPY_HOST_PTR) &&
      (owner.getContext().devices().size() == 1)) {
    // To avoid recurssive call to Device::createMemory, we perform
    // data transfer to the view of the image
    drv::Image* imageView = owner.asImage()->createView(owner.getContext(),
        owner.asImage()->getImageFormat(), xferQueue());

    if (imageView == nullptr) {
      LogError("[OCL] Fail to allocate view of image object");
      return nullptr;
    }

    Image* devImageView = new roc::Image(static_cast<const Device&>(*this), *imageView);
    if (devImageView == nullptr) {
      LogError("[OCL] Fail to allocate device mem object for the view");
      imageView->release();
      return nullptr;
    }

    if (devImageView != nullptr && !devImageView->createView(static_cast<roc::Image&>(*memory))) {
      LogError("[OCL] Fail to create device mem object for the view");
      delete devImageView;
      imageView->release();
      return nullptr;
    }

    imageView->replaceDeviceMemory(this, devImageView);

    result = xferMgr().writeImage(owner.getHostMem(), *devImageView, drv::Coord3D(0, 0, 0),
                                  imageView->getRegion(), 0, 0, true);

    // Release host memory, since runtime copied data
    owner.setHostMem(nullptr);

    imageView->release();
  }

  // Prepin sysmem buffer for possible data synchronization between CPU and GPU
  if (!memory->isHostMemDirectAccess() &&
      // Pin memory for the parent object only
      (owner.parent() == nullptr) &&
      (owner.getHostMem() != nullptr) &&
      (owner.getSvmPtr() == nullptr)) {
    memory->pinSystemMemory(owner.getHostMem(), owner.getSize());
  }

  if (!result) {
    delete memory;
    DevLogError("Cannot Write Image \n");
    return nullptr;
  }

  return memory;
}

// ================================================================================================
void* Device::hostAlloc(size_t size, size_t alignment, MemorySegment mem_seg) const {
  void* ptr = nullptr;

  mempool_mgr_->
  hsa_amd_memory_pool_t segment{0};
  switch (mem_seg) {
    case kKernArg : {
      if (settings().fgs_kernel_arg_) {
        segment = system_kernarg_segment_;
        break;
      }
      // Falls through on else case.
    }
    case kNoAtomics :
      // If runtime disables barrier, then all host allocations must have L2 disabled
      if (system_coarse_segment_.handle != 0) {
        segment = system_coarse_segment_;
        break;
      }
      // Falls through on else case.
    case kAtomics :
      segment = system_segment_;
      break;
    default :
      guarantee(false && "Invalid Memory Segment");
      break;
  }

  assert(segment.handle != 0);
  hsa_status_t stat = hsa_amd_memory_pool_allocate(segment, size, 0, &ptr);
  ClPrint(drv::LOG_DEBUG, drv::LOG_MEM, "Allocate hsa host memory %p, size 0x%zx", ptr, size);
  if (stat != HSA_STATUS_SUCCESS) {
    LogPrintfError("Fail allocation host memory with err %d", stat);
    return nullptr;
  }

  stat = hsa_amd_agents_allow_access(gpu_agents_.size(), &gpu_agents_[0], nullptr, ptr);
  if (stat != HSA_STATUS_SUCCESS) {
    LogPrintfError("Fail hsa_amd_agents_alloc_access with err %d", stat);
    hostFree(ptr, size);
    return nullptr;
  }

  return ptr;
}

// ================================================================================================
void* Device::hostAgentAlloc(size_t size, const AgentInfo& agentInfo, bool atomics) const {
  void* ptr = nullptr;
  const hsa_amd_memory_pool_t segment =
      // If runtime disables barrier, then all host allocations must have L2 disabled
      !atomics ? (agentInfo.coarse_grain_pool.handle != 0) ?
              agentInfo.coarse_grain_pool : agentInfo.fine_grain_pool
               : agentInfo.fine_grain_pool;
  assert(segment.handle != 0);
  hsa_status_t stat = hsa_amd_memory_pool_allocate(segment, size, 0, &ptr);
  ClPrint(drv::LOG_DEBUG, drv::LOG_MEM, "Allocate hsa host memory %p, size 0x%zx", ptr, size);
  if (stat != HSA_STATUS_SUCCESS) {
    LogPrintfError("Fail allocation host memory with err %d", stat);
    return nullptr;
  }

  stat = hsa_amd_agents_allow_access(gpu_agents_.size(), &gpu_agents_[0], nullptr, ptr);
  if (stat != HSA_STATUS_SUCCESS) {
    LogPrintfError("Fail hsa_amd_agents_allow_access with err %d", stat);
    hostFree(ptr, size);
    return nullptr;
  }

  return ptr;
}

// ================================================================================================
void* Device::hostNumaAlloc(size_t size, size_t alignment, bool atomics) const {
  void* ptr = nullptr;
#ifndef ROCCLR_SUPPORT_NUMA_POLICY
  ptr = hostAlloc(size, alignment, atomics
                  ? Device::MemorySegment::kAtomics : Device::MemorySegment::kNoAtomics);
#else
  int mode = MPOL_DEFAULT;
  unsigned long nodeMask = 0;
  auto cpuCount = cpu_agents_.size();

  constexpr unsigned long maxNode = sizeof(nodeMask) * 8;
  long res = get_mempolicy(&mode, &nodeMask, maxNode, NULL, 0);
  if (res) {
    LogPrintfError("get_mempolicy failed with error %ld", res);
    return ptr;
  }
  ClPrint(drv::LOG_INFO, drv::LOG_RESOURCE,
          "get_mempolicy() succeed with mode %d, nodeMask 0x%lx, cpuCount %zu",
          mode, nodeMask, cpuCount);

  switch (mode) {
    // For details, see "man get_mempolicy".
    case MPOL_BIND:
    case MPOL_PREFERRED:
      // We only care about the first CPU node
      for (unsigned int i = 0; i < cpuCount; i++) {
        if ((1u << i) & nodeMask) {
          ptr = hostAgentAlloc(size, cpu_agents_[i], atomics);
          break;
        }
      }
      break;
    default:
      //  All other modes fall back to default mode
      ptr = hostAlloc(size, alignment, atomics
                      ? Device::MemorySegment::kAtomics : Device::MemorySegment::kNoAtomics);
  }
#endif // ROCCLR_SUPPORT_NUMA_POLICY
  return ptr;
}

void Device::hostFree(void* ptr, size_t size) const { memFree(ptr, size); }

bool Device::enableP2P(drv::Device* ptrDev) {
  assert(ptrDev != nullptr);

  Device* peerDev = static_cast<Device*>(ptrDev);
  if (std::find(enabled_p2p_devices_.begin(), enabled_p2p_devices_.end(), peerDev) ==
      enabled_p2p_devices_.end()) {
    enabled_p2p_devices_.push_back(peerDev);
    // Update access to all old allocations
    drv::MemObjMap::UpdateAccess(static_cast<drv::Device*>(this));
  }
  return true;
}

bool Device::disableP2P(drv::Device* ptrDev) {
  assert(ptrDev != nullptr);

  Device* peerDev = static_cast<Device*>(ptrDev);
  //if device is present then remove
  auto it = std::find(enabled_p2p_devices_.begin(), enabled_p2p_devices_.end(), peerDev);
  if (it != enabled_p2p_devices_.end()) {
    enabled_p2p_devices_.erase(it);
  }
  return true;
}

bool Device::deviceAllowAccess(void* ptr) const {
  std::lock_guard<std::mutex> lock(lock_allow_access_);
  if (!p2pAgents().empty()) {
    hsa_status_t stat = hsa_amd_agents_allow_access(p2pAgents().size(),
                                                    p2pAgents().data(), nullptr, ptr);
    if (stat != HSA_STATUS_SUCCESS) {
      LogError("Allow p2p access");
      return false;
    }
  }
  return true;
}

void* Device::deviceLocalAlloc(size_t size, bool atomics) const {
  const hsa_amd_memory_pool_t& pool = (atomics)? gpu_fine_grained_segment_ : gpuvm_segment_;

  if (pool.handle == 0 || gpuvm_segment_max_alloc_ == 0) {
    DevLogPrintfError("Invalid argument, pool_handle: 0x%x , max_alloc: %u \n",
                      pool.handle, gpuvm_segment_max_alloc_);
    return nullptr;
  }

  void* ptr = nullptr;
  hsa_status_t stat = hsa_amd_memory_pool_allocate(pool, size, 0, &ptr);
  ClPrint(drv::LOG_DEBUG, drv::LOG_MEM, "Allocate hsa device memory %p, size 0x%zx", ptr, size);
  if (stat != HSA_STATUS_SUCCESS) {
    LogError("Fail allocation local memory");
    return nullptr;
  }

  if (isP2pEnabled() && deviceAllowAccess(ptr) == false) {
    LogError("Allow p2p access for memory allocation");
    memFree(ptr, size);
    return nullptr;
  }
  return ptr;
}

void Device::memFree(void* ptr, size_t size) const {
  hsa_status_t stat = hsa_amd_memory_pool_free(ptr);
  ClPrint(drv::LOG_DEBUG, drv::LOG_MEM, "Free hsa memory %p", ptr);
  if (stat != HSA_STATUS_SUCCESS) {
    LogError("Fail freeing local memory");
  }
}

void Device::updateFreeMemory(size_t size, bool free) {
  if (free) {
    freeMem_ += size;
  }
  else {
    if (size > freeMem_) {
      // To avoid underflow of the freeMem_
      // This can happen if the free mem tracked is inaccurate, as some allocations can happen
      // directly via ROCr
      ClPrint(drv::LOG_ERROR, drv::LOG_ALWAYS,
             "Free memory set to zero on device 0x%lx, requested size = 0x%x, freeMem_ = 0x%x",
             this, size, freeMem_.load());
      freeMem_ = 0;
      return;
    }
    freeMem_ -= size;
  }
  ClPrint(drv::LOG_INFO, drv::LOG_MEM, "device=0x%lx, freeMem_ = 0x%x", this, freeMem_.load());
}

bool Device::IpcCreate(void* dev_ptr, size_t* mem_size, void* handle, size_t* mem_offset) const {
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;

  drv::Memory* amd_mem_obj = drv::MemObjMap::FindMemObj(dev_ptr);
  if (amd_mem_obj == nullptr) {
    DevLogPrintfError("Cannot retrieve amd_mem_obj for dev_ptr: 0x%x \n", dev_ptr);
    return false;
  }

  // Get the original pointer from the drv::Memory object
  void* orig_dev_ptr = nullptr;
  if (amd_mem_obj->getSvmPtr() != nullptr) {
    orig_dev_ptr = amd_mem_obj->getSvmPtr();
  } else if (amd_mem_obj->getHostMem() != nullptr) {
    orig_dev_ptr = amd_mem_obj->getHostMem();
  } else {
    ShouldNotReachHere();
  }

  // Check if the dev_ptr is lesser than original dev_ptr
  if (orig_dev_ptr > dev_ptr) {
    //If this happens, then revisit FindMemObj logic
    DevLogPrintfError("Original dev_ptr: 0x%x cannot be greater than dev_ptr: 0x%x",
                      orig_dev_ptr, dev_ptr);
    return false;
  }

  //Calculate the memory offset from the original base ptr
  *mem_offset = reinterpret_cast<address>(dev_ptr) - reinterpret_cast<address>(orig_dev_ptr);
  *mem_size = amd_mem_obj->getSize();

  //Check if the dev_ptr is greater than memory allocated
  if (*mem_offset > *mem_size) {
    DevLogPrintfError("Memory offset: %u cannot be greater than size of "
                      "original memory allocated: %u", *mem_size, *mem_offset);
    return false;
  }

  // Pass the pointer and memory size to retrieve the handle
  hsa_status = hsa_amd_ipc_memory_create(orig_dev_ptr, drv::alignUp(*mem_size, alloc_granularity()),
                                         reinterpret_cast<hsa_amd_ipc_memory_t*>(handle));

  if (hsa_status != HSA_STATUS_SUCCESS) {
    LogPrintfError("Failed to create memory for IPC, failed with hsa_status: %d \n", hsa_status);
    return false;
  }

  return true;
}

bool Device::IpcAttach(const void* handle, size_t mem_size, size_t mem_offset,
                       unsigned int flags, void** dev_ptr) const {
  drv::Memory* amd_mem_obj = nullptr;
  void* orig_dev_ptr = nullptr;

  // Retrieve the devPtr from the handle
  hsa_status_t hsa_status =
      hsa_amd_ipc_memory_attach(reinterpret_cast<const hsa_amd_ipc_memory_t*>(handle),
                                mem_size, (1 + p2p_agents_.size()), p2p_agents_list_,
                                &orig_dev_ptr);

  if (hsa_status != HSA_STATUS_SUCCESS) {
    LogPrintfError("HSA failed to attach IPC memory with status: %d \n", hsa_status);
    return false;
  }

  amd_mem_obj = drv::MemObjMap::FindMemObj(orig_dev_ptr);
  if (amd_mem_obj == nullptr) {

    // Memory does not exist, create an amd Memory object for the pointer
    amd_mem_obj = new (context()) drv::Buffer(context(), flags, mem_size, orig_dev_ptr);
    if (amd_mem_obj == nullptr) {
      LogError("failed to create a mem object!");
      return false;
    }

    if (!amd_mem_obj->create(nullptr)) {
      LogError("failed to create a svm hidden buffer!");
      amd_mem_obj->release();
      return false;
    }

    // Add the original mem_ptr to the MemObjMap with newly created amd_mem_obj
    drv::MemObjMap::AddMemObj(orig_dev_ptr, amd_mem_obj);

  } else {
    //Memory already exists, just retain the old one.
    amd_mem_obj->retain();
  }

  //Make sure the mem_offset doesnt overflow the allocated memory
  guarantee((mem_offset < mem_size) && "IPC mem offset greater than allocated size");

  // Return orig_dev_ptr
  *dev_ptr = reinterpret_cast<address>(orig_dev_ptr);

  return true;
}

bool Device::IpcDetach (void* dev_ptr) const {
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;

  drv::Memory* amd_mem_obj = drv::MemObjMap::FindMemObj(dev_ptr);
  if (amd_mem_obj == nullptr) {
    DevLogPrintfError("Memory object for the ptr: 0x%x cannot be null \n", dev_ptr);
    return false;
  }

  // Get the original pointer from the drv::Memory object
  void* orig_dev_ptr = nullptr;
  if (amd_mem_obj->getSvmPtr() != nullptr) {
    orig_dev_ptr = amd_mem_obj->getSvmPtr();
  } else if (amd_mem_obj->getHostMem() != nullptr) {
    orig_dev_ptr = amd_mem_obj->getHostMem();
  } else {
    ShouldNotReachHere();
  }

  if (amd_mem_obj->release() == 0) {
    drv::MemObjMap::RemoveMemObj(orig_dev_ptr);

    // Detach the memory from HSA
    hsa_status = hsa_amd_ipc_memory_detach(orig_dev_ptr);
    if (hsa_status != HSA_STATUS_SUCCESS) {
      LogPrintfError("HSA failed to detach memory with status: %d \n", hsa_status);
      return false;
    }
  }

  return true;
}

// ================================================================================================
void* Device::svmAlloc(drv::Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags,
                       void* svmPtr) const {
  drv::Memory* mem = nullptr;

  if (nullptr == svmPtr) {
    // create a hidden buffer, which will allocated on the device later
    mem = new (context) drv::Buffer(context, flags, size,
                reinterpret_cast<void*>(drv::Memory::MemoryType::kSvmMemoryPtr));
    if (mem == nullptr) {
      LogError("failed to create a svm mem object!");
      return nullptr;
    }

    if (!mem->create(nullptr)) {
      LogError("failed to create a svm hidden buffer!");
      mem->release();
      return nullptr;
    }
    // if the device supports SVM FGS, return the committed CPU address directly.
    Memory* gpuMem = getRocMemory(mem);

    if (mem->getSvmPtr() != nullptr) {
      // add the information to context so that we can use it later.
      drv::MemObjMap::AddMemObj(mem->getSvmPtr(), mem);
    }
    svmPtr = mem->getSvmPtr();
  } else {
    // Find the existing drv::mem object
    mem = drv::MemObjMap::FindMemObj(svmPtr);
    if (nullptr == mem) {
      DevLogPrintfError("Cannot find svm_ptr: 0x%x \n", svmPtr);
      return nullptr;
    }

    svmPtr = mem->getSvmPtr();
  }

  return svmPtr;
}

// ================================================================================================
bool Device::SetSvmAttributesInt(const void* dev_ptr, size_t count,
                              drv::MemoryAdvice advice, bool first_alloc, bool use_cpu) const {
  if ((settings().hmmFlags_ & Settings::Hmm::EnableSvmTracking) && !first_alloc) {
    drv::Memory* svm_mem = drv::MemObjMap::FindMemObj(dev_ptr);
    if ((nullptr == svm_mem) || ((svm_mem->getMemFlags() & CL_MEM_ALLOC_HOST_PTR) == 0) ||
        // Validate the range of provided memory
        ((svm_mem->getSize() - (reinterpret_cast<const_address>(dev_ptr) -
          reinterpret_cast<address>(svm_mem->getSvmPtr()))) < count)) {
      LogPrintfError("SetSvmAttributes received unknown memory for update: %p!", dev_ptr);
      return false;
    }
  }
  if (info().hmmSupported_) {
    std::vector<hsa_amd_svm_attribute_pair_t> attr;

    switch (advice) {
      case drv::MemoryAdvice::SetReadMostly:
        attr.push_back({HSA_AMD_SVM_ATTRIB_READ_MOSTLY, true});
        break;
      case drv::MemoryAdvice::UnsetReadMostly:
        attr.push_back({HSA_AMD_SVM_ATTRIB_READ_MOSTLY, false});
        break;
      case drv::MemoryAdvice::SetPreferredLocation:
        if (use_cpu) {
          attr.push_back({HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, getCpuAgent().handle});
        } else {
          attr.push_back({HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, getBackendDevice().handle});
        }
        break;
      case drv::MemoryAdvice::UnsetPreferredLocation:
        // @note: 0 may cause a failure on old runtimes
        attr.push_back({HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, 0});
        break;
      case drv::MemoryAdvice::SetAccessedBy: {
        const uint64_t attrib = (first_alloc) ? HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE :
                                                HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE;
        if (use_cpu) {
          attr.push_back({attrib, getCpuAgent().handle});
        } else {
          if (first_alloc) {
            // Provide access to all possible devices.
            //! @note: HMM should support automatic page table update with xnack enabled,
            //! but currently it doesn't and runtime explicitly enables access from all devices
            for (const auto dev : devices()) {
              // Skip null devices
              if (static_cast<Device*>(dev)->getBackendDevice().handle != 0) {
                attr.push_back({attrib, static_cast<Device*>(dev)->getBackendDevice().handle});
              }
            }
          } else {
            attr.push_back({attrib, getBackendDevice().handle});
          }
        }
        break;
      }
      case drv::MemoryAdvice::UnsetAccessedBy:
        // When unsetting we should use HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE for the agent
        attr.push_back({HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE, getBackendDevice().handle});
        break;
      case drv::MemoryAdvice::SetCoarseGrain:
        attr.push_back({HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG, HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED});
        break;
      case drv::MemoryAdvice::UnsetCoarseGrain:
        attr.push_back({HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG, HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED});
        break;
      default:
        return false;
      break;
    }

    hsa_status_t status = hsa_amd_svm_attributes_set(const_cast<void*>(dev_ptr), count,
                                                    attr.data(), attr.size());
    if (status != HSA_STATUS_SUCCESS) {
      LogPrintfError("hsa_amd_svm_attributes_set() failed. Advice: %d, status: %d", advice, status);
      return false;
    }
  } else {
    LogWarning("hsa_amd_svm_attributes_set() is ignored, because no HMM support");
  }
  return true;
}

// ================================================================================================
bool Device::SetSvmAttributes(const void* dev_ptr, size_t count,
                              drv::MemoryAdvice advice, bool use_cpu) const {
  constexpr bool kFirstAlloc = false;
  return SetSvmAttributesInt(dev_ptr, count, advice, kFirstAlloc, use_cpu);
}

// ================================================================================================
bool Device::GetSvmAttributes(void** data, size_t* data_sizes, int* attributes,
                              size_t num_attributes, const void* dev_ptr, size_t count) const {
  if (settings().hmmFlags_ & Settings::Hmm::EnableSvmTracking) {
    drv::Memory* svm_mem = drv::MemObjMap::FindMemObj(dev_ptr);
    if ((nullptr == svm_mem) || ((svm_mem->getMemFlags() & CL_MEM_ALLOC_HOST_PTR) == 0) ||
        // Validate the range of provided memory
        ((svm_mem->getSize() - (reinterpret_cast<const_address>(dev_ptr) -
          reinterpret_cast<address>(svm_mem->getSvmPtr()))) < count)) {
      LogPrintfError("GetSvmAttributes received unknown memory %p for state!", dev_ptr);
      return false;
    }
  }

  hsa_amd_pointer_info_t ptr_info = {};
  for (size_t i = 0; i < num_attributes; ++i) {
    if (attributes[i] == drv::MemRangeAttribute::CoherencyMode) {
      ptr_info.size = sizeof(hsa_amd_pointer_info_t);
      // Query ptr type to see if it's a HMM allocation
      hsa_status_t status = hsa_amd_pointer_info(
        const_cast<void*>(dev_ptr), &ptr_info, nullptr, nullptr, nullptr);
      // The call shoudl never fail in ROCR, but just check for an error and continue
      if (status != HSA_STATUS_SUCCESS) {
        LogError("hsa_amd_pointer_info() failed");
      }
      // Check if it's a legacy non-HMM allocation and update query
      if (ptr_info.type != HSA_EXT_POINTER_TYPE_UNKNOWN) {
        if (ptr_info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
          *reinterpret_cast<uint32_t*>(data[i]) = HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED;
        } else if (ptr_info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
          *reinterpret_cast<uint32_t*>(data[i]) = HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED;
        }
      }
    }
  }

  if (info().hmmSupported_) {
    uint32_t accessed_by = 0;
    std::vector<hsa_amd_svm_attribute_pair_t> attr;

    for (size_t i = 0; i < num_attributes; ++i) {
      switch (attributes[i]) {
        case drv::MemRangeAttribute::ReadMostly:
          attr.push_back({HSA_AMD_SVM_ATTRIB_READ_MOSTLY, 0});
          break;
        case drv::MemRangeAttribute::PreferredLocation:
          attr.push_back({HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, 0});
          break;
        case drv::MemRangeAttribute::AccessedBy:
          accessed_by = attr.size();
          // Add all GPU devices into the query
          for (const auto agent : getGpuAgents()) {
            attr.push_back({HSA_AMD_SVM_ATTRIB_ACCESS_QUERY, agent.handle});
          }
          // Add CPU devices
          for (const auto agent_info : getCpuAgents()) {
            attr.push_back({HSA_AMD_SVM_ATTRIB_ACCESS_QUERY, agent_info.agent.handle});
          }
          accessed_by = attr.size() - accessed_by;
          break;
        case drv::MemRangeAttribute::LastPrefetchLocation:
          attr.push_back({HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION, 0});
          break;
        case drv::MemRangeAttribute::CoherencyMode:
          if (ptr_info.type == HSA_EXT_POINTER_TYPE_UNKNOWN) {
            attr.push_back({HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG, 0});
          }
          break;
        default:
          return false;
        break;
      }
    }

    hsa_status_t status = hsa_amd_svm_attributes_get(const_cast<void*>(dev_ptr), count,
                                                    attr.data(), attr.size());
    if (status != HSA_STATUS_SUCCESS) {
      LogError("hsa_amd_svm_attributes_get() failed");
      return false;
    }

    uint32_t idx = 0;
    uint32_t rocr_attr = 0;
    for (size_t i = 0; i < num_attributes; ++i) {
      const auto& it = attr[rocr_attr];
      switch (attributes[i]) {
        case drv::MemRangeAttribute::ReadMostly:
          if (data_sizes[idx] != sizeof(uint32_t)) {
            return false;
          }
          // Cast ROCr value into the hip format
          *reinterpret_cast<uint32_t*>(data[idx]) =
              (static_cast<uint32_t>(it.value) > 0) ? true : false;
          break;
        // The logic should be identical for the both queries
        case drv::MemRangeAttribute::PreferredLocation:
        case drv::MemRangeAttribute::LastPrefetchLocation:
          if (data_sizes[idx] != sizeof(uint32_t)) {
            return false;
          }
          *reinterpret_cast<int32_t*>(data[idx]) = static_cast<int32_t>(drv::InvalidDeviceId);
          // Find device agent returned by ROCr
          for (auto& device : devices()) {
            if (static_cast<Device*>(device)->getBackendDevice().handle == it.value) {
              *reinterpret_cast<uint32_t*>(data[idx]) = static_cast<uint32_t>(device->index());
            }
          }
          // Find CPU agent returned by ROCr
          for (auto& agent_info : getCpuAgents()) {
            if (agent_info.agent.handle == it.value) {
              *reinterpret_cast<int32_t*>(data[idx]) = static_cast<int32_t>(drv::CpuDeviceId);
            }
          }
          break;
        case drv::MemRangeAttribute::AccessedBy: {
          uint32_t entry = 0;
          uint32_t device_count = data_sizes[idx] / 4;
          // Make sure it's multiple of 4
          if (data_sizes[idx] % 4 != 0) {
            return false;
          }
          for (uint32_t att = 0; att < accessed_by; ++att) {
            const auto& it = attr[rocr_attr + att];
            if (entry >= device_count) {
              // The size of the array is less than the amount of available devices
              break;
            }
            switch (it.attribute) {
              case HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE:
              case HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS:
                break;
              case HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE:
                reinterpret_cast<int32_t*>(data[idx])[entry] =
                  static_cast<int32_t>(drv::InvalidDeviceId);
                // Find device agent returned by ROCr
                for (auto& device : devices()) {
                  if (static_cast<Device*>(device)->getBackendDevice().handle == it.value) {
                    reinterpret_cast<uint32_t*>(data[idx])[entry] =
                      static_cast<uint32_t>(device->index());
                  }
                }
                // Find CPU agent returned by ROCr
                for (auto& agent_info : getCpuAgents()) {
                  if (agent_info.agent.handle == it.value) {
                    reinterpret_cast<int32_t*>(data[idx])[entry] =
                      static_cast<int32_t>(drv::CpuDeviceId);
                  }
                }
                ++entry;
                break;
              default:
                LogWarning("Unexpected result from HSA_AMD_SVM_ATTRIB_ACCESS_QUERY");
                break;
            }
          }
          rocr_attr += accessed_by;
          for (uint32_t i = entry; i < device_count; ++i) {
            reinterpret_cast<int32_t*>(data[idx])[i] =
              static_cast<int32_t>(drv::InvalidDeviceId);
          }
          break;
        }
        case drv::MemRangeAttribute::CoherencyMode:
          if (data_sizes[idx] != sizeof(uint32_t)) {
            return false;
          }
          // if ptr is HMM alloc then overwrite the values
          if (ptr_info.type == HSA_EXT_POINTER_TYPE_UNKNOWN) {
            // Cast ROCr value into the hip format
            *reinterpret_cast<uint32_t*>(data[idx]) = static_cast<uint32_t>(it.value);
          }
          break;
        default:
          return false;
        break;
      }
      // Find the next location in the query
      ++idx;
    }
  } else if (ptr_info.type == HSA_EXT_POINTER_TYPE_UNKNOWN) {
    LogError("GetSvmAttributes() failed, because no HMM support");
    return false;
  }

  return true;
}

// ================================================================================================
bool Device::SvmAllocInit(void* memory, size_t size) const {
  drv::MemoryAdvice advice = drv::MemoryAdvice::SetAccessedBy;
  constexpr bool kFirstAlloc = true;
  if (!SetSvmAttributesInt(memory, size, advice, kFirstAlloc)) {
    return false;
  }

  if ((settings().hmmFlags_ & Settings::Hmm::EnableMallocPrefetch) == 0) {
    return true;
  }

  if (info().hmmSupported_) {
    // Initialize signal for the barrier
    hsa_signal_store_relaxed(prefetch_signal_, kInitSignalValueOne);

    // Initiate a prefetch command which should force memory update in HMM
    hsa_status_t status = hsa_amd_svm_prefetch_async(memory, size, getBackendDevice(),
                                                     0, nullptr, prefetch_signal_);
    if (status != HSA_STATUS_SUCCESS) {
      LogError("hsa_amd_svm_prefetch_async() failed");
      return false;
    }

    // Wait for the prefetch
    if (!WaitForSignal(prefetch_signal_)) {
      LogError("Barrier packet submission failed");
      return false;
    }
  } else {
    LogWarning("Early prefetch failed, because no HMM support");
  }

  return true;
}

// ================================================================================================
void Device::svmFree(void* ptr) const {
  drv::Memory* svmMem = drv::MemObjMap::FindMemObj(ptr);
  if (nullptr != svmMem) {
    drv::MemObjMap::RemoveMemObj(svmMem->getSvmPtr());
    svmMem->release();
  }
}

// ================================================================================================
VirtualGPU* Device::xferQueue() const {
  if (!xferQueue_) {
    // Create virtual device for internal memory transfer
    Device* thisDevice = const_cast<Device*>(this);
    thisDevice->xferQueue_ = reinterpret_cast<VirtualGPU*>(thisDevice->createVirtualDevice());
    if (!xferQueue_) {
      LogError("Couldn't create the device transfer manager!");
      return nullptr;
    }
  }
  xferQueue_->enableSyncBlit();
  return xferQueue_;
}

// ================================================================================================
bool Device::SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput,
  cl_set_device_clock_mode_output_amd* pSetClockModeOutput) {
  bool result = true;
  return result;
}

// ================================================================================================
bool Device::IsHwEventReady(const drv::Event& event, bool wait) const {
  void* hw_event = (event.NotifyEvent() != nullptr) ?
    event.NotifyEvent()->HwEvent() : event.HwEvent();
  if (hw_event == nullptr) {
    ClPrint(drv::LOG_INFO, drv::LOG_SIG, "No HW event");
    return false;
  } else if (wait) {
    return WaitForSignal(reinterpret_cast<ProfilingSignal*>(hw_event)->signal_, ActiveWait());
  }
  static constexpr bool Timeout = true;
  return WaitForSignal<Timeout>(reinterpret_cast<ProfilingSignal*>(hw_event)->signal_);
}

// ================================================================================================
static void callbackQueue(hsa_status_t status, hsa_queue_t* queue, void* data) {
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    // Abort on device exceptions.
    const char* errorMsg = 0;
    hsa_status_string(status, &errorMsg);
    ClPrint(drv::LOG_NONE, drv::LOG_ALWAYS,
            "Device::callbackQueue aborting with error : %s code: 0x%x", errorMsg, status);
    abort();
  }
}

// ================================================================================================
hsa_queue_t* Device::getQueueFromPool(const uint qIndex) {
  if (qIndex < QueuePriority::Total && queuePool_[qIndex].size() > 0) {
    typedef decltype(queuePool_)::value_type::const_reference PoolRef;
    auto lowest = std::min_element(queuePool_[qIndex].begin(),
        queuePool_[qIndex].end(), [] (PoolRef A, PoolRef B) {
          return A.second.refCount < B.second.refCount;
        });
    ClPrint(drv::LOG_INFO, drv::LOG_QUEUE,
        "selected queue with least refCount: %p (%d)", lowest->first,
        lowest->second.refCount);
    lowest->second.refCount++;
    return lowest->first;
  } else {
    return nullptr;
  }
}

hsa_queue_t* Device::acquireQueue(uint32_t queue_size_hint, bool coop_queue,
                                  const std::vector<uint32_t>& cuMask,
                                  drv::CommandQueue::Priority priority) {
  assert(queuePool_[QueuePriority::Low].size() <= GPU_MAX_HW_QUEUES ||
         queuePool_[QueuePriority::Normal].size() <= GPU_MAX_HW_QUEUES ||
         queuePool_[QueuePriority::High].size() <= GPU_MAX_HW_QUEUES);

  ClPrint(drv::LOG_INFO, drv::LOG_QUEUE, "number of allocated hardware queues with low priority: %d,"
      " with normal priority: %d, with high priority: %d, maximum per priority is: %d",
      queuePool_[QueuePriority::Low].size(),
      queuePool_[QueuePriority::Normal].size(),
      queuePool_[QueuePriority::High].size(), GPU_MAX_HW_QUEUES);

  hsa_amd_queue_priority_t queue_priority;
  uint qIndex;
  switch (priority) {
    case drv::CommandQueue::Priority::Low:
      queue_priority = HSA_AMD_QUEUE_PRIORITY_LOW;
      qIndex = QueuePriority::Low;
      break;
    case drv::CommandQueue::Priority::High:
      queue_priority = HSA_AMD_QUEUE_PRIORITY_HIGH;
      qIndex = QueuePriority::High;
      break;
    case drv::CommandQueue::Priority::Normal:
    case drv::CommandQueue::Priority::Medium:
    default:
      queue_priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
      qIndex = QueuePriority::Normal;
      break;
  }

  // If we have reached the max number of queues, reuse an existing queue with the matching queue priority,
  // choosing the one with the least number of users.
  // Note: Don't attempt to reuse the cooperative queue, since it's single per device
  if (!coop_queue && (cuMask.size() == 0) && (queuePool_[qIndex].size() == GPU_MAX_HW_QUEUES)) {
    return getQueueFromPool(qIndex);
  }

  // Else create a new queue. This also includes the initial state where there
  // is no queue.
  uint32_t queue_max_packets = 0;
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max_packets)) {
    DevLogError("Cannot get hsa agent info \n");
    return nullptr;
  }
  auto queue_size = (queue_max_packets < queue_size_hint) ? queue_max_packets : queue_size_hint;

  hsa_queue_t* queue;
  auto queue_type = HSA_QUEUE_TYPE_MULTI;

  // Enable cooperative queue for the device queue
  if (coop_queue) {
    queue_type = HSA_QUEUE_TYPE_COOPERATIVE;
  }

  while (hsa_queue_create(_bkendDevice, queue_size, queue_type, callbackQueue, this,
                          std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max(),
                          &queue) != HSA_STATUS_SUCCESS) {
    queue_size >>= 1;
    if (queue_size < 64) {
      // if a queue with the same requested priority available from the pool, returns it here
      if (!coop_queue && (cuMask.size() == 0) && (queuePool_[qIndex].size() > 0)) {
        return getQueueFromPool(qIndex);
      }
      DevLogError("Device::acquireQueue: hsa_queue_create failed!");
      return nullptr;
    }
  }

  // default priority is normal so no need to set it again
  if (queue_priority != HSA_AMD_QUEUE_PRIORITY_NORMAL) {
    hsa_status_t st =  hsa_amd_queue_set_priority(queue, queue_priority);
    if (st != HSA_STATUS_SUCCESS) {
      DevLogError("Device::acquireQueue: hsa_amd_queue_set_priority failed!");
      hsa_queue_destroy(queue);
      return nullptr;
    }
  }

  ClPrint(drv::LOG_INFO, drv::LOG_QUEUE, "created hardware queue %p with size %d with priority %d,"
      " cooperative: %i", queue, queue_size, queue_priority, coop_queue);

  hsa_amd_profiling_set_profiler_enabled(queue, 1);
  if (cuMask.size() != 0 || info_.globalCUMask_.size() != 0) {
    std::stringstream ss;
    ss << std::hex;
    std::vector<uint32_t> mask = {};

    // handle scenarios where cuMask (custom-defined), globalCUMask_ or both are valid and
    // fill the final mask which will be appiled to the current queue
    if (cuMask.size() != 0 && info_.globalCUMask_.size() == 0) {
      mask = cuMask;
    } else if (cuMask.size() != 0 && info_.globalCUMask_.size() != 0) {
      for (unsigned int i = 0; i < std::min(cuMask.size(), info_.globalCUMask_.size()); i++) {
        mask.push_back(cuMask[i] & info_.globalCUMask_[i]);
      }
      // check to make sure after ANDing cuMask (custom-defined) with global
      //CU mask, we have non-zero mask, oterwise just apply global CU mask
      bool zeroCUMask = true;
      for (auto m : mask) {
        if (m != 0) {
          zeroCUMask = false;
          break;
        }
      }
      if (zeroCUMask) {
        mask = info_.globalCUMask_;
      }
    } else {
      mask = info_.globalCUMask_;
    }


    for (int i = mask.size() - 1; i >= 0; i--) {
      ss << mask[i];
    }
    ClPrint(drv::LOG_INFO, drv::LOG_QUEUE, "setting CU mask 0x%s for hardware queue %p",
            ss.str().c_str(), queue);

    hsa_status_t status = hsa_amd_queue_cu_set_mask(queue, mask.size() * 32, mask.data());
    if (status != HSA_STATUS_SUCCESS) {
      DevLogError("Device::acquireQueue: hsa_amd_queue_cu_set_mask failed!");
      hsa_queue_destroy(queue);
      return nullptr;
    }
    if (cuMask.size() != 0) {
      // add queues with custom CU mask into their special pool to keep track
      // of mapping of these queues to their associated queueInfo (i.e., hostcall buffers)
      auto result = queueWithCUMaskPool_[qIndex].emplace(std::make_pair(queue, QueueInfo()));
      assert(result.second && "QueueInfo already exists");
      auto& qInfo = result.first->second;
      qInfo.refCount = 1;

      return queue;
    }
  }

  if (coop_queue) {
    // Skip queue recycling for cooperative queues, since it should be just one
    // per device.
    return queue;
  }
  auto result = queuePool_[qIndex].emplace(std::make_pair(queue, QueueInfo()));
  assert(result.second && "QueueInfo already exists");
  auto &qInfo = result.first->second;
  qInfo.refCount = 1;
  return queue;
}

void Device::releaseQueue(hsa_queue_t* queue, const std::vector<uint32_t>& cuMask) {
  for (auto& it : cuMask.size() == 0 ? queuePool_ : queueWithCUMaskPool_) {
    auto qIter = it.find(queue);
    if (qIter != it.end()) {
      auto &qInfo = qIter->second;
      assert(qInfo.refCount > 0);
      qInfo.refCount--;
      if (qInfo.refCount != 0) {
        return;
      }
      ClPrint(drv::LOG_INFO, drv::LOG_QUEUE,
          "deleting hardware queue %p with refCount 0", queue);

      if (qInfo.hostcallBuffer_) {
        ClPrint(drv::LOG_INFO, drv::LOG_QUEUE,
            "deleting hostcall buffer %p for hardware queue %p",
            qInfo.hostcallBuffer_, queue);
        disableHostcalls(qInfo.hostcallBuffer_);
        context().svmFree(qInfo.hostcallBuffer_);
      }

      ClPrint(drv::LOG_INFO, drv::LOG_QUEUE,
          "deleting hardware queue %p with refCount 0", queue);
      it.erase(qIter);
      break;
    }
  }
  hsa_queue_destroy(queue);
}

void* Device::getOrCreateHostcallBuffer(hsa_queue_t* queue, bool coop_queue,
                                        const std::vector<uint32_t>& cuMask) {
  decltype(queuePool_)::value_type::iterator qIter;

  if (!coop_queue) {
    for (auto &it : cuMask.size() == 0 ? queuePool_ : queueWithCUMaskPool_) {
      qIter = it.find(queue);
      if (qIter != it.end()) {
        break;
      }
    }
    if (cuMask.size() == 0) {
      assert(qIter != queuePool_[QueuePriority::High].end());
    } else {
      assert(qIter != queueWithCUMaskPool_[QueuePriority::High].end());
    }

    if (qIter->second.hostcallBuffer_) {
      return qIter->second.hostcallBuffer_;
    }
  } else {
    if (coopHostcallBuffer_) {
      return coopHostcallBuffer_;
    }
  }

  // The number of packets required in each buffer is at least equal to the
  // maximum number of waves supported by the device.
  auto wavesPerCu = info().maxThreadsPerCU_ / info().wavefrontWidth_;
  auto numPackets = info().maxComputeUnits_ * wavesPerCu;

  auto size = getHostcallBufferSize(numPackets);
  auto align = getHostcallBufferAlignment();

  void* buffer = context().svmAlloc(size, align, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS);
  if (!buffer) {
    ClPrint(drv::LOG_ERROR, drv::LOG_QUEUE,
            "Failed to create hostcall buffer for hardware queue %p", queue);
    return nullptr;
  }
  ClPrint(drv::LOG_INFO, drv::LOG_QUEUE, "Created hostcall buffer %p for hardware queue %p", buffer,
          queue);
  if (!coop_queue) {
    qIter->second.hostcallBuffer_ = buffer;
  } else {
    coopHostcallBuffer_ = buffer;
  }
  if (!enableHostcalls(*this, buffer, numPackets)) {
    ClPrint(drv::LOG_ERROR, drv::LOG_QUEUE, "Failed to register hostcall buffer %p with listener",
            buffer);
    return nullptr;
  }
  return buffer;
}

bool Device::findLinkInfo(const drv::Device& other_device,
                          std::vector<LinkAttrType>* link_attrs) {
  return findLinkInfo((static_cast<const roc::Device*>(&other_device))->gpuvm_segment_,
                       link_attrs);
}

bool Device::findLinkInfo(const hsa_amd_memory_pool_t& pool,
                          std::vector<LinkAttrType>* link_attrs) {

  if ((!pool.handle) || (link_attrs == nullptr)) {
    return false;
  }

  // Retrieve the hops between 2 devices.
  int32_t hops = 0;
  hsa_status_t hsa_status = hsa_amd_agent_memory_pool_get_info(_bkendDevice, pool,
                            HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS, &hops);

  if (hsa_status != HSA_STATUS_SUCCESS) {
    DevLogPrintfError("Cannot get hops info, hsa failed with status: %d", hsa_status);
    return false;
  }

  if (hops < 0) {
    return false;
  }

  // The pool is on its agent
  if (hops == 0) {
    for (auto& link_attr : (*link_attrs)) {
      switch (link_attr.first) {
        case kLinkLinkType: {
          // No link, so type is meaningless,
          // caller should ignore it
          link_attr.second = -1;
          break;
        }
        case kLinkHopCount: {
          // no hop
          link_attr.second = 0;
          break;
        }
        case kLinkDistance: {
          // distance is zero, if no hops
          link_attr.second = 0;
          break;
        }
        case kLinkAtomicSupport: {
          // atomic support if its on the same agent
          link_attr.second = 1;
          break;
        }
        default: {
          DevLogPrintfError("Invalid LinkAttribute: %d ", link_attr.first);
          return false;
        }
      }
    }
    return true;
  }

  // Retrieve link info on the pool.
  std::vector<hsa_amd_memory_pool_link_info_t> link_info(hops);
  hsa_status = hsa_amd_agent_memory_pool_get_info(_bkendDevice, pool,
               HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO, link_info.data());

  if (hsa_status != HSA_STATUS_SUCCESS) {
    DevLogPrintfError("Cannot retrieve link info, hsa failed with status: %d", hsa_status);
    return false;
  }

  for (auto& link_attr : (*link_attrs)) {
    switch (link_attr.first) {
      case kLinkLinkType: {
        link_attr.second = static_cast<int32_t>(link_info[0].link_type);
        break;
      }
      case kLinkHopCount: {
        uint32_t distance = 0;
        // Because of Rocrs limitation hops is set to 1 always between two different devices
        // If Rocr Changes the behaviour revisit this logic
        for (size_t hop_idx = 0; hop_idx < static_cast<size_t>(hops); ++hop_idx) {
          distance += link_info[hop_idx].numa_distance;
        }
        uint32_t oneHopDistance
          = (link_info[0].link_type == HSA_AMD_LINK_INFO_TYPE_XGMI) ? 15 : 20;
        link_attr.second = static_cast<int32_t>(distance/oneHopDistance);
        break;
      }
      case kLinkDistance: {
        uint32_t distance = 0;
        // Sum of distances between hops
        for (size_t hop_idx = 0; hop_idx < static_cast<size_t>(hops); ++hop_idx) {
          distance += link_info[hop_idx].numa_distance;
        }
        link_attr.second = static_cast<int32_t>(distance);
        break;
      }
      case kLinkAtomicSupport: {
        // if either of the atomic is supported
        link_attr.second = static_cast<int32_t>(link_info[0].atomic_support_64bit
                                                || link_info[0].atomic_support_32bit);
        break;
      }
      default: {
        DevLogPrintfError("Invalid LinkAttribute: %d ", link_attr.first);
        return false;
      }
    }
  }

  return true;
}

// ================================================================================================
void Device::getGlobalCUMask(std::string cuMaskStr) {
  if (cuMaskStr.length() != 0) {
    std::string pre = cuMaskStr.substr(0, 2);
    if (pre.compare("0x") == 0 || pre.compare("0X") == 0) {
      cuMaskStr = cuMaskStr.substr(2, cuMaskStr.length());
    }

    int end = cuMaskStr.length();

    // the number of current physical CUs compressed in 4-bits
    size_t compPhysicalCUs = static_cast<size_t>((settings().enableWgpMode_ ?
           info_.maxComputeUnits_ * 2 : info_.maxComputeUnits_)/ 4);

    // the number of final available compute units after applying the requested CU mask
    uint32_t availCUs = 0;

    // read numCharToRead characters (8 or less) from the cuMask string each time, convert
    // it into hex, and store it into the globalCUMask_. If the length of the cuMask string
    // is more than the compressed physical available CUs, ignore the rest
    for (unsigned i = 0; i < std::min(cuMaskStr.length(), compPhysicalCUs); i += 8) {
      int numCharToRead = (i + 8 <= compPhysicalCUs) ? 8 : compPhysicalCUs - 8;
      std::string temp = cuMaskStr.substr(std::max(0, end - numCharToRead),
          std::min(numCharToRead, end));
      end -= numCharToRead;
      unsigned long ul = 0;
      try {
        ul = std::stoul(temp, 0, 16);
      } catch (const std::invalid_argument&) {
        info_.globalCUMask_ = {};
        availCUs = 0;
        break;
      }
      info_.globalCUMask_.push_back(static_cast<uint32_t>(ul));
      // count number of set bits in ul to find the number of active CUs
      // in each iteration
      while (ul) {
        ul &= (ul - 1);
        availCUs++;
      }
    }
    //update the maxComputeUnits_ based on the requested CU mask
    if (availCUs != 0 && availCUs < compPhysicalCUs * 4) {
      info_.maxComputeUnits_ = settings().enableWgpMode_ ?
      availCUs / 2 : availCUs;
    } else {
      info_.globalCUMask_ = {};
    }
  } else {
    info_.globalCUMask_ = {};
  }
}

// ================================================================================================
device::Signal* Device::createSignal() const {
  return new roc::Signal();
}

// ================================================================================================
drv::Memory* Device::GetArenaMemObj(const void* ptr, size_t& offset) {
  // If arena_mem_obj_ is null, then HMM and Xnack is disabled. Return nullptr.
  if (arena_mem_obj_ == nullptr) {
    return nullptr;
  }

  // Calculate the offset of the pointer.
  const void* dev_ptr = reinterpret_cast<void*>(arena_mem_obj_->getDeviceMemory(
                          *arena_mem_obj_->getContext().devices()[0])->virtualAddress());
  offset = reinterpret_cast<size_t>(ptr) - reinterpret_cast<size_t>(dev_ptr);

  return arena_mem_obj_;
}

// ================================================================================================
void Device::ReleaseGlobalSignal(void* signal) const {
  if (signal != nullptr) {
    reinterpret_cast<ProfilingSignal*>(signal)->release();
  }
}

// ================================================================================================
ProfilingSignal::~ProfilingSignal() {
  if (signal_.handle != 0) {
    if (hsa_signal_load_relaxed(signal_) > 0) {
      LogError("Runtime shouldn't destroy a signal that is still busy!");
      if (hsa_signal_wait_scacquire(signal_, HSA_SIGNAL_CONDITION_LT, kInitSignalValueOne,
                                    kUnlimitedWait, HSA_WAIT_STATE_BLOCKED) != 0) {
      }
    }
    hsa_signal_destroy(signal_);
  }
}


}  // namespace amd

