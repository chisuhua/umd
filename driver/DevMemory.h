#pragma once

#include "top.hpp"

//! Device-independent cache memory, base class for the device-specific
//! memories. One Memory instance refers to one or more of these.
class DevMemory : public drv::HeapObject {
 public:
  //! Resource map flags
  enum CpuMapFlags {
    CpuReadWrite = 0x00000000,  //!< Lock for CPU read/Write
    CpuReadOnly = 0x00000001,   //!< Lock for CPU read only operation
    CpuWriteOnly = 0x00000002,  //!< Lock for CPU write only operation
  };

  union SyncFlags {
    struct {
      uint skipParent_ : 1;  //!< Skip parent synchronization
      uint skipViews_ : 1;   //!< Skip views synchronization
      uint skipEntire_ : 1;  //!< Skip entire synchronization
    };
    uint value_;
    SyncFlags() : value_(0) {}
  };

  struct WriteMapInfo : public drv::HeapObject {
    drv::Coord3D origin_;  //!< Origin of the map location
    drv::Coord3D region_;  //!< Mapped region
    drv::Image* baseMip_;  //!< The base mip level for images
    union {
      struct {
        uint32_t count_ : 8;       //!< The same map region counter
        uint32_t unmapWrite_ : 1;  //!< Unmap write operation
        uint32_t unmapRead_ : 1;   //!< Unmap read operation
        uint32_t entire_ : 1;      //!< Process the entire memory
      };
      uint32_t flags_;
    };

    //! Returns the state of entire map
    bool isEntire() const { return (entire_) ? true : false; }

    //! Returns the state of map write flag
    bool isUnmapWrite() const { return (unmapWrite_) ? true : false; }

    //! Returns the state of map read flag
    bool isUnmapRead() const { return (unmapRead_) ? true : false; }

    WriteMapInfo() : origin_(0, 0, 0), region_(0, 0, 0), baseMip_(NULL), flags_(0) {}
  };

  //! Constructor (from an drv::Memory object).
  DevMemory(const Device& dev, drv::Memory& owner)
      : flags_(0), owner_(&owner), version_(0), mapMemory_(NULL), indirectMapCount_(0) {
    size_ = owner.getSize();
  }

  //! Constructor (no owner), always eager allocation.
  DevMemory(const Device& dev, size_t size)
      : flags_(0), owner_(NULL), version_(0), mapMemory_(NULL), indirectMapCount_(0), size_(size) {}
#if 0
  TODO
  //! Constructor memory for images (without global heap allocation)
  DevMemory(const Device& gpuDev,          //!< GPU device object
         drv::Memory& owner,            //!< Abstraction layer memory object
         size_t width,                  //!< Allocated memory width
         size_t height,                 //!< Allocated memory height
         size_t depth,                  //!< Allocated memory depth
         cl_image_format format,        //!< Memory format
         cl_mem_object_type imageType,  //!< CL image type
         uint mipLevels                 //!< The number of mip levels
  );

  //! Constructor memory for images (without global heap allocation)
  DevMemory(const Device& gpuDev,          //!< GPU device object
         size_t size,                   //!< Memory object size
         size_t width,                  //!< Allocated memory width
         size_t height,                 //!< Allocated memory height
         size_t depth,                  //!< Allocated memory depth
         cl_image_format format,        //!< Memory format
         cl_mem_object_type imageType,  //!< CL image type
         uint mipLevels                 //!< The number of mip levels
  );
  //! Creates the interop memory
  bool createInterop();

  enum GLResourceOP {
    GLDecompressResource = 0,  // orders the GL driver to decompress any depth-stencil or MSAA
                               // resource to be sampled by a CL kernel.
    GLInvalidateFBO  // orders the GL driver to invalidate any FBO the resource may be bound to,
                     // since the resource internal state changed.
  };
#endif
  //! Default destructor for the device memory object
  virtual ~DevMemory(){};

  // Getter for deviceMemory_
  address getDeviceMemory() const { return reinterpret_cast<address>(deviceMemory_); }

  //! Overloads the resource create method
  virtual bool create(Resource::MemoryType memType,             //!< Memory type
                      Resource::CreateParams* params = nullptr, //!< Prameters for create
                      bool forceLinear = false                  //!< Forces linear tiling for images
  );


  //! Releases virtual objects associated with this memory
  void releaseVirtual();

  //! Read the size
  size_t size() const { return size_; }

  //! Gets the owner Memory instance
  drv::Memory* owner() const { return owner_; }

  //! Immediate blocking write from device cache to owners's backing store.
  //! Marks owner as "current" by resetting the last writer to NULL.
  virtual void syncHostFromCache(SyncFlags syncFlags = SyncFlags()) {}

  //! Allocate memory for API-level maps
  virtual void* allocMapTarget(const drv::Coord3D& origin,  //!< The map location in memory
                               const drv::Coord3D& region,  //!< The map region in memory
                               uint mapFlags,               //!< Map flags
                               size_t* rowPitch = NULL,     //!< Row pitch for the mapped memory
                               size_t* slicePitch = NULL    //!< Slice for the mapped memory
  ) {
    return NULL;
  }

  bool isPersistentMapped() const { return (flags_ & PersistentMap) ? true : false; }
  void setPersistentMapFlag(bool persistentMapped) {
    if (persistentMapped == true) {
      flags_ |= PersistentMap;
    }
    else {
      flags_ &= ~PersistentMap;
    }
  }

  bool pinSystemMemory(void* hostPtr,  //!< System memory address
                               size_t size     //!< Size of allocated system memory
  );

  //! Updates device memory from the owner's host allocation
  void syncCacheFromHost(EmuGPU& gpu,  //!< Virtual GPU device object
                         //! Synchronization flags
                         DevMemory::SyncFlags syncFlags = DevMemory::SyncFlags());

  // Immediate blocking write from device cache to owners's backing store.
  // Marks owner as "current" by resetting the last writer to nullptr.
  void syncHostFromCache(SyncFlags syncFlags = SyncFlags()) override;

  //! Allocates host memory for synchronization with MGPU context
  void mgpuCacheWriteBack();

  // Releases indirect map surface
  void releaseIndirectMap() override { decIndMapCount(); }

  //! Creates a view from current resource
  DevMemory* createBufferView(
      drv::Memory& subBufferOwner  //!< The abstraction layer subbuf owner
  );

  // Decompress GL depth-stencil/MSAA resources for CL access
  // Invalidates any FBOs the resource may be bound to, otherwise the GL driver may crash.
  virtual bool processGLResource(GLResourceOP operation);

  //! Returns the interop resource for this memory object
  const DevMemory* parent() const { return parent_; }


  virtual uint64_t virtualAddress() const override { return vmAddress(); }

  //! Quick view update for managed buffers. It should avoid expensive object allocations
  void updateView(Resource* view, size_t offset, size_t size) {
    size_ = size;
    flags_ |= HostMemoryDirectAccess;
    Resource::updateView(view, offset, size);
  }

  //! Map the device memory to CPU visible
  void* cpuMap(uint flags = 0,       //!< flags for the map operation
  ) {
    drv::Image* image = owner()->asImage();
    if (image != NULL) {
      /* TODO
      *rowPitch = image->getRowPitch();
      *slicePitch = image->getSlicePitch();
      */
    }
    // Default behavior uses preallocated host mem for CPU
    return owner()->getHostMem();
  }
#if 0
  //! Map the device memory to CPU visible
  void* cpuMap( uint flags = 0,       //!< flags for the map operation
                       // Optimization for multilayer map/unmap
                       uint startLayer = 0,       //!< Start layer for multilayer map
                       uint numLayers = 0,        //!< End layer for multilayer map
                       size_t* rowPitch = NULL,   //!< Row pitch for the device memory
                       size_t* slicePitch = NULL  //!< Slice pitch for the device memory
  );
#endif
  //! Unmap the device memory
  void cpuUnmap();

  //! Saves map info for this object
  //! @note: It's not a thread safe operation, the app must implement
  //! synchronization for the multiple write maps if necessary
  void saveMapInfo(const void* mapAddress,        //!< Map cpu address
                   const drv::Coord3D origin,     //!< Origin of the map location
                   const drv::Coord3D region,     //!< Mapped region
                   uint mapFlags,                 //!< Map flags
                   bool entire,                   //!< True if the enitre memory was mapped
                   drv::Image* baseMip = nullptr  //!< The base mip level for map
  );

  drv::Memory* mapMemory() const { return mapMemory_; }

  const WriteMapInfo* writeMapInfo(const void* mapAddress) const {
    // Unmap must be serialized.
    drv::ScopedLock lock(owner()->lockMemoryOps());

    auto it = writeMapInfo_.find(mapAddress);
    if (it == writeMapInfo_.end()) {
      if (writeMapInfo_.size() == 0) {
        LogError("Unmap is a NOP!");
        return nullptr;
      }
      LogWarning("Unknown unmap signature!");
      // Get the first map info
      it = writeMapInfo_.begin();
    }
    return &it->second;
  }

  //! Clear memory object as mapped read only
  void clearUnmapInfo(const void* mapAddress) {
    // Unmap must be serialized.
    drv::ScopedLock lock(owner()->lockMemoryOps());
    auto it = writeMapInfo_.find(mapAddress);
    if (it == writeMapInfo_.end()) {
      // Get the first map info
      it = writeMapInfo_.begin();
    }
    if (--it->second.count_ == 0) {
      writeMapInfo_.erase(it);
    }
  }

  MEMORY_KIND getKind() const { return kind_; }
  const Device& dev() const { return dev_; }

  //! Returns the state of memory direct access flag
  bool isHostMemDirectAccess() const { return (flags_ & HostMemoryDirectAccess) ? true : false; }

  //! Returns the state of host memory registration flag
  bool isHostMemoryRegistered() const { return (flags_ & HostMemoryRegistered) ? true : false; }

  //! Returns the state of CPU uncached access
  bool isCpuUncached() const { return (flags_ & MemoryCpuUncached) ? true : false; }

  uint64_t virtualAddress() const { return 0; }

  uint64_t originalDeviceAddress() const { return 0; }

  //! Returns CPU pointer to HW state
  virtual const address cpuSrd() const { return nullptr; }

  bool getAllowedPeerAccess() const { return (flags_ & AllowedPeerAccess) ? true : false; }
  void setAllowedPeerAccess(bool flag) {
    if (flag == true) {
      flags_ |= AllowedPeerAccess;
    }
    else {
      flags_ &= ~AllowedPeerAccess;
    }
  }

  bool IsPersistentDirectMap() const { return (persistent_host_ptr_ != nullptr); }

  void* PersistentHostPtr() const { return persistent_host_ptr_; }


 protected:
  // bool allocateMapMemory(size_t allocationSize);

  // Decrement map count
  // void decIndMapCount() override;

  // Free / deregister device memory.
  virtual void destroy() = 0;


 protected:
  enum Flags {
    HostMemoryDirectAccess = 0x00000001,  //!< GPU has direct access to the host memory
    MapResourceAlloced = 0x00000002,      //!< Map resource was allocated
    PinnedMemoryAlloced = 0x00000004,     //!< An extra pinned resource was allocated
    SubMemoryObject = 0x00000008,         //!< Memory is sub-memory
    HostMemoryRegistered = 0x00000010,    //!< Host memory was registered
    MemoryCpuUncached = 0x00000020,       //!< Memory is uncached on CPU access(slow read)
    AllowedPeerAccess = 0x00000040,       //!< Memory can be accessed from peer
    PersistentMap = 0x00000080            //!< Map Peristent memory
  };
  uint flags_;  //!< Memory object flags

  drv::Memory* owner_;  //!< The Memory instance that we cache,
                        //!< or NULL if we're device-private workspace.
                        //
  // Pointer to the device associated with this memory object.
  const Device& dev_;

  // Pointer to the device memory. This could be in system or device local mem.
  void* deviceMemory_;

  // Pointer to the interop device memory, which has an offset from deviceMemory_
  void* interop_deviceMemory_;

  // Track if this memory is interop, lock, gart, or normal.
  MEMORY_KIND kind_;

  void* persistent_host_ptr_;  //!< Host accessible pointer for persistent memory



  volatile size_t version_;  //!< The version we're currently shadowing

  //! NB, the map data below is for an API-level map (from clEnqueueMapBuffer),
  //! not a physical map. When a memory object does not use USE_HOST_PTR we
  //! can use a remote resource and DMA, avoiding the additional CPU memcpy.
  drv::Memory* mapMemory_;            //!< Memory used as map target buffer
  volatile size_t indirectMapCount_;  //!< Number of maps
  std::unordered_map<const void*, WriteMapInfo>
      writeMapInfo_;  //!< Saved write map info for partial unmap

  //! Increment map count
  void incIndMapCount() { ++indirectMapCount_; }

  //! Decrement map count
  virtual void decIndMapCount() {}

  size_t size_;  //!< Memory size

 private:
  //! Disable default copy constructor
  DevMemory& operator=(const DevMemory&) = delete;

  //! Disable operator=
  DevMemory(const DevMemory&) = delete;
  const DevMemory* parent_;  //!< Parent memory object

  drv::Memory* pinnedMemory_;  //!< Memory used as pinned system memory
};

class Buffer : public DevMemory {
 public:
  Buffer(const Device& dev, drv::Memory& owner);
  Buffer(const Device& dev, size_t size);

  // Create device memory according to OpenCL memory flag.
  virtual bool create();

  // Recreate the device memory using new size and alignment.
  bool recreate(size_t newSize, size_t newAlignment, bool forceSystem);

 private:
  //! Disable copy constructor
  Buffer(const Buffer&);

  //! Disable operator=
  Buffer& operator=(const Buffer&);
  // Free device memory.
  void destroy();
};

#if 0
class Image : public roc::Memory {
 public:
  Image(const roc::Device& dev, drv::Memory& owner);

  virtual ~Image();

  //! Create device memory according to OpenCL memory flag.
  virtual bool create();

  //! Create an image view
  bool createView(const Memory& parent);

  //! Gets a pointer to a region of host-visible memory for use as the target
  //! of an indirect map for a given memory object
  virtual void* allocMapTarget(const drv::Coord3D& origin, const drv::Coord3D& region,
                               uint mapFlags, size_t* rowPitch, size_t* slicePitch);

  size_t getDeviceDataSize() { return deviceImageInfo_.size; }
  size_t getDeviceDataAlignment() { return deviceImageInfo_.alignment; }

  hsa_ext_image_t getHsaImageObject() const { return hsaImageObject_; }
  const hsa_ext_image_descriptor_t& getHsaImageDescriptor() const { return imageDescriptor_; }

  virtual const address cpuSrd() const { return reinterpret_cast<const address>(getHsaImageObject().handle); }

  //! Validates allocated memory for possible workarounds
  bool ValidateMemory() final;

  drv::Image* CopyImageBuffer() const { return copyImageBuffer_; }

  virtual uint64_t originalDeviceAddress() const { return reinterpret_cast<uint64_t>(originalDeviceMemory_); }
 private:
  //! Disable copy constructor
  Image(const Buffer&);

  //! Disable operator=
  Image& operator=(const Buffer&);

  // Setup an interop image
  bool createInteropImage();

  // Free / deregister device memory.
  void destroy();

  void populateImageDescriptor();

  hsa_ext_image_descriptor_t imageDescriptor_;
  hsa_access_permission_t permission_;
  hsa_ext_image_data_info_t deviceImageInfo_;
  hsa_ext_image_t hsaImageObject_;

  void* originalDeviceMemory_;
  drv::Image* copyImageBuffer_ = nullptr;
};
#endif
}

