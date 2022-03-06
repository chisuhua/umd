#pragma once


namespace drv {
class VirtualDevice;

class Device : public RuntimeObject {
  //! Transfer buffers
  class BlitBuffers : public drv::HeapObject {
   public:
    static const size_t MaxBlitBufListSize = 8;

    //! Default constructor
    BlitBuffers(const Device& device, size_t bufSize)
        : bufSize_(bufSize), acquiredCnt_(0), gpuDevice_(device) {}

    //! Default destructor
    ~BlitBuffers();

    //! Creates the blit buffers object
    bool create();

    //! Acquires an instance of the transfer buffers
    DevMemory& acquire();

    //! Releases transfer buffer
    void release(VirtualGPU& gpu,  //!< Virual GPU object used with the buffer
                 DevMemory& buffer    //!< Transfer buffer for release
                 );

    //! Returns the buffer's size for transfer
    size_t bufSize() const { return bufSize_; }

   private:
    //! Disable copy constructor
    BlitBuffers(const BlitBuffers&);

    //! Disable assignment operator
    BlitBuffers& operator=(const BlitBuffers&);

    //! Get device object
    const Device& dev() const { return gpuDevice_; }

    size_t bufSize_;                  //!< Staged buffer size
    std::list<DevMemory*> freeBuffers_;  //!< The list of free buffers
    std::atomic_uint acquiredCnt_;   //!< The total number of acquired buffers
    drv::Monitor lock_;               //!< Stgaed buffer acquire/release lock
    const Device& gpuDevice_;         //!< GPU device object
  };

 public:
  // The structures below for MGPU launch match the device library format
  struct MGSyncData {
    uint32_t w0;
    uint32_t w1;
  };

  struct MGSyncInfo {
    struct MGSyncData* mgs;
    uint32_t grid_id;
    uint32_t num_grids;
    uint64_t prev_sum;
    uint64_t all_sum;
  };

  //Attributes that could be retrived from hsa_amd_memory_pool_link_info_t.
  typedef enum LinkAttribute {
    kLinkLinkType = 0,
    kLinkHopCount,
    kLinkDistance,
    kLinkAtomicSupport
  } LinkAttribute;

  typedef enum MemorySegment {
    kNoAtomics = 0,
    kAtomics = 1,
    kKernArg = 2
  } MemorySegment;

  typedef std::pair<LinkAttribute, int32_t /* value */> LinkAttrType;

  static constexpr size_t kP2PStagingSize = 4 * Mi;
  static constexpr size_t kMGSyncDataSize = sizeof(MGSyncData);
  static constexpr size_t kMGInfoSizePerDevice = kMGSyncDataSize + sizeof(MGSyncInfo);

  typedef std::list<CommandQueue*> CommandQueues;

  struct BlitProgram : public drv::HeapObject {
    Program* program_;  //!< GPU program object
    Context* context_;  //!< A dummy context

    BlitProgram(Context* context) : program_(NULL), context_(context) {}
    ~BlitProgram();

    //! Creates blit program for this device
    bool create(Device* device,                  //!< Device object
                const std::string& extraKernel,  //!< Extra kernels from the device layer
                const std::string& extraOptions  //!< Extra compilation options
    );
  };


  Device();
  virtual ~Device();

  //! Initializes abstraction layer device object
  bool create();

  uint retain() {
    // Overwrite the RuntimeObject::retain().
    // There is an issue in the old SHOC11_DeviceMemory test on TC
    return 0u;
  }

  uint release() {
    // Overwrite the RuntimeObject::release().
    // There is an issue in the old SHOC11_DeviceMemory test on TC
    return 0u;
  }

  //! Register a device as available
  void registerDevice();

  //! Initialize the device layer (enumerate known devices)
  static bool init();

  //! Shutdown the device layer
  static void tearDown();

  static std::vector<Device*> getDevices(cl_device_type type,  //!< Device type
                                         bool offlineDevices   //!< Enable offline devices
  );

  static size_t numDevices(cl_device_type type,  //!< Device type
                           bool offlineDevices   //!< Enable offline devices
  );

  static bool getDeviceIDs(cl_device_type deviceType,  //!< Device type
                           uint32_t numEntries,         //!< Number of entries in the array
                           cl_device_id* devices,      //!< Array of the device ID(s)
                           uint32_t* numDevices,        //!< Number of available devices
                           bool offlineDevices         //!< Report offline devices
  );

  const device::Info& info() const { return info_; }

  //! Return svm support capability.
  bool svmSupport() const {
    return (info().svmCapabilities_ &
            (CL_DEVICE_SVM_COARSE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_BUFFER |
             CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)) != 0
        ? true
        : false;
  }

  //! check svm FGS support capability.
  inline bool isFineGrainedSystem(bool FGSOPT = false) const {
    return FGSOPT && (info().svmCapabilities_ & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) != 0 ? true
                                                                                      : false;
  }

  //! Return this device's type.
  cl_device_type type() const { return info().type_ & ~(CL_DEVICE_TYPE_DEFAULT); }

  //! Create a new virtual device environment.
  virtual device::VirtualDevice* createVirtualDevice(CommandQueue* queue = NULL) ;

  //! Create a program for device.
  // virtual device::Program* createProgram(drv::Program& owner, option::Options* options = NULL) ;

  //! Allocate a chunk of device memory as a cache for a CL memory object
  virtual DevMemory* createMemory(drv::Memory& owner) const ;

  //! Allocate a device sampler object
  virtual bool createSampler(const Sampler&, device::Sampler**) const ;

  //! Allocates a view object from the device memory
  virtual device::Memory* createView(
      drv::Memory& owner,           //!< Owner memory object
      const device::Memory& parent  //!< Parent device memory object for the view
      ) const ;

  ///! Allocates a device signal object
  virtual device::Signal* createSignal() const ;

  //! Gets free memory on a GPU device
  virtual bool globalFreeMemory(size_t* freeMemory  //!< Free memory information on a GPU device
                                ) const ;

  virtual bool importExtSemaphore(void** extSemaphore, const drv::Os::FileDesc& handle) ;
  virtual void DestroyExtSemaphore(void* extSemaphore) ;
  /**
   * @return True if the device has its own custom host allocator to be used
   * instead of the generic OS allocation routines
   */
  bool customHostAllocator() const { return settings().customHostAllocator_ == 1; }

  /**
   * @copydoc drv::Context::hostAlloc
   */
  virtual void* hostAlloc(size_t size, size_t alignment,
                          MemorySegment mem_seg = kNoAtomics) const ;

  virtual void hostFree(void* ptr, size_t size = 0) const;

  virtual bool deviceAllowAccess(void* dst) const ;

  void* deviceLocalAlloc(size_t size, bool atomics = false) const;
  void memFree(void* ptr, size_t size) const;

  virtual bool enableP2P(drv::Device* ptrDev);
  virtual bool disableP2P(drv::Device* ptrDev);

  virtual void* svmAlloc(Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags,
                         void* svmPtr) const ;
  virtual void svmFree(void* ptr) const ;

  /**
   * @return True if the device successfully applied the SVM attributes in HMM for device memory
   */
  virtual bool SetSvmAttributes(const void* dev_ptr, size_t count,
                                drv::MemoryAdvice advice, bool use_cpu = false) const;

  /**
   * @return True if the device successfully retrieved the SVM attributes from HMM for device memory
   */
  virtual bool GetSvmAttributes(void** data, size_t* data_sizes, int* attributes,
                                size_t num_attributes, const void* dev_ptr, size_t count) const;

  //! Validate kernel
  virtual bool validateKernel(const drv::Kernel& kernel,
                              const device::VirtualDevice* vdev,
                              bool coop_groups = false) {
    return true;
  };

  // Returns the status of HW event, associated with drv::Event
  virtual bool IsHwEventReady(
      const drv::Event& event,  //!< AMD event for HW status validation
      bool wait = false         //!< If true then forces the event completion
      ) const ;

  virtual const uint32_t getPreferredNumaNode() const { return preferred_numa_node_; }
  virtual void ReleaseGlobalSignal(void* signal) const {}

  //! Returns TRUE if the device is available for computations
  bool isOnline() const { return online_; }

  //! Allocate host memory in terms of numa policy set by user
  void* hostNumaAlloc(size_t size, size_t alignment, bool atomics = false) const;

  //! Allocate host memory from agent info
  void* hostAgentAlloc(size_t size, const AgentInfo& agentInfo, bool atomics = false) const;


  const device::BlitManager& blitMgr() const { return blitQueue()->blitMgr(); }

  //! Returns device settings
  const device::Settings& settings() const { return *settings_; }

  //! Create internal blit program
  bool createBlitProgram();

  //! Returns blit program info structure
  BlitProgram* blitProgram() const { return blitProgram_; }

  //! RTTI internal implementation
  virtual ObjectType objectType() const { return ObjectTypeDevice; }

  //! Returns app profile
  static const AppProfile* appProfile() { return &appProfile_; }

  //! Returns transfer buffer object
  BlitBuffers& blitWrite() const { return *blitWrite_; }

  //! Returns transfer buffer object
  BlitBuffers& blitRead() const { return *blitRead_; }


  //! Register a hardware debugger manager
  HwDebugManager* hwDebugMgr() const { return hwDebugMgr_; }

  //! Initialize the Hardware Debug Manager
  virtual int32_t hwDebugManagerInit(drv::Context* context, uintptr_t messageStorage) {
    return CL_SUCCESS;
  }

  //! Remove the Hardware Debug Manager
  virtual void hwDebugManagerRemove() {}

  //! Adds GPU memory to the VA cache list
  void addVACache(device::Memory* memory) const;

  //! Removes GPU memory from the VA cache list
  void removeVACache(const device::Memory* memory) const;

  //! Finds GPU memory from virtual address
  device::Memory* findMemoryFromVA(const void* ptr, size_t* offset) const;

  static std::vector<Device*>& devices() { return *devices_; }

  // P2P agents avaialble for this device
  const std::vector<hsa_agent_t>& p2pAgents() const { return p2p_agents_; }

  // User enabled peer devices
  const bool isP2pEnabled() const { return (enabled_p2p_devices_.size() > 0) ? true : false; }

  // Update the global free memory size
  void updateFreeMemory(size_t size, bool free);



  // P2P devices that are accessible from the current device
  std::vector<cl_device_id> p2pDevices_;

  // P2P devices for memory allocation. This list contains devices that can have access to the
  // current device
  std::vector<Device*> p2p_access_devices_;

  //! Checks if OCL runtime can use code object manager for compilation
  bool ValidateComgr();

  //! Checks if OCL runtime can use hsail for compilation
  bool ValidateHsail();

  virtual bool IpcCreate(void* dev_ptr, size_t* mem_size, void* handle, size_t* mem_offset) const ;
  virtual bool IpcAttach(const void* handle, size_t mem_size, size_t mem_offset,
                         unsigned int flags, void** dev_ptr) const ;
  virtual bool IpcDetach(void* dev_ptr) const ;

  //! Return context
  drv::Context& context() const { return *context_; }

  //! Return private global device context for P2P allocations
  drv::Context& GlbCtx() const { return *glb_ctx_; }

  //! Lock protect P2P staging operations
  Monitor& P2PStageOps() const { return p2p_stage_ops_; }

  //! Staging buffer for P2P transfer
  Memory* P2PStage() const { return p2p_stage_; }

  //! Does this device allow P2P access?
  bool P2PAccessAllowed() const { return (p2p_access_devices_.size() > 0) ? true : false; }

  //! Returns the list of devices that can have access to the current
  const std::vector<Device*>& P2PAccessDevices() const { return p2p_access_devices_; }

  //! Returns index of current device
  uint32_t index() const { return index_; }

  //! Returns value for LinkAttribute for lost of vectors
  virtual bool findLinkInfo(const drv::Device& other_device,
                            std::vector<LinkAttrType>* link_attr) ;

  //! Returns the lock object for the virtual gpus list
  drv::Monitor& vgpusAccess() const { return vgpusAccess_; }
  typedef std::vector<VirtualGPU*> VirtualGPUs;
  const VirtualGPUs& vgpus() const { return vgpus_; }
  VirtualGPU* blitQueue() const;

  MemPoolMgr* getMemPoolMgr() const { return mempool_mgr_; }

  hsa_queue_t* acquireQueue(uint32_t queue_size_hint, bool coop_queue = false,
                            const std::vector<uint32_t>& cuMask = {},
                            drv::CommandQueue::Priority priority = drv::CommandQueue::Priority::Normal);

  //! Release HSA queue
  void releaseQueue(hsa_queue_t*, const std::vector<uint32_t>& cuMask = {});

  //! For the given HSA queue, return an existing hostcall buffer or create a
  //! new one. queuePool_ keeps a mapping from HSA queue to hostcall buffer.
  void* getOrCreateHostcallBuffer(hsa_queue_t* queue, bool coop_queue = false,
                                  const std::vector<uint32_t>& cuMask = {});

  //! Return multi GPU grid launch sync buffer
  address MGSync() const { return mg_sync_; }

  //! Initialize memory in AMD HMM on the current device or keeps it in the host memory
  bool SvmAllocInit(void* memory, size_t size) const;

  // Notifies device about context destroy
  virtual void ContextDestroy() {}

  //! Returns active wait state for this device
  bool ActiveWait() const { return activeWait_; }

  void SetActiveWait(bool state) { activeWait_ = state; }

  void getGlobalCUMask(std::string cuMaskStr);
  virtual drv::Memory* GetArenaMemObj(const void* ptr, size_t& offset) ;

 protected:
  //! Enable the specified extension
  char* getExtensionString();

  device::Info info_;             //!< Device info structure
  device::Settings* settings_;    //!< Device settings
  union {
    struct {
      uint32_t online_: 1;        //!< The device in online
      uint32_t activeWait_: 1;    //!< If true device requires active wait
    };
    uint32_t  state_;             //!< State bit mask
  };

  BlitProgram* blitProgram_;      //!< Blit program info
  static AppProfile appProfile_;  //!< application profile
  HwDebugManager* hwDebugMgr_;    //!< Hardware Debug manager
  drv::Context* context_;         //!< Context

  static drv::Context* glb_ctx_;      //!< Global context with all devices
  static drv::Monitor p2p_stage_ops_; //!< Lock to serialise cache for the P2P resources
  static Memory* p2p_stage_;          //!< Staging resources

  drv::Memory* arena_mem_obj_;        //!< Arena memory object

 private:
  const Isa *isa_;                //!< Device isa
  bool IsTypeMatching(cl_device_type type, bool offlineDevices);

  uint32_t preferred_numa_node_;
  std::vector<Device*>* devices_;  //!< All known devices
  std::vector<Device*> enabled_p2p_devices_;  //!< List of user enabled P2P devices for this device
  mutable std::mutex lock_allow_access_; //!< To serialize allow_access calls
  uint32_t pciDeviceId_;
  MemPoolMgr *mempool_mgr_;

  size_t gpuvm_segment_max_alloc_;
  size_t alloc_granularity_;

  VirtualGPU* blitQueue_;  //!< Transfer queue, created on demand
  BlitBuffers* blitRead_;   //!< Transfer buffers read
  BlitBuffers* blitWrite_;  //!< Transfer buffers write
  std::atomic<size_t> freeMem_;   //!< Total of free memory available

  static address mg_sync_;  //!< MGPU grid launch sync memory (SVM location)

  Monitor* vaCacheAccess_;                            //!< Lock to serialize VA caching access
  std::map<uintptr_t, device::Memory*>* vaCacheMap_;  //!< VA cache map
  uint32_t index_;  //!< Unique device index

  struct QueueInfo {
    int refCount;
    void* hostcallBuffer_;
  };

  //! a vector for keeping Pool of HSA queues with low, normal and high priorities for recycling
  std::vector<std::map<Queue*, QueueInfo>> queuePool_;
  //! returns a hsa queue from queuePool with least refCount and updates the refCount as well
  Queue* getQueueFromPool(const uint qIndex);

  void* coopHostcallBuffer_;

  //! Pool of HSA queues with custom CU masks
  std::vector<std::map<hsa_queue_t*, QueueInfo>> queueWithCUMaskPool_;

 public:
  std::atomic<uint> numOfVgpus_;  //!< Virtual gpu unique index

  //! enum for keeping the total and available queue priorities
  enum QueuePriority : uint { Low = 0, Normal = 1, High = 2, Total = 3};
};

/*! @}
 *  @}
 */

}  // namespace drv

