#pragma once

/// Specifies flags for @ref IGpuMemory creation.
union MemoryCreateFlags
{
    struct
    {
        uint32_t hostAllocDefault       :  1;
        uint32_t hostAllocPortable      :  1;
        uint32_t hostAllocMapped        :  1;
        uint32_t HostAllocWriteCombined :  1;
        uint32_t HostAllocCoherent      :  1;
        uint32_t HostAllocNonCoherent   :  1;
        uint32_t reserved         :  5; ///< Reserved for future use.
    };
    uint32     u32All;                 ///< Flags packed as 32-bit uint.
};

class IAgent;

namespace rt {

// template file
class IRuntime {
public:
    virtual void* allocMemory(Context& context, size_t size, size_t alignment, uint32_t flags, const Device* cur_dev  ) = 0;

    virtual void freeMemory(const Context& context, void* ptr);

    // virtual void fillMemory(void* dst, const void* src, size_t size, size_t times);

    virtual status_t copyMemory(void* dst, const void* src, size_t size) = 0;

    virtual bool isMemAlloced(const void* ptr);

    void Add(uintptr_t k, uintptr_t v);
    void Remove(uintptr_t k);
    bool Contains(uintptr_t ptr);

    std::map<uintptr_t, uintptr_t> Allocated_;  // !< Allocated buffers
    Monitor AllocatedLock_;

    inline bool initialized();
    bool init();
    void tearDown();

    //! Return true if the Runtime is still single-threaded.
    bool singleThreaded() { return !initialized(); }

    // status_t findDevices(Device** agent, void *data) ;

    std::function<void*(size_t, size_t)> system_allocator_;
    std::function<void(void*)> system_deallocator_;

    const Flag& flag() const { return flag_; }

#if 0
    IRuntime();
    IRuntime(const IRuntime&);
    IRuntime& operator=(const IRuntime&);
#endif
    virtual ~IRuntime() {}

    // SharedSignalPool* GetSignalPool() { return signal_pool_; };
    // EventPool* GetEventPool() { return event_pool_; };
    // StreamPool* GetStreamPool() { return stream_pool_; };
    // Device* GetDevice() { return device_; };

    uint64_t sys_clock_freq_;
    std::atomic<uint32_t> ref_count_;
    Flag flag_;

    // SharedSignalPool* signal_pool_;

    // Pools KFD Events for InterruptSignal
    // EventPool* event_pool_;

    // StreamPool* stream_pool_;

    // Device* device_;

    virtual status_t Load() {};
    virtual void Unload() {};

    virtual const timer::fast_clock::duration GetTimeout(double timeout) = 0;

    virtual const timer::fast_clock::time_point GetTimeNow() = 0;

    virtual void Sleep(uint32_t milisecond) = 0;

    // void DestroyEvent(HsaEvent* evt);
    virtual std::vector<IAgent*> GetCpuAgents() = 0;


    bool use_interrupt_wait_;
};

}
