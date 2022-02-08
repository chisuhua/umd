#pragma once
#include "StreamType.h"
#include "util/flag.h"
#include "util/timer.h"
#include <atomic>
#include <functional>
#include <vector>
// #include "Signal.h"
// #include "InterruptSignal.h"
// typedef device_status_t (*CreateQueue)()
//
//

class IAgent;


// template file
class IRuntime {
public:
#if 0
    static status_t Acquire();
    static status_t Release();
    static IRuntime* runtime_singleton_;
    __forceinline static IRuntime& getInstance();
#endif

    virtual status_t AllocateMemory(size_t size, void** address) = 0;

    virtual status_t FreeMemory(void* ptr) = 0;

    virtual status_t CopyMemory(void* dst, const void* src, size_t size) = 0;

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

