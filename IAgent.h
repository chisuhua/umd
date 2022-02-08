#pragma once
#include <vector>

class ISignal;
class IMemRegion;
class ICache;
class IQueue;

#define DISALLOW_COPY_AND_ASSIGN(TypeName)                                                         \
  TypeName(const TypeName&) = delete;                                                              \
  TypeName(TypeName&&) = delete;                                                                   \
  void operator=(const TypeName&) = delete;                                                        \
  void operator=(TypeName&&) = delete;


class IAgent {
public:
    // Lightweight RTTI for vendor specific implementations.
    enum AgentType { kGpu = 0,
        kCpu = 1,
        kUnknown = 2 };

    explicit IAgent(uint32_t node_id, AgentType type)
        : node_id_(node_id)
        , agent_type_(uint32_t(type))
        , profiling_enabled_(false)
    {
    }

    virtual ~IAgent() { }

    virtual status_t DmaCopy(void* dst, const void* src, size_t size)
    {
        return ERROR;
    }

    virtual status_t DmaCopy(void* dst, IAgent& dst_agent,
        const void* src, IAgent& src_agent,
        size_t size,
        std::vector<signal_t>& dep_signals,
        signal_t out_signal)
    {
        return ERROR;
    }

    virtual status_t DmaFill(void* ptr, uint32_t value, size_t count)
    {
        return ERROR;
    }

    virtual status_t IterateRegion(
        status_t (*callback)(const IMemRegion* region, void* data),
        void* data) const = 0;

    virtual status_t IterateCache(
        status_t (*callback)(ICache* cache, void* data),
        void* data) const = 0;

    virtual status_t QueueCreate(size_t size, queue_type32_t queue_type,
        HsaEventCallback event_callback, void* data,
        uint32_t private_segment_size,
        uint32_t group_segment_size,
        queue_t* queue)
        = 0;

    virtual const std::vector<const IMemoryRegion*>& regions() const = 0;

    virtual uint64_t HiveId() const { return 0; }

    uint32_t agent_type() const { return agent_type_; }

    uint32_t node_id() const { return node_id_; }

    bool profiling_enabled() const { return profiling_enabled_; }

    virtual status_t profiling_enabled(bool enable)
    {
        const status_t stat = EnableDmaProfiling(enable);
        if (SUCCESS == stat) {
            profiling_enabled_ = enable;
        }

        return stat;
    }

    virtual status_t EnableDmaProfiling(bool enable)
    {
        return SUCCESS;
    }
#if 0
    HSA_CAPABILITY GetCapability() {
        return capability_;
    }

    HSA_CAPABILITY capability_;
#endif


    IRuntime* GetRuntime() {return runtime_;}

private:
    // @brief Node id.
    const uint32_t node_id_;

    const uint32_t agent_type_;

    bool profiling_enabled_;

    IRuntime* runtime_;

    // Forbid copying and moving of this object
    DISALLOW_COPY_AND_ASSIGN(IAgent);
    friend class IMemRegion;
};

