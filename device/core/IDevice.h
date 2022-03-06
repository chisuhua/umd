#pragma once
// #include "Stream.h"
// #include "StreamType.h"
#include "inc/device_type.h"
#include <cassert>

#if 0
typedef enum {
    NAME = 0,
    // * Agent capability. The type of this attribute is ::hsa_agent_feature_t.
    FEATURE = 2,
    MACHINE_MODEL = 3,
    PROFILE = 4,
    // Must be a power of 2 greater than 0. Must not exceed
    QUEUE_MIN_SIZE = 13,
    QUEUE_MAX_SIZE = 14,
    // * Type of a queue created in the agent. The type of this attribute is hsa_queue_type32_t.
    QUEUE_TYPE = 15,
    IS_APU_NODE = 25
} device_info_t;
#endif


#define HSA_ARGUMENT_ALIGN_BYTES 16
#define HSA_QUEUE_ALIGN_BYTES 64
#define HSA_PACKET_ALIGN_BYTES 64

extern void cmdio_set_event(HsaEvent*);

namespace core {

class IDevice {
public:
    void DestroyQueue(QUEUEID QueueId) {};

    // uint64_t QueueAllocator(device_t device, uint32_t, uint32_t) {};

    // void QueueDeallocator(device_t device, AqlPacket*) {};

    HsaNodeProperties properties(device_t device)
    {
        return properties_;
    };

    void CreateQueue(uint32_t node_id, HSA_QUEUE_TYPE Type,
        uint32_t QueuePercentage,
        HSA_QUEUE_PRIORITY Priority,
        void* QueueAddress,
        uint64_t QueueSizeInBytes,
        HsaEvent* Event,
        QueueResource* QueueResource) {};

    bool CreateEvent(HsaEventDescriptor* EventDesc,
        bool ManualReset, bool IsSignaled,
        HsaEvent** Event)
    {
    }

    void SetEvent(HsaEvent* event) {
        // cmdio_set_event(event);
    };

    void DestroyEvent(HsaEvent* evt) {};

    void WaitOnEvent(HsaEvent* Event, uint32_t Milliseconds) {};

    void WaitOnMultipleEvents(HsaEvent* Events[], uint32_t NumEvents,
        bool WaitOnAll, uint32_t Milliseconds) {};

    void* QueueBufferAllocator(uint32_t queue_size_pkts, uint32_t align) {
        // FIXME
        assert(false && "should override QueueBuffferAllocator");
    };

    void QueueBufferDeallocator(void* queue_address) {
        // FIXME
        assert(false && "should override QUeueBuffferDeallocator");
    };

    status_t GetInfo(device_t, uint32_t query_info, uint32_t* max_packets)
    {
        return SUCCESS;
    }

    HsaNodeProperties properties_;
};
}
