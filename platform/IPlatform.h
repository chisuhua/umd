#pragma once
#include "status.h"
#include "common.h"
#include <cstddef>

class IMemRegion;
class IAgent;
class CUctx;


class IPlatform {
public:
    IPlatform() {};
    ~IPlatform() {};

    CUctx *m_ctx;

    void setCtx(CUctx *ctx) { m_ctx = ctx;}

    virtual status_t memory_register(void* address, size_t size) = 0;
    virtual status_t memory_deregister(void* address, size_t size) = 0;
    virtual status_t memory_allocate(size_t size, void** ptr, IMemRegion *region = nullptr) = 0;
    virtual status_t memory_copy(void* dst, const void* src, size_t count, UmdMemcpyKind kind) = 0;
    virtual status_t memory_free(void* ptr) = 0;

    virtual status_t getDeviceCount(int* count) {};
    virtual status_t getDeviceProperties(void* prop, int id = 0) {};
    virtual status_t getDevice(int* device) {};

    virtual IMemRegion* get_system_memregion() = 0;
    virtual IMemRegion* get_device_memregion(IAgent* agent) = 0;
    virtual status_t free_memregion(IMemRegion *region) = 0;
};
