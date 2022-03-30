#pragma once
#include "driver/cuda/Context.h"
#include "../IPlatform.h"


class plat_libcuda : public IPlatform {
public:
    plat_libcuda(std::string name, drv::Context* ctx)
        : IPlatform(name, ctx)
    {
        cuctx_ = ctx;
    }
    ~plat_libcuda() = default;
    drv::Context* cuctx_;

    status_t memory_register(void* address, size_t size);
    status_t memory_deregister(void* address, size_t size);
    status_t memory_allocate(size_t size, void** ptr, IMemRegion *region = nullptr);
    status_t memory_copy(void* dst, const void* src, size_t count, UmdMemcpyKind kind);
    status_t memory_free(void* ptr);

    status_t getDeviceCount(int* count) {};
    status_t getDeviceProperties(void* prop, int id = 0) {};
    status_t getDevice(int* device) {};

    IMemRegion* get_system_memregion();
    IMemRegion* get_device_memregion(IAgent* agent);
    status_t free_memregion(IMemRegion *region);

};


