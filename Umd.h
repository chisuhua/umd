#pragma once
#include "status.h"
#include <cstddef>
#include <string>
#include <map>
#include <unordered_map>
#include <dlfcn.h>
#include <assert.h>
#include "common.h"


class Umd;
class IMemRegion;
class IAgent;
class CUctx;
class DispatchInfo;
class IPlatform;

namespace loader {
    class Executable;
}

typedef IPlatform* (*pfn_create_platform)();

// extern "C" IPlatform* create_platform();

static std::unordered_map<std::string, IPlatform*> g_platform_instance;
static std::map<CUctx*, Umd*> g_umd_instance;
#if 0
typedef status_t (*pfn_memory_register)(CUctx* ctx, void* address, size_t size);
typedef status_t (*pfn_memory_deregister)(CUctx* ctx, void* address, size_t size);
typedef status_t (*pfn_memory_allocate)(CUctx* ctx, size_t size, void** ptr, IMemRegion *region = nullptr);
typedef status_t (*pfn_memory_free)(CUctx* ctx, void* ptr) ;
typedef IMemRegion* (*pfn_get_system_memregion)(CUctx* ctx);
typedef IMemRegion* (*pfn_get_device_memregion)(CUctx* ctx, IAgent* agent);
typedef status_t (*pfn_free_memregion)(CUctx* ctx, IMemRegion *region);

struct UmdImpl {
}
#endif

#if !defined(__VECTOR_TYPES_H__)
/*
struct dim3
{
    unsigned int x, y, z;
};
*/
#endif

class Umd {
public:
    static Umd* get(CUctx* ctx);

    Umd(CUctx *ctx, IPlatform *platform) {
        m_ctx = ctx;
        m_platform = platform;
    }


    CUctx* m_ctx;
    IPlatform* m_platform;

    status_t memory_register(void* address, size_t size);
    status_t memory_deregister(void* address, size_t size);
    status_t memory_allocate(size_t size, void** ptr, IMemRegion *region = nullptr);
    status_t memory_copy(void* dst, const void* src, size_t count, UmdMemcpyKind kind);
    status_t memory_free(void* ptr);
    status_t memory_set(void* ptr, int c, size_t count);
    IMemRegion* get_system_memregion();
    IMemRegion* get_device_memregion(IAgent* agent);
    status_t free_memregion(IMemRegion *region);
    loader::Executable* load_program(const std::string& file);
    void set_kernel_disp(const std::string& kernel_name, loader::Executable* exec, DispatchInfo** disp_info, struct dim3 gridDim, struct dim3 blockDim, uint64_t param_addr);

    status_t getDeviceCount(int* count);
    status_t getDeviceProperties(void* prop, int id = 0);
    status_t getDevice(int* device);
};

