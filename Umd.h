#include "status.h"
#include <cstddef>
#include <string>
#include <map>
#include <unordered_map>
#include <dlfcn.h>
#include <assert.h>
#include "IPlatform.h"


class Umd;
class IMemRegion;
class IAgent;
class CUctx;
class DispatchInfo;

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
    static Umd* get(CUctx* ctx) {
        if (g_umd_instance.count(ctx) == 1) return g_umd_instance[ctx];
        char *buff = getenv("UMD");
        std::string ret = "umdcuda";
        if (buff) {
            ret = buff;
        }
        if (g_platform_instance.count(ret) == 0)  {
            std::string umd_libname = "lib";
            umd_libname += ret;
            umd_libname += ".so";

            pfn_create_platform create_platform = nullptr;
            void* handle = dlopen(umd_libname.c_str(), RTLD_LAZY | RTLD_GLOBAL);
            if (handle == nullptr) {
                printf("dlopen error - %s\n", dlerror());
                assert(false);
            }
            create_platform = (pfn_create_platform)dlsym(handle, "create_platform");
            if (create_platform == nullptr) {
                printf("dlsym error - %s\n", dlerror());
                assert(false);
            }
            g_platform_instance[ret] = (*create_platform)();
        }
        Umd* umd = new Umd(ctx, g_platform_instance[ret]);
        g_platform_instance[ret]->setCtx(ctx);
        g_umd_instance[ctx] = umd;
        return umd;
    };

    Umd(CUctx *ctx, IPlatform *platform) {
        m_ctx = ctx;
        m_platform = platform;
    }


    CUctx* m_ctx;
    IPlatform* m_platform;

    status_t memory_register(void* address, size_t size);
    status_t memory_deregister(void* address, size_t size);
    status_t memory_allocate(size_t size, void** ptr, IMemRegion *region = nullptr);
    status_t memory_free(void* ptr);
    IMemRegion* get_system_memregion();
    IMemRegion* get_device_memregion(IAgent* agent);
    status_t free_memregion(IMemRegion *region);
    loader::Executable* load_program(const std::string& file);
    void set_kernel_disp(const std::string& kernel_name, loader::Executable* exec, DispatchInfo* disp_info, struct dim3 gridDim, struct dim3 blockDim, uint64_t param_addr);
};

