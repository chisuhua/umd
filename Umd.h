#include "status.h"
#include <cstddef>
#include <string>
#include <map>
#include <dlfcn.h>
#include <assert.h>


class Umd;
class IMemRegion;
class IAgent;
class CUctx;

namespace loader {
    class Executable;
}

typedef Umd* (*pfn_get_umd)(CUctx*);

extern "C" Umd* get_umd(CUctx* ctx);

static std::map<CUctx*, Umd*> g_umd_instance;
static std::map<std::string, pfn_get_umd> g_get_umd_func;
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

class Umd {
public:
    static Umd* get(CUctx* ctx) {
        if (g_umd_instance.count(ctx) == 1) return g_umd_instance[ctx];
        char *buff = getenv("UMD");
        std::string ret = "umdcuda";
        if (buff) {
            ret = buff;
        }
        std::string umd_libname = "lib";
        umd_libname += ret;
        umd_libname += ".so";

        pfn_get_umd get_umd = nullptr;
        if (g_get_umd_func.count(umd_libname) == 1)  {
            get_umd = g_get_umd_func[umd_libname];
        } else {
            void* handle = dlopen(umd_libname.c_str(), RTLD_LAZY | RTLD_GLOBAL);
            if (handle == nullptr) {
                printf("dlopen error - %s\n", dlerror());
                assert(false);
            }
            get_umd = (pfn_get_umd)dlsym(handle, "get_umd");
        }
        g_umd_instance[ctx] = (*get_umd)(ctx);
        return g_umd_instance[ctx];
    };

    Umd(CUctx *ctx) {
        m_ctx = ctx;
    }


    CUctx* m_ctx;

    virtual status_t memory_register(void* address, size_t size) = 0;
    virtual status_t memory_deregister(void* address, size_t size) = 0;
    virtual status_t memory_allocate(size_t size, void** ptr, IMemRegion *region = nullptr) = 0;
    virtual status_t memory_free(void* ptr) = 0;
    virtual IMemRegion* get_system_memregion() = 0;
    virtual IMemRegion* get_device_memregion(IAgent* agent) = 0;
    virtual status_t free_memregion(IMemRegion *region) = 0;
    loader::Executable* load_program(const std::string& file, CUctx* ctx, IAgent* agent = nullptr);
};

