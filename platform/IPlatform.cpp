#include "KernelDispInfo.h"
#include "program/Program.h"
#include "program/loader/Loader.h"
#include "IContext.h"
#include <map>
#include <memory>
#include <unordered_map>
#include <dlfcn.h>
#include <assert.h>
#include "IPlatform.h"

class IMemRegion;
class IAgent;

typedef IPlatform* (*pfn_create_platform)(IContext*);
static std::unordered_map<std::string, pfn_create_platform> g_platform_creator;

IPlatform* IPlatform::getInstance(IContext *ctx) {
    std::string platform_name;
    platform_name = IContext::platformName(ctx);

    if (g_platform_creator.count(platform_name) == 0)  {
        std::string umd_libname = "lib";
        umd_libname += platform_name;
        umd_libname += ".so";

        pfn_create_platform creator = nullptr;
        void* handle = dlopen(umd_libname.c_str(), RTLD_LAZY | RTLD_GLOBAL);
        if (handle == nullptr) {
            printf("dlopen error - %s\n", dlerror());
            assert(false);
        }
        creator = (pfn_create_platform)dlsym(handle, "create_platform");
        if (creator == nullptr) {
            printf("dlsym error - %s\n", dlerror());
            assert(false);
        }
        g_platform_creator[platform_name] = creator;
    }

    return (*g_platform_creator[platform_name])(ctx);
};

exec_handle_t IPlatform::load_program(const std::string& file) {
    loader::Executable* exec = LoadProgram(file, ctx_, ctx_->get_agent());
    return static_cast<exec_handle_t>(exec);
};

void IPlatform::set_kernel_disp(const std::string& kernel_name,
        exec_handle_t exec_handle,
        DispatchInfo** disp_info,
        unsigned int gridDimX,
        unsigned int gridDimY,
        unsigned int gridDimZ,
        unsigned int blockDimX,
        unsigned int blockDimY,
        unsigned int blockDimZ,
        uint64_t param_addr) {
    loader::Executable* exec = static_cast<loader::Executable*>(exec_handle);
    loader::Symbol * sym = exec->GetSymbol((kernel_name + ".kd").c_str(), ctx_->get_agent());
    DispatchInfo* dinfo = new DispatchInfo;
    sym->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, (void*)&dinfo->kernel_prog_addr);
    sym->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CTRL, (void*)&dinfo->kernel_ctrl);
    sym->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_MODE, (void*)&dinfo->kernel_mode);
    dinfo->grid_dim_x = gridDimX;
    dinfo->grid_dim_y = gridDimY;
    dinfo->grid_dim_z = gridDimZ;
    dinfo->block_dim_x = blockDimX;
    dinfo->block_dim_y = blockDimY;
    dinfo->block_dim_z = blockDimZ;
    dinfo->kernel_param_addr = param_addr;
    *disp_info = dinfo;
}

