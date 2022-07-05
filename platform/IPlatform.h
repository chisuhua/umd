#pragma once
#include "status.h"
#include "common.h"
#include <cstddef>
#include <string>
#include "libcuda/interface.h"
#include "libgem5cuda/interface.h"
#include "gem5umd/interface.h"
#include "gem5kmd/interface.h"
#include "IContext.h"

class IMemRegion;
class IAgent;
// class IContext;
class DispatchInfo;

namespace loader {
    class Executable;
}
typedef void* exec_handle_t;

class IPlatform {
public:
    IPlatform(std::string name, IContext* ctx)
        : name_(name)
        , ctx_(ctx)
    {};
    virtual ~IPlatform() {};

    IContext *ctx_;
    std::string name_;
    void* handle_;
    bool initialized {false};

    // void setCtx(ctx_t *ctx) { ctx_ = ctx;}
    static IPlatform* getInstance(IContext *ctx);

    virtual status_t memory_register(void* address, size_t size) = 0;
    virtual status_t memory_deregister(void* address, size_t size) = 0;
    virtual status_t memory_allocate(size_t size, void** ptr, IMemRegion *region = nullptr) = 0;
    virtual status_t memory_copy(void* dst, const void* src, size_t count, UmdMemcpyKind kind) = 0;
    virtual status_t memory_free(void* ptr) = 0;

    virtual status_t getDeviceCount(int* count) {};
    virtual status_t getDeviceProperties(void* prop, int id = 0) {};
    virtual status_t getDevice(int* device) {};

    exec_handle_t load_program(const std::string& file);
    void set_kernel_disp(const std::string& kernel_name,
            exec_handle_t exec,
            DispatchInfo** disp_info,
            unsigned int gridDimX,
            unsigned int gridDimY,
            unsigned int gridDimZ,
            unsigned int blockDimX,
            unsigned int blockDimY,
            unsigned int blockDimZ,
            uint64_t param_addr);


    virtual IMemRegion* get_system_memregion() = 0;
    virtual IMemRegion* get_device_memregion(IAgent* agent) = 0;
    virtual status_t free_memregion(IMemRegion *region) = 0;

    void* pLaunchKernel = nullptr;
    void* pSetupArgument = nullptr;
    void* pSetupPtxSimArgument = nullptr;

    template<typename... Args>
    status_t launchKernel(Args&&... args) {
        if (ctx_->name_ == "platlibgem5cuda") {
            (*reinterpret_cast<pfn_libgem5cuda_launchKernel>(pLaunchKernel))(this, std::forward<Args>(args)...);
        } else if (ctx_->name_ == "platgem5umd") {
            (*reinterpret_cast<pfn_libgem5umd_launchKernel>(pLaunchKernel))(this, std::forward<Args>(args)...);
        } else if (ctx_->name_ == "platgem5kmd") {
            (*reinterpret_cast<pfn_libgem5kmd_launchKernel>(pLaunchKernel))(this, std::forward<Args>(args)...);
        }
    }

    template<typename... Args>
    status_t setupKernelArgument(Args&&... args) {
        if (ctx_->name_ == "platlibgem5cuda") {
            (*reinterpret_cast<pfn_libgem5cuda_setupArgument>(pSetupArgument))(this, std::forward<Args>(args)...);
        } else if (ctx_->name_ == "platgem5umd") {
            (*reinterpret_cast<pfn_libgem5umd_setupArgument>(pSetupArgument))(this, std::forward<Args>(args)...);
        } else if (ctx_->name_ == "platgem5kmd") {
            (*reinterpret_cast<pfn_libgem5kmd_setupArgument>(pSetupArgument))(this, std::forward<Args>(args)...);
        }
    }

    template<typename... Args>
    status_t setupPtxSimArgument(Args&&... args) {
        (*reinterpret_cast<pfn_libgem5cuda_setupPtxSimArgument>(pSetupPtxSimArgument))(this, std::forward<Args>(args)...);
    }
};
