#include "driver/cuda/Context.h"
#include "cuda.h"
#include "platform/IPlatform.h"
#include "driver_types.h"

static drv::Context *g_ctx;
namespace libcuda {
struct function_info;
}

namespace drv {

CUresult CUDAAPI setDefaultCtx(::CUctx *ctx) {
    g_ctx = new drv::Context(ctx);
}

CUresult CUDAAPI getDeviceCount(int* count) {
    IPlatform::getInstance(g_ctx)->getDeviceCount(count);
}

CUresult CUDAAPI getDeviceProperties(void* prop, int id = 0) {
    IPlatform::getInstance(g_ctx)->getDeviceProperties(prop, id);
}

CUresult CUDAAPI getDevice(int* device) {
    IPlatform::getInstance(g_ctx)->getDevice(device);
}

CUresult CUDAAPI memory_allocate(size_t size, void** ptr) {
    IPlatform::getInstance(g_ctx)->memory_allocate(size, ptr);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI memory_copy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    UmdMemcpyKind umd_kind;
    if (kind == cudaMemcpyHostToDevice)
        umd_kind = UmdMemcpyKind::HostToDevice;
    else if (kind == cudaMemcpyDeviceToHost)
        umd_kind = UmdMemcpyKind::DeviceToHost;
    else if (kind == cudaMemcpyDeviceToDevice)
        umd_kind = UmdMemcpyKind::DeviceToDevice;
    else if (kind == cudaMemcpyDefault)
        umd_kind = UmdMemcpyKind::Default;
    IPlatform::getInstance(g_ctx)->memory_copy(dst, src, count, umd_kind);
};

void* load_program(const std::string& file) {
    return IPlatform::getInstance(g_ctx)->load_program(file);
}

void set_kernel_disp(const std::string& kernel_name, exec_handle_t exec, DispatchInfo** disp_info, struct dim3 gridDim, struct dim3 blockDim, uint64_t param_addr) {
    IPlatform::getInstance(g_ctx)->set_kernel_disp(kernel_name, exec, disp_info,
            gridDim.x,
            gridDim.y,
            gridDim.z,
            blockDim.x,
            blockDim.y,
            blockDim.z,
            param_addr);
}

CUresult CUDAAPI launchKernel(const char* f)
{
    IPlatform::getInstance(g_ctx)->launchKernel(f);
}

CUresult setupKernelArgument(const void *arg, size_t size, size_t offset) {
    IPlatform::getInstance(g_ctx)->setupKernelArgument(
                                arg,
                                size,
                                offset);
}

CUresult setupPtxSimArgument(libcuda::function_info *finfo, const void** arg) {
    IPlatform::getInstance(g_ctx)->setupPtxSimArgument(
                                (void*)finfo,
                                arg);
}


}
