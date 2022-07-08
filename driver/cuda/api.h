typedef void* exec_handle_t;

namespace drv {
CUresult CUDAAPI setDefaultCtx(::CUctx *ctx) ;
CUresult CUDAAPI getDeviceCount(int* count) ;
CUresult CUDAAPI getDeviceProperties(void* prop, int id = 0) ;
CUresult CUDAAPI getDevice(int* device) ;
CUresult CUDAAPI memory_allocate(size_t size, void** ptr) ;
CUresult CUDAAPI memory_copy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) ;
void* load_program(const std::string& file) ;
void set_kernel_disp(const std::string& kernel_name, exec_handle_t exec, DispatchInfo** disp_info, struct dim3 gridDim, struct dim3 blockDim, uint64_t param_addr) ;
CUresult CUDAAPI launchKernel(const char* f, DispatchInfo* disp_info = nullptr, void *stream= nullptr);
CUresult setupKernelArgument(const void *arg, size_t size, size_t offset);
CUresult setupPtxSimArgument(function_info *finfo, const void **arg);


}
