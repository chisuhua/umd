#pragma once
class IPlatform;

extern "C" {

typedef status_t (*pfn_libgem5umd_launchKernel)(IPlatform*, const void *hostFunction/*,
            unsigned int gridDimX,
            unsigned int gridDimY,
            unsigned int gridDimZ,
            unsigned int blockDimX,
            unsigned int blockDimY,
            unsigned int blockDimZ,
                                      void** args,
                                      size_t sharedMemBytes,
                                      cudaStream_t void** stream*/);

#if 0
status_t libgem5umd_launchKernel(IPlatform*, const void *hostFunction,
            unsigned int gridDimX,
            unsigned int gridDimY,
            unsigned int gridDimZ,
            unsigned int blockDimX,
            unsigned int blockDimY,
            unsigned int blockDimZ,
                                      void** args,
                                      size_t sharedMemBytes,
                                      /*cudaStream_t*/void** stream);
#endif

typedef status_t (*pfn_libgem5umd_setupArgument)(IPlatform*, const void *arg, size_t size,
                                      size_t offset);
#if 0
status_t libgem5umd_setupKernelArgument(IPlatform*, const void *arg, size_t size,
                                      size_t offset);
#endif

typedef status_t (*pfn_libgem5umd_setupPtxSimArgument)(IPlatform*, void *finfo, const void **arg) ;

}
