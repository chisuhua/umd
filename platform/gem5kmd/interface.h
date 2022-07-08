#pragma once
class IPlatform;

extern "C" {

typedef status_t (*pfn_libgem5kmd_launchKernel)(IPlatform*, const void *hostFunction, void* disp_info, void *stream/*
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
status_t libgem5kmd_launchKernel(IPlatform*, const void *hostFunction,
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

typedef status_t (*pfn_libgem5kmd_setupArgument)(IPlatform*, const void *arg, size_t size,
                                      size_t offset);
#if 0
status_t libgem5kmd_setupKernelArgument(IPlatform*, const void *arg, size_t size,
                                      size_t offset);
#endif

typedef status_t (*pfn_libgem5kmd_setupPtxSimArgument)(IPlatform*, void *finfo, const void **arg) ;

}
