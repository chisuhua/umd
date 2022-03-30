#pragma once
class IPlatform;

extern "C" {
#if 0
status_t libcuda_launchKernel(IPlatform*, const void *hostFunction,
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

typedef status_t (*pfn_libcuda_launchKernel)(IPlatform*, const void *hostFunction);


// status_t libcuda_setupKernelArgument(IPlatform*, const void *arg, size_t size,
//                                      size_t offset);

typedef status_t (*pfn_libcuda_setupArgument)(IPlatform*, const void *arg, size_t size,
                                      size_t offset);

typedef status_t (*pfn_libcuda_setupPtxSimArgument)(IPlatform*, void *finfo, const void **arg) ;

}
