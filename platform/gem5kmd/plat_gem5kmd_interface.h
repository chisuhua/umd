#pragma once
class plat_libgem5kmd;

status_t launchKernel(plat_libgem5kmd*, const void *hostFunction,
                                      dim3 gridDim,
                                      dim3 blockDim,
                                      void** args,
                                      size_t sharedMemBytes,
                                      cudaStream_t stream);

status_t setupKernelArgument(plat_libgem5kmd*, const void *arg, size_t size,
                                      size_t offset);

