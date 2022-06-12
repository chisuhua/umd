#include "../../libcuda/CUctx.h"
#include "../../libcuda/abstract_hardware_model.h"
#include "../../libcuda/gpu-sim.h"
#include "../../libcuda/stream_manager.h"
#include "../../libcuda/gpgpu_context.h"
#include "driver/cuda/Context.h"
#include "plat_gem5umd.h"
#include "../../libcuda/gem5/opuumd/gem5umd_runtime_api.h"
#include <map>
#include <typeinfo>

using namespace libcuda;

extern "C" IPlatform* create_platform(IContext* ctx) {
    static std::unordered_map<std::string, IPlatform*> platform_instance;
    std::string instance_name = "default";
    if (ctx != nullptr) {
        instance_name = typeid(*ctx).name();
    }
    if (platform_instance.count(instance_name) == 0) {
        platform_instance[instance_name] = new plat_libgem5umd(instance_name, dynamic_cast<drv::Context*>(ctx));
    }
    return platform_instance[instance_name];
};


status_t plat_libgem5umd::memory_register(void* address, size_t size) {
  if (size == 0 && address != NULL) {
    return ERROR_INVALID_ARGUMENT;
  }

  return SUCCESS;
}

status_t plat_libgem5umd::memory_deregister(void* address, size_t size) {
  return SUCCESS;
}

IMemRegion* sys_memregion;
status_t plat_libgem5umd::memory_allocate(size_t size, void** ptr, IMemRegion *region) {
  if (region != nullptr) {
    *ptr = malloc(size);
  } else {
    gem5umdMalloc(ptr, size);
  }
  return SUCCESS;
}

status_t plat_libgem5umd::memory_copy(void* dst, const void* src, size_t count, UmdMemcpyKind kind) {
    if (kind == UmdMemcpyKind::HostToDevice)
      gem5umdMemcpy(dst, src, count, cudaMemcpyHostToDevice);
    else if (kind == UmdMemcpyKind::DeviceToHost)
      gem5umdMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
    else if (kind == UmdMemcpyKind::DeviceToDevice)
      gem5umdMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
    else if (kind == UmdMemcpyKind::Default) {
      if ((size_t)src >= GLOBAL_HEAP_START) {
        if ((size_t)dst >= GLOBAL_HEAP_START)
          gem5umdMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
        else
          gem5umdMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
      } else {
        if ((size_t)dst >= GLOBAL_HEAP_START)
          gem5umdMemcpy(dst, src, count, cudaMemcpyHostToDevice);
        else {
          printf(
              "GPGPU-Sim PTX: UmdMemcpyKind:: - ERROR : unsupported transfer: host to "
              "host\n");
          abort();
        }
      }
    } else {
      printf("GPGPU-Sim PTX: UmdMemcpyKind:: - ERROR : unsupported UmdMemcpyKind::Kind\n");
      abort();
    }

};

status_t plat_libgem5umd::memory_free(void* ptr) {
  gem5umdFree(ptr);
  return SUCCESS;
}

IMemRegion* plat_libgem5umd::get_system_memregion() {
  // FIXME fake
  uint64_t *tmp = (uint64_t*)&sys_memregion;
  *tmp = 1;
  return sys_memregion;
}

IMemRegion* plat_libgem5umd::get_device_memregion(IAgent* agent) {
  return nullptr;
}

status_t plat_libgem5umd::free_memregion(IMemRegion *region) {
  return SUCCESS;
}

status_t plat_libgem5umd::getDeviceCount(int* count) {
  gem5umdGetDeviceCount(count);
  return SUCCESS;
};

status_t plat_libgem5umd::getDeviceProperties(void* prop, int id) {
  gem5umdGetDeviceProperties((cudaDeviceProp*)prop, id);
  return SUCCESS;
};

status_t plat_libgem5umd::getDevice(int* device) {
  gem5umdGetDevice(device);
  return SUCCESS;
};

extern "C" {
status_t libgem5umd_launchKernel(IPlatform*, const void *hostFun/*,
            unsigned int gridDimX,
            unsigned int gridDimY,
            unsigned int gridDimZ,
            unsigned int blockDimX,
            unsigned int blockDimY,
            unsigned int blockDimZ,
                                       void** args,
                                      size_t sharedMemBytes,
                                      void** stream*/) {
  //dim3 gridDim(gridDimX, gridDimY, gridDimZ);
  //dim3 blockDim(blockDimX, blockDimY, blockDimZ);
  //gem5cudaConfigureCall(gridDim, blockDim, sharedMemBytes, (cudaStream_t)stream);
  gem5umdLaunch((const char*)hostFun);
}

status_t libgem5umd_setupKernelArgument(IPlatform*, const void *arg, size_t size,
                                      size_t offset) {
  gem5umdSetupArgument(arg, size, offset);
}

status_t libgem5umd_setupPtxSimArgument(IPlatform*, void *finfo, const void **arg) {
  assert(false);
  // gem5cudaSetupPtxSimArgument((function_info*)finfo, arg);
}

}


