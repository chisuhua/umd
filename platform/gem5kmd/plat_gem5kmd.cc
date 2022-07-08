#include "../../libcuda/CUctx.h"
#include "../../libcuda/abstract_hardware_model.h"
#include "../../libcuda/gpu-sim.h"
#include "../../libcuda/stream_manager.h"
#include "../../libcuda/gpgpu_context.h"
#include "driver/cuda/Context.h"
#include "plat_gem5kmd.h"
// #include "../../libcuda/gem5/opukmd/gem5kmd_runtime_api.h"
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
        platform_instance[instance_name] = new plat_libgem5kmd(instance_name, dynamic_cast<drv::Context*>(ctx));
    }
    return platform_instance[instance_name];
};


status_t plat_libgem5kmd::memory_register(void* address, size_t size) {
  if (size == 0 && address != NULL) {
    return ERROR_INVALID_ARGUMENT;
  }

  return SUCCESS;
}

status_t plat_libgem5kmd::memory_deregister(void* address, size_t size) {
  return SUCCESS;
}

IMemRegion* sys_memregion;
status_t plat_libgem5kmd::memory_allocate(size_t size, void** ptr, IMemRegion *region) {
  if (region != nullptr) {
    *ptr = malloc(size);
  } else {
    // gem5kmdMalloc(ptr, size);
  }
  return SUCCESS;
}

status_t plat_libgem5kmd::memory_copy(void* dst, const void* src, size_t count, UmdMemcpyKind kind) {
#if 0
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
#endif
};

status_t plat_libgem5kmd::memory_free(void* ptr) {
  // gem5kmdFree(ptr);
  return SUCCESS;
}

IMemRegion* plat_libgem5kmd::get_system_memregion() {
  // FIXME fake
  uint64_t *tmp = (uint64_t*)&sys_memregion;
  *tmp = 1;
  return sys_memregion;
}

IMemRegion* plat_libgem5kmd::get_device_memregion(IAgent* agent) {
  return nullptr;
}

status_t plat_libgem5kmd::free_memregion(IMemRegion *region) {
  return SUCCESS;
}

status_t plat_libgem5kmd::getDeviceCount(int* count) {
  // gem5kmdGetDeviceCount(count);
  return SUCCESS;
};

status_t plat_libgem5kmd::getDeviceProperties(void* prop, int id) {
  // gem5kmdGetDeviceProperties((cudaDeviceProp*)prop, id);
  return SUCCESS;
};

status_t plat_libgem5kmd::getDevice(int* device) {
  // gem5kmdGetDevice(device);
  return SUCCESS;
};

extern "C" {
status_t libgem5kmd_launchKernel(IPlatform*, const void *hostFun, void* disp_info, void *stream/*,
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
  // gem5kmdLaunch((const char*)hostFun);
}

status_t libgem5kmd_setupKernelArgument(IPlatform*, const void *arg, size_t size,
                                      size_t offset) {
  // gem5kmdSetupArgument(arg, size, offset);
}

status_t libgem5kmd_setupPtxSimArgument(IPlatform*, void *finfo, const void **arg) {
  assert(false);
  // gem5cudaSetupPtxSimArgument((function_info*)finfo, arg);
}

}


