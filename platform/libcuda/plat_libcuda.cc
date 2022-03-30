#include "../../libcuda/abstract_hardware_model.h"
#include "../../libcuda/gpu-sim.h"
#include "../../libcuda/stream_manager.h"
#include "../../libcuda/gpgpu_context.h"
#include "driver/cuda/Context.h"
#include "plat_libcuda.h"
#include <map>
#include <typeinfo>

using namespace libcuda;

extern "C" IPlatform* create_platform(IContext* ctx) {
    static std::unordered_map<std::string, IPlatform*> platform_instance;
    std::string instance_name = "default";
    assert (ctx != nullptr) ;
    instance_name = typeid(*ctx).name();
    if (platform_instance.count(instance_name) == 0) {
        platform_instance[instance_name] = new plat_libcuda(instance_name, dynamic_cast<drv::Context*>(ctx));
    }
    return platform_instance[instance_name];
};


status_t plat_libcuda::memory_register(void* address, size_t size) {
  if (size == 0 && address != NULL) {
    return ERROR_INVALID_ARGUMENT;
  }

  return SUCCESS;
}

status_t plat_libcuda::memory_deregister(void* address, size_t size) {
  return SUCCESS;
}

IMemRegion* sys_memregion;
status_t plat_libcuda::memory_allocate(size_t size, void** ptr, IMemRegion *region) {
  if (region != nullptr) {
    *ptr = malloc(size);
  } else {
    *ptr = cuctx_->get()->get_device()->get_gpgpu()->gpu_malloc(size);
  }
  return SUCCESS;
}

status_t plat_libcuda::memory_copy(void* dst, const void* src, size_t count, UmdMemcpyKind kind) {
    if (kind == UmdMemcpyKind::HostToDevice)
      cuctx_->get()->get_device()->get_gpgpu()->gpgpu_ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation(src, (size_t)dst, count, 0));
    else if (kind == UmdMemcpyKind::DeviceToHost)
      cuctx_->get()->get_device()->get_gpgpu()->gpgpu_ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation((size_t)src, dst, count, 0));
    else if (kind == UmdMemcpyKind::DeviceToDevice)
      cuctx_->get()->get_device()->get_gpgpu()->gpgpu_ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation((size_t)src, (size_t)dst, count, 0));
    else if (kind == UmdMemcpyKind::Default) {
      if ((size_t)src >= GLOBAL_HEAP_START) {
        if ((size_t)dst >= GLOBAL_HEAP_START)
          cuctx_->get()->get_device()->get_gpgpu()->gpgpu_ctx->the_gpgpusim->g_stream_manager->push(stream_operation(
              (size_t)src, (size_t)dst, count, 0));  // device to device
        else
          cuctx_->get()->get_device()->get_gpgpu()->gpgpu_ctx->the_gpgpusim->g_stream_manager->push(
              stream_operation((size_t)src, dst, count, 0));  // device to host
      } else {
        if ((size_t)dst >= GLOBAL_HEAP_START)
          cuctx_->get()->get_device()->get_gpgpu()->gpgpu_ctx->the_gpgpusim->g_stream_manager->push(
              stream_operation(src, (size_t)dst, count, 0));
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

status_t plat_libcuda::memory_free(void* ptr) {
  return SUCCESS;
}

IMemRegion* plat_libcuda::get_system_memregion() {
  // FIXME fake
  uint64_t *tmp = (uint64_t*)&sys_memregion;
  *tmp = 1;
  return sys_memregion;
}

IMemRegion* plat_libcuda::get_device_memregion(IAgent* agent) {
  return nullptr;
}

status_t plat_libcuda::free_memregion(IMemRegion *region) {
  return SUCCESS;
}

extern "C" {
status_t libcuda_launchKernel(IPlatform*, const void *hostFunction) {};

status_t libcuda_setupKernelArgument(IPlatform*, const void *arg, size_t size,
                                      size_t offset) {};

}
