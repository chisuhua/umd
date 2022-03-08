#include "../../libcuda/CUctx.h"
#include "../../libcuda/abstract_hardware_model.h"
#include "../../libcuda/gpu-sim.h"
#include "../../libcuda/stream_manager.h"
#include "../../libcuda/gpgpu_context.h"
#include "Gem5Cuda.h"
#include "../../libcuda/gem5cuda/gem5cuda_runtime_api.h"

using namespace libcuda;

extern "C" IPlatform* create_platform() {
  IPlatform* platform = new Gem5Cuda();
  return platform;
};


status_t Gem5Cuda::memory_register(void* address, size_t size) {
  if (size == 0 && address != NULL) {
    return ERROR_INVALID_ARGUMENT;
  }

  return SUCCESS;
}

status_t Gem5Cuda::memory_deregister(void* address, size_t size) {
  return SUCCESS;
}

IMemRegion* sys_memregion;
status_t Gem5Cuda::memory_allocate(size_t size, void** ptr, IMemRegion *region) {
  if (region != nullptr) {
    *ptr = malloc(size);
  } else {
    *ptr = m_ctx->get_device()->get_gpgpu()->gpu_malloc(size);
  }
  return SUCCESS;
}

status_t Gem5Cuda::memory_copy(void* dst, const void* src, size_t count, UmdMemcpyKind kind) {
    if (kind == UmdMemcpyKind::HostToDevice)
      m_ctx->get_device()->get_gpgpu()->gpgpu_ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation(src, (size_t)dst, count, 0));
    else if (kind == UmdMemcpyKind::DeviceToHost)
      m_ctx->get_device()->get_gpgpu()->gpgpu_ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation((size_t)src, dst, count, 0));
    else if (kind == UmdMemcpyKind::DeviceToDevice)
      m_ctx->get_device()->get_gpgpu()->gpgpu_ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation((size_t)src, (size_t)dst, count, 0));
    else if (kind == UmdMemcpyKind::Default) {
      if ((size_t)src >= GLOBAL_HEAP_START) {
        if ((size_t)dst >= GLOBAL_HEAP_START)
          m_ctx->get_device()->get_gpgpu()->gpgpu_ctx->the_gpgpusim->g_stream_manager->push(stream_operation(
              (size_t)src, (size_t)dst, count, 0));  // device to device
        else
          m_ctx->get_device()->get_gpgpu()->gpgpu_ctx->the_gpgpusim->g_stream_manager->push(
              stream_operation((size_t)src, dst, count, 0));  // device to host
      } else {
        if ((size_t)dst >= GLOBAL_HEAP_START)
          m_ctx->get_device()->get_gpgpu()->gpgpu_ctx->the_gpgpusim->g_stream_manager->push(
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

status_t Gem5Cuda::memory_free(void* ptr) {
  return SUCCESS;
}

IMemRegion* Gem5Cuda::get_system_memregion() {
  // FIXME fake
  uint64_t *tmp = (uint64_t*)&sys_memregion;
  *tmp = 1;
  return sys_memregion;
}

IMemRegion* Gem5Cuda::get_device_memregion(IAgent* agent) {
  return nullptr;
}

status_t Gem5Cuda::free_memregion(IMemRegion *region) {
  return SUCCESS;
}

status_t Gem5Cuda::getDeviceCount(int* count) {
  gem5cudaGetDeviceCount(count);
  return SUCCESS;
};

status_t Gem5Cuda::getDeviceProperties(void* prop, int id) {
  gem5cudaGetDeviceProperties((cudaDeviceProp*)prop, id);
  return SUCCESS;
};

status_t Gem5Cuda::getDevice(int* device) {
  gem5cudaGetDevice((int*)the_device);
  return SUCCESS;
};


