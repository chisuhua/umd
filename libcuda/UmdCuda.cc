#include "../../libcuda/CUctx.h"
#include "../../libcuda/abstract_hardware_model.h"
#include "../../libcuda/gpu-sim.h"
#include "UmdCuda.h"

extern "C" IPlatform* create_platform() {
  IPlatform* platform = new UmdCuda();
  return platform;
};


status_t UmdCuda::memory_register(void* address, size_t size) {
  if (size == 0 && address != NULL) {
    return ERROR_INVALID_ARGUMENT;
  }

  return SUCCESS;
}

status_t UmdCuda::memory_deregister(void* address, size_t size) {
  return SUCCESS;
}

status_t UmdCuda::memory_allocate(size_t size, void** ptr, IMemRegion *region) {
  *ptr = m_ctx->get_device()->get_gpgpu()->gpu_malloc(size);
  return SUCCESS;
}

status_t UmdCuda::memory_free(void* ptr) {
  return SUCCESS;
}

IMemRegion* UmdCuda::get_system_memregion() {
  return nullptr;
}

IMemRegion* UmdCuda::get_device_memregion(IAgent* agent) {
  return nullptr;
}

status_t UmdCuda::free_memregion(IMemRegion *region) {
  return SUCCESS;
}
