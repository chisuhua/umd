
#include "runtime/Context.h"
// #include "runtime/CommandQueue.hpp"
//
#include <algorithm>
#include <functional>

#ifdef _WIN32
#include <d3d10_1.h>
#include <dxgi.h>
#include "CL/cl_d3d10.h"
#include "CL/cl_d3d11.h"
#include "CL/cl_dx9_media_sharing.h"
#endif  //_WIN32

namespace amd {

Context::Context(const std::vector<Device*>& devices, const Info& info)
    : devices_(devices),
      info_(info),
      properties_(NULL),
      glenv_(NULL),
      customHostAllocDevice_(NULL) {
  for (const auto& device : devices) {
    device->retain();
    if (customHostAllocDevice_ == NULL && device->customHostAllocator()) {
      customHostAllocDevice_ = device;
    }
    if (device->svmSupport()) {
      svmAllocDevice_.push_back(device);
    }
  }

  if (svmAllocDevice_.size() > 1) {
    uint isFirstDeviceFGSEnabled = svmAllocDevice_.front()->isFineGrainedSystem(true);
    for (auto& dev : svmAllocDevice_) {
      // allocation on fine - grained system incapable device first
      if (isFirstDeviceFGSEnabled && !dev->isFineGrainedSystem(true)) {
        std::swap(svmAllocDevice_.front(), dev);
        break;
      }
    }
  }
}

Context::~Context() {
  static const bool VALIDATE_ONLY = false;

  // Loop through all devices
  for (const auto& it : devices_) {
    // Notify device about context destroy
    it->ContextDestroy();

    // Release device
    it->release();
  }
}

void* Context::hostAlloc(size_t size, size_t alignment, bool atomics) const {
  return AlignedMemory::allocate(size, alignment);
}

void Context::hostFree(void* ptr) const {
  AlignedMemory::deallocate(ptr);
}

void* Context::svmAlloc(size_t size, size_t alignment, cl_svm_mem_flags flags,
                        const rt::Device* curDev) {
  unsigned int numSVMDev = svmAllocDevice_.size();
  if (numSVMDev < 1) {
    return nullptr;
  }

  void* svmPtrAlloced = nullptr;

  rt::ScopedLock lock(&ctxLock_);

  if (curDev != nullptr) {
    if (!(flags & CL_MEM_SVM_ATOMICS) ||
        (curDev->info().svmCapabilities_ & CL_DEVICE_SVM_ATOMICS)) {
      svmPtrAlloced = curDev->svmAlloc(*this, size, alignment, flags, svmPtrAlloced);
      if (svmPtrAlloced == nullptr) {
        return nullptr;
      }
    }
  }

  for (const auto& dev : svmAllocDevice_) {
    if (dev == curDev) {
      continue;
    }
    // check if the device support svm platform atomics,
    // skipped allocation for platform atomics if not supported by this device
    if ((flags & CL_MEM_SVM_ATOMICS) &&
        !(dev->info().svmCapabilities_ & CL_DEVICE_SVM_ATOMICS)) {
      continue;
    }
    svmPtrAlloced = dev->svmAlloc(*this, size, alignment, flags, svmPtrAlloced);
    if (svmPtrAlloced == nullptr) {
      return nullptr;
    }
  }
  return svmPtrAlloced;
}

void Context::svmFree(void* ptr) const {
  rt::ScopedLock lock(&ctxLock_);
  for (const auto& dev : svmAllocDevice_) {
    dev->svmFree(ptr);
  }
  return;
}

bool Context::containsDevice(const Device* device) const {
  for (const auto& it : devices_) {
    if (device == it) {
      return true;
    }
  }
  return false;
}

DeviceQueue* Context::defDeviceQueue(const Device& dev) const {
  const auto it = deviceQueues_.find(&dev);
  if (it != deviceQueues_.cend()) {
    return it->second.defDeviceQueue_;
  } else {
    return NULL;
  }
}

bool Context::isDevQueuePossible(const Device& dev) {
  return (deviceQueues_[&dev].deviceQueueCnt_ < dev.info().maxOnDeviceQueues_) ? true : false;
}

void Context::addDeviceQueue(const Device& dev, DeviceQueue* queue, bool defDevQueue) {
  DeviceQueueInfo& info = deviceQueues_[&dev];
  info.deviceQueueCnt_++;
  if (defDevQueue) {
    info.defDeviceQueue_ = queue;
  }
}

void Context::removeDeviceQueue(const Device& dev, DeviceQueue* queue) {
  DeviceQueueInfo& info = deviceQueues_[&dev];
  assert((info.deviceQueueCnt_ != 0) && "The device queue map is empty!");
  info.deviceQueueCnt_--;
  if (info.defDeviceQueue_ == queue) {
    info.defDeviceQueue_ = NULL;
  }
}

}  // namespace amd
