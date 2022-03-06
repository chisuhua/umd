#include "IRuntime.h"

namespace rt {


IRuntime::IRuntime() {
// Monitor SvmBuffer::AllocatedLock_ ROCCLR_INIT_PRIORITY(101) ("Guards SVM allocation list");
// std::map<uintptr_t, uintptr_t> SvmBuffer::Allocated_ ROCCLR_INIT_PRIORITY(101);
};

// The allocation flags are ignored for now.
void* IRuntime::allocMemory(Context& context, size_t size, size_t alignment, uint32_t flags,
                        const Device* curDev) {
  void* ret = context.svmAlloc(size, alignment, flags, curDev);
  if (ret == nullptr) {
    LogError("Unable to allocate aligned memory");
    return nullptr;
  }
  uintptr_t ret_u = reinterpret_cast<uintptr_t>(ret);
  Add(ret_u, ret_u + size);
  return ret;
}

void IRuntime::freeMemory(const Context& context, void* ptr) {
  Remove(reinterpret_cast<uintptr_t>(ptr));
  context.svmFree(ptr);
}

bool IRuntime::isMemAlloced(const void* ptr) { return Contains(reinterpret_cast<uintptr_t>(ptr)); }

void IRuntime::Add(uintptr_t k, uintptr_t v) {
  ScopedLock lock(AllocatedLock_);
  Allocated_.insert(std::pair<uintptr_t, uintptr_t>(k, v));
}

void IRuntime::Remove(uintptr_t k) {
  ScopedLock lock(AllocatedLock_);
  Allocated_.erase(k);
}

bool IRuntime::Contains(uintptr_t ptr) {
  ScopedLock lock(AllocatedLock_);
  auto it = Allocated_.upper_bound(ptr);
  if (it == Allocated_.begin()) {
    return false;
  }
  --it;
  return ptr >= it->first && ptr < it->second;
}

volatile bool Runtime::initialized_ = false;
inline bool Runtime::initialized() { return initialized_; }

bool Runtime::init() {
  if (initialized_) {
    return true;
  }
  // Enter a very basic critical region. We want to prevent 2 threads
  // from concurrently executing the init() routines. We can't use a
  // Monitor since the system is not yet initialized.

  static std::atomic_flag lock = ATOMIC_FLAG_INIT;
  struct CriticalRegion {
    std::atomic_flag& lock_;
    CriticalRegion(std::atomic_flag& lock) : lock_(lock) {
      while (lock.test_and_set(std::memory_order_acquire)) {
        Os::yield();
      }
    }
    ~CriticalRegion() { lock_.clear(std::memory_order_release); }
  } region(lock);

  if (initialized_) {
    return true;
  }
  if (!Flag::init() || !option::init() || !Device::init()
      // Agent initializes last
      || !Agent::init()) {
    ClPrint(LOG_ERROR, LOG_INIT, "Runtime initilization failed");
    return false;
  }

  initialized_ = true;
  ClTrace(LOG_DEBUG, LOG_INIT);
  return true;
}

void Runtime::tearDown() {
  if (!initialized_) {
    return;
  }
  ClTrace(LOG_DEBUG, LOG_INIT);

  Agent::tearDown();
  Device::tearDown();
  option::teardown();
  Flag::tearDown();
  if (outFile != stderr && outFile != nullptr) {
    fclose(outFile);
  }
  initialized_ = false;
}



}
