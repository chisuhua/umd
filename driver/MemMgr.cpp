#include "MemMgr.h"

Monitor MemObjMap::AllocatedLock_ ROCCLR_INIT_PRIORITY(101) ("Guards MemObjMap allocation list");
std::map<uintptr_t, drv::Memory*> MemObjMap::MemObjMap_ ROCCLR_INIT_PRIORITY(101);

size_t MemObjMap::size() {
  drv::ScopedLock lock(AllocatedLock_);
  return MemObjMap_.size();
}

void MemObjMap::RemoveMemObj(const void* k) {
  drv::ScopedLock lock(AllocatedLock_);
  auto rval = MemObjMap_.erase(reinterpret_cast<uintptr_t>(k));
  if (rval != 1) {
    DevLogPrintfError("Memobj map does not have ptr: 0x%x",
                      reinterpret_cast<uintptr_t>(k));
    guarantee(false, "Memobj map does not have ptr");
  }
}

drv::Memory* MemObjMap::FindMemObj(const void* k) {
  drv::ScopedLock lock(AllocatedLock_);
  uintptr_t key = reinterpret_cast<uintptr_t>(k);
  auto it = MemObjMap_.upper_bound(key);
  if (it == MemObjMap_.begin()) {
    return nullptr;
  }

  --it;
  drv::Memory* mem = it->second;
  if (key >= it->first && key < (it->first + mem->getSize())) {
    // the k is in the range
    return mem;
  } else {
    return nullptr;
  }
}

void MemObjMap::UpdateAccess(drv::Device *peerDev) {
  if (peerDev == nullptr) {
    return;
  }

  // Provides access to all memory allocated on peerDev but
  // hsa_amd_agents_allow_access was not called because there was no peer
  drv::ScopedLock lock(AllocatedLock_);
  for (auto it : MemObjMap_) {
    const std::vector<Device*>& devices = it.second->getContext().devices();
    if (devices.size() == 1 && devices[0] == peerDev) {
      device::Memory* devMem = it.second->getDeviceMemory(*devices[0]);
      if (!devMem->getAllowedPeerAccess()) {
        peerDev->deviceAllowAccess(reinterpret_cast<void*>(it.first));
        devMem->setAllowedPeerAccess(true);
      }
    }
  }
}

void MemObjMap::Purge(drv::Device* dev) {
  assert(dev != nullptr);

  drv::ScopedLock lock(AllocatedLock_);
  for (auto it = MemObjMap_.cbegin(); it != MemObjMap_.cend(); ) {
    drv::Memory* memObj = it->second;
    unsigned int flags = memObj->getMemFlags();
    const std::vector<Device*>& devices = memObj->getContext().devices();
    if (devices.size() == 1 && devices[0] == dev && !(flags & ROCCLR_MEM_INTERNAL_MEMORY)) {
      it = MemObjMap_.erase(it);
    } else {
      ++it;
    }
  }
}

void MemObjMap::AddMemObj(const void* k, drv::Memory* v) {
  drv::ScopedLock lock(AllocatedLock_);
  auto rval = MemObjMap_.insert({ reinterpret_cast<uintptr_t>(k), v });
  if (!rval.second) {
    DevLogPrintfError("Memobj map already has an entry for ptr: 0x%x",
                      reinterpret_cast<uintptr_t>(k));
    guarantee(false, "Memobj map already has an entry for ptr");
  }
}


void Memory::saveMapInfo(const void* mapAddress, const drv::Coord3D origin,
                         const drv::Coord3D region, uint mapFlags, bool entire,
                         drv::Image* baseMip) {
  // Map/Unmap must be serialized.
  drv::ScopedLock lock(owner()->lockMemoryOps());

  WriteMapInfo info = {};
  WriteMapInfo* pInfo = &info;
  auto it = writeMapInfo_.find(mapAddress);
  if (it != writeMapInfo_.end()) {
    LogWarning("Double map of the same or overlapped region!");
    pInfo = &it->second;
  }

  if (mapFlags & (CL_MAP_WRITE | CL_MAP_WRITE_INVALIDATE_REGION)) {
    pInfo->origin_ = origin;
    pInfo->region_ = region;
    pInfo->entire_ = entire;
    pInfo->unmapWrite_ = true;
  }
  if (mapFlags & CL_MAP_READ) {
    pInfo->unmapRead_ = true;
  }
  pInfo->baseMip_ = baseMip;

  // Insert into the map if it's the first region
  if (++pInfo->count_ == 1) {
    writeMapInfo_.insert({mapAddress, info});
  }
}

