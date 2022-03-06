#include "MemPoolMgr.h"

#include <algorithm>
#include <cstring>
#include <set>


#include "Runtime.h"
#include "Device.h"

// Tracks aggregate size of system memory available on platform
size_t MemPoolMgr::max_sysmem_alloc_size_ = 0;

void* MemPoolMgr::AllocateKfdMemory(const HsaMemFlags& flag,
    HSAuint32 node_id, size_t size)
{
    void* ret = NULL;
    const device_status_t status = hal_dev_->AllocMemory(node_id, size, flag, &ret);
    return (status == DEVICE_STATUS_SUCCESS) ? ret : NULL;
}

void MemPoolMgr::FreeKfdMemory(void* ptr, size_t size)
{
    if (ptr == NULL || size == 0) {
        return;
    }

    //device_status_t status = hal_dev_->FreeMemory(ptr, size);
    //assert(status == DEVICE_STATUS_SUCCESS);
    hal_dev_->FreeMemory(ptr, size);
}

bool MemPoolMgr::RegisterMemory(void* ptr, size_t size, const HsaMemFlags& MemFlags)
{
    assert(ptr != NULL);
    assert(size != 0);

    const device_status_t status = hal_dev_->RegisterMemoryWithFlags(ptr, size, MemFlags);
    return (status == DEVICE_STATUS_SUCCESS);
}

void MemPoolMgr::DeregisterMemory(void* ptr) { hal_dev_->DeregisterMemory(ptr); }

bool MemPoolMgr::MakeKfdMemoryResident(size_t num_node, const uint32_t* nodes, const void* ptr,
    size_t size, uint64_t* alternate_va,
    HsaMemMapFlags map_flag)
{
    assert(num_node > 0);
    assert(nodes != NULL);

    *alternate_va = 0;
    const device_status_t status = hal_dev_->MapMemoryToGPUNodes(
        const_cast<void*>(ptr), size, alternate_va, map_flag, num_node, const_cast<uint32_t*>(nodes));

    return (status == DEVICE_STATUS_SUCCESS);
}

void MemPoolMgr::MakeKfdMemoryUnresident(const void* ptr)
{
    hal_dev_->UnmapMemoryToGPU(const_cast<void*>(ptr));
}

MemPoolMgr::MemPoolMgr(bool fine_grain, bool full_profile, core::IAgent* owner,
    const HsaMemoryProperties& mem_props)
    : core::IMemPoolMgr(fine_grain, full_profile, owner)
    , mem_props_(mem_props)
    , max_single_alloc_size_(0)
    , virtual_size_(0)
    , fragment_allocator_(BlockAllocator(*this))
{
    virtual_size_ = GetPhysicalSize();

    mem_flag_.Value = 0;
    map_flag_.Value = 0;

    static const HSAuint64 kGpuVmSize = (1ULL << 40);

    if (IsLocalMemory()) {
        mem_flag_.ui32.PageSize = HSA_PAGE_SIZE_4KB;
        mem_flag_.ui32.NoSubstitute = 1;
        mem_flag_.ui32.HostAccess = (mem_props_.HeapType == HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE) ? 0 : 1;
        mem_flag_.ui32.NonPaged = 1;

        map_flag_.ui32.PageSize = HSA_PAGE_SIZE_4KB;

        virtual_size_ = kGpuVmSize;
    } else if (IsSystem()) {
        mem_flag_.ui32.PageSize = HSA_PAGE_SIZE_4KB;
        mem_flag_.ui32.NoSubstitute = 1;
        mem_flag_.ui32.HostAccess = 1;
        mem_flag_.ui32.CachePolicy = HSA_CACHING_CACHED;

        virtual_size_ = (full_profile) ? os::GetUserModeVirtualMemorySize() : kGpuVmSize;
    }

    // Bind if memory region is coarse or fine grain
    mem_flag_.ui32.CoarseGrain = (fine_grain) ? 0 : 1;

    // Adjust allocatable size per page align
    max_single_alloc_size_ = alignDown(static_cast<size_t>(GetPhysicalSize()), kPageSize_);

    // Keep track of total system memory available
    // @note: System memory is surfaced as both coarse
    // and fine grain memory regions. To track total system
    // memory only fine grain is considered as it avoids
    // double counting
    if (IsSystem() && (fine_grain)) {
        max_sysmem_alloc_size_ += max_single_alloc_size_;
    }

    assert(GetVirtualSize() != 0);
    assert(GetPhysicalSize() <= GetVirtualSize());
    assert(isMultipleOf(max_single_alloc_size_, kPageSize_));
}

MemPoolMgr::~MemPoolMgr() { }

status_t MemPoolMgr::Allocate(size_t& size, AllocateFlags alloc_flags, void** address) const
{
    if (address == NULL) {
        return ERROR_INVALID_ARGUMENT;
    }

    if (!IsSystem() && !IsLocalMemory()) {
        return ERROR_INVALID_ALLOCATION;
    }

    // Alocation requests for system memory considers aggregate
    // memory available on all CPU devices
    if (size > ((IsSystem() ? max_sysmem_alloc_size_ : max_single_alloc_size_))) {
        return ERROR_INVALID_ALLOCATION;
    }

    size = alignUp(size, kPageSize_);

    HsaMemFlags kmt_alloc_flags(mem_flag_);
    kmt_alloc_flags.ui32.ExecuteAccess = (alloc_flags & AllocateExecutable ? 1 : 0);
    kmt_alloc_flags.ui32.AQLQueueMemory = (alloc_flags & AllocateDoubleMap ? 1 : 0);
    if (IsSystem() && (alloc_flags & AllocateIPC))
        kmt_alloc_flags.ui32.NonPaged = 1;

    // Only allow using the suballocator for ordinary VRAM.
    if (IsLocalMemory()) {
        bool subAllocEnabled = !Runtime::runtime_singleton_->flag().disable_fragment_alloc();
        // Avoid modifying executable or queue allocations.
        bool useSubAlloc = subAllocEnabled;
        useSubAlloc &= ((alloc_flags & (~AllocateRestrict)) == 0);
        useSubAlloc &= (size <= fragment_allocator_.max_alloc());
        if (useSubAlloc) {
            *address = fragment_allocator_.alloc(size);
            return SUCCESS;
        }
        if (subAllocEnabled) {
            // Pad up larger VRAM allocations.
            size = alignUp(size, fragment_allocator_.max_alloc());
        }
    }

    // Allocate memory.
    // If it fails attempt to release memory from the block allocator and retry.
    *address = AllocateKfdMemory(kmt_alloc_flags, owner()->node_id(), size);
    if (*address == nullptr) {
        fragment_allocator_.trim();
        *address = AllocateKfdMemory(kmt_alloc_flags, owner()->node_id(), size);
    }

    if (*address != nullptr) {
        // Commit the memory.
        // For system memory, on non-restricted allocation, map it to all GPUs. On
        // restricted allocation, only CPU is allowed to access by default, so
        // no need to map
        // For local memory, only map it to the owning GPU. Mapping to other GPU,
        // if the access is allowed, is performed on AllowAccess.
        HsaMemMapFlags map_flag = map_flag_;
        size_t map_node_count = 1;
        const uint32_t owner_node_id = owner()->node_id();
        const uint32_t* map_node_id = &owner_node_id;

        if (IsSystem()) {
            if ((alloc_flags & AllocateRestrict) == 0) {
                // Map to all GPU agents.
                map_node_count = Runtime::runtime_singleton_->gpu_ids().size();

                if (map_node_count == 0) {
                    // No need to pin since no GPU in the platform.
                    return SUCCESS;
                }

                map_node_id = &Runtime::runtime_singleton_->gpu_ids()[0];
            } else {
                // No need to pin it for CPU exclusive access.
                return SUCCESS;
            }
        }

        uint64_t alternate_va = 0;
        const bool is_resident = MakeKfdMemoryResident(
            map_node_count, map_node_id, *address, size, &alternate_va, map_flag);

        const bool require_pinning = (!full_profile() || IsLocalMemory() || IsScratch());

        if (require_pinning && !is_resident) {
            FreeKfdMemory(*address, size);
            *address = NULL;
            return ERROR_OUT_OF_RESOURCES;
        }

        return SUCCESS;
    }

    return ERROR_OUT_OF_RESOURCES;
}

status_t MemPoolMgr::Free(void* address, size_t size) const
{
    if (fragment_allocator_.free(address)) return SUCCESS;

    MakeKfdMemoryUnresident(address);

    FreeKfdMemory(address, size);

    return SUCCESS;
}

// TODO:  Look into a better name and/or making this process transparent to exporting.
status_t MemPoolMgr::IPCFragmentExport(void* address) const
{
    if (!fragment_allocator_.discardBlock(address)) return ERROR_INVALID_ALLOCATION;
    return SUCCESS;
}

status_t MemPoolMgr::GetInfo(region_info_t attribute,
    void* value) const
{
    switch (attribute) {
    case HSA_REGION_INFO_SEGMENT:
        switch (mem_props_.HeapType) {
        case HSA_HEAPTYPE_SYSTEM:
        case HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE:
        case HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC:
            *((hsa_region_segment_t*)value) = HSA_REGION_SEGMENT_GLOBAL;
            break;
        case HSA_HEAPTYPE_GPU_LDS:
            *((hsa_region_segment_t*)value) = HSA_REGION_SEGMENT_GROUP;
            break;
        default:
            assert(false && "Memory region should only be global, group");
            break;
        }
        break;
    case HSA_REGION_INFO_GLOBAL_FLAGS:
        switch (mem_props_.HeapType) {
        case HSA_HEAPTYPE_SYSTEM:
            *((uint32_t*)value) = fine_grain()
                ? (HSA_REGION_GLOBAL_FLAG_KERNARG | HSA_REGION_GLOBAL_FLAG_FINE_GRAINED)
                : HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED;
            break;
        case HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE:
        case HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC:
            *((uint32_t*)value) = fine_grain() ? HSA_REGION_GLOBAL_FLAG_FINE_GRAINED
                                               : HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED;
            break;
        default:
            *((uint32_t*)value) = 0;
            break;
        }
        break;
    case HSA_REGION_INFO_SIZE:
        *((size_t*)value) = static_cast<size_t>(GetPhysicalSize());
        break;
    case HSA_REGION_INFO_ALLOC_MAX_SIZE:
        switch (mem_props_.HeapType) {
        case HSA_HEAPTYPE_SYSTEM:
            *((size_t*)value) = max_sysmem_alloc_size_;
            break;
        case HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE:
        case HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC:
        case HSA_HEAPTYPE_GPU_SCRATCH:
            *((size_t*)value) = max_single_alloc_size_;
            break;
        default:
            *((size_t*)value) = 0;
        }
        break;
    case HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED:
        switch (mem_props_.HeapType) {
        case HSA_HEAPTYPE_SYSTEM:
        case HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE:
        case HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC:
            *((bool*)value) = true;
            break;
        default:
            *((bool*)value) = false;
            break;
        }
        break;
    case HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE:
        switch (mem_props_.HeapType) {
        case HSA_HEAPTYPE_SYSTEM:
        case HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE:
        case HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC:
            *((size_t*)value) = kPageSize_;
            break;
        default:
            *((size_t*)value) = 0;
            break;
        }
        break;
    case HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT:
        switch (mem_props_.HeapType) {
        case HSA_HEAPTYPE_SYSTEM:
        case HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE:
        case HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC:
            *((size_t*)value) = kPageSize_;
            break;
        default:
            *((size_t*)value) = 0;
            break;
        }
        break;
    default:
        switch ((hsa_amd_region_info_t)attribute) {
        case HSA_AMD_REGION_INFO_HOST_ACCESSIBLE:
            *((bool*)value) = (mem_props_.HeapType == HSA_HEAPTYPE_SYSTEM) ? true : false;
            break;
        case HSA_AMD_REGION_INFO_BASE:
            *((void**)value) = reinterpret_cast<void*>(GetBaseAddress());
            break;
        case HSA_AMD_REGION_INFO_BUS_WIDTH:
            *((uint32_t*)value) = BusWidth();
            break;
        case HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY:
            *((uint32_t*)value) = MaxMemCloc();
            break;
        default:
            return ERROR_INVALID_ARGUMENT;
            break;
        }
        break;
    }
    return SUCCESS;
}

status_t MemPoolMgr::GetPoolInfo(hsa_amd_memory_pool_info_t attribute,
    void* value) const
{
    switch (attribute) {
    case HSA_AMD_MEMORY_POOL_INFO_SEGMENT:
    case HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS:
    case HSA_AMD_MEMORY_POOL_INFO_SIZE:
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED:
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE:
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT:
        return GetInfo(static_cast<region_info_t>(attribute), value);
        break;
    case HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL:
        *((bool*)value) = IsSystem() ? true : false;
        break;
    case HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE:
        switch (mem_props_.HeapType) {
        case HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE:
        case HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC:
        case HSA_HEAPTYPE_GPU_SCRATCH:
            return GetInfo(HSA_REGION_INFO_ALLOC_MAX_SIZE, value);
        case HSA_HEAPTYPE_SYSTEM:
            // Aggregate size available for allocation
            *((size_t*)value) = max_sysmem_alloc_size_;
            break;
        default:
            *((size_t*)value) = 0;
        }
        break;
    default:
        return ERROR_INVALID_ARGUMENT;
    }

    return SUCCESS;
}

hsa_amd_memory_pool_access_t MemPoolMgr::GetAccessInfo(
    const core::IAgent& agent, const LinkInfo& link_info) const
{

    // Return allowed by default if memory pool is owned by requesting device
    // if (agent.public_handle().handle == owner()->public_handle().handle) {
    if (&agent == owner()) {
        return HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT;
    }

    // Requesting device does not have a link
    if (link_info.num_hop < 1) {
        return HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
    }

    // Determine access to fine and coarse grained system memory
    // Return allowed by default if requesting device is a CPU
    // Return disallowed by default if requesting device is not a CPU
    if (IsSystem()) {
        return (agent.agent_type() == core::IAgent::kCpu) ? (HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT) : (HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT);
    }

    // Determine access type for device local memory which is
    // guaranteed to be HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC
    // Return disallowed by default if framebuffer is coarse grained
    // without regard to type of requesting device (CPU / GPU)
    // Return disallowed by default if framebuffer is fine grained
    // and requesting device is connected via xGMI link
    // Return never allowed if framebuffer is fine grained and
    // requesting device is connected via PCIe link
    if (IsLocalMemory()) {

        // Return disallowed by default if memory is coarse
        // grained without regard to link type
        if (fine_grain() == false) {
            return HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT;
        }

        // Determine if pool is pseudo fine-grained due to env flag
        // Return disallowed by default
        /* FIXME
    if (core::IRuntime::runtime_singleton_->flag().fine_grain_pcie()) {
      return HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT;
    }
    */

        // Return disallowed by default if memory is fine
        // grained and link type is xGMI.
        if (agent.HiveId() == owner()->HiveId()) {
            return HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT;
        }

        // Return never allowed if memory is fine grained
        // link type is not xGMI i.e. link is PCIe
        return HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
    }

    // Return never allowed if above conditions are not satisified
    // This can happen when memory pool references neither system
    // or device local memory
    return HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
}

status_t MemPoolMgr::GetAgentPoolInfo(
    const core::IAgent& agent, hsa_amd_agent_memory_pool_info_t attribute,
    void* value) const
{
    const uint32_t node_id_from = agent.node_id();
    const uint32_t node_id_to = owner()->node_id();

    const LinkInfo link_info = Runtime::runtime_singleton_->GetLinkInfo(node_id_from, node_id_to);

    const hsa_amd_memory_pool_access_t access_type = GetAccessInfo(agent, link_info);
    /**
   *  ---------------------------------------------------
   *  |              |CPU        |GPU (owner)|GPU (peer) |
   *  ---------------------------------------------------
   *  |system memory |allowed    |disallowed |disallowed |
   *  ---------------------------------------------------
   *  |fb private    |never      |allowed    |never      |
   *  ---------------------------------------------------
   *  |fb public     |disallowed |allowed    |disallowed |
   *  ---------------------------------------------------
   *  |others        |never      |allowed    |never      |
   *  ---------------------------------------------------
   */
    /*
  const hsa_amd_memory_pool_access_t access_type =
      ((IsSystem() && (agent.agent_type() == core::IAgent::kCpu)) ||
       (agent.node_id() == owner()->node_id()))
          ? HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT
          : (IsSystem() || (IsPublic() && link_info.num_hop > 0))
                ? HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT
                : HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
                */

    switch (attribute) {
    case HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS:
        *((hsa_amd_memory_pool_access_t*)value) = access_type;
        break;
    case HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS:
        *((uint32_t*)value) = (access_type != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED)
            ? link_info.num_hop
            : 0;
        break;
    case HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO:
        memset(value, 0, sizeof(hsa_amd_memory_pool_link_info_t));
        if ((access_type != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) && (link_info.num_hop > 0)) {
            memcpy(value, &link_info.info, sizeof(hsa_amd_memory_pool_link_info_t));
        }
        break;
    default:
        return ERROR_INVALID_ARGUMENT;
    }
    return SUCCESS;
}

status_t MemPoolMgr::AllowAccess(uint32_t num_agents,
    core::IAgent** agents,
    const void* ptr, size_t size) const
{
    if (num_agents == 0 || agents == NULL || ptr == NULL || size == 0) {
        return ERROR_INVALID_ARGUMENT;
    }

    if (!IsSystem() && !IsLocalMemory()) {
        return ERROR;
    }

    // Adjust for fragments.  Make accessibility sticky for fragments since this will satisfy the
    // union of accessible agents between the fragments in the block.
    hsa_amd_pointer_info_t info;
    uint32_t agent_count = 0;
    core::IAgent* accessible = nullptr;
    MAKE_SCOPE_GUARD([&]() { free(accessible); });
    Runtime::PtrInfoBlockData blockInfo;
    std::vector<core::IAgent*> union_agents;
    info.size = sizeof(info);

    ScopedAcquire<KernelMutex> lock(&access_lock_);
    if (Runtime::runtime_singleton_->PtrInfo(const_cast<void*>(ptr), &info, malloc,
            &agent_count, &accessible,
            &blockInfo)
        == SUCCESS) {
        if (blockInfo.length != size || info.sizeInBytes != size) {
            for (uint32_t i = 0; i < num_agents; i++)
                union_agents.push_back(const_cast<core::IAgent*>(agents[i]));
            for (uint32_t i = 0; i < agent_count; i++)
                union_agents.push_back(&accessible[i]);
            std::sort(union_agents.begin(), union_agents.end());
            const auto& last = std::unique(union_agents.begin(), union_agents.end());
            union_agents.erase(last, union_agents.end());

            // agents = reinterpret_cast<agent_t*>(&union_agents[0]);
            agents = &union_agents[0];
            num_agents = union_agents.size();
            size = blockInfo.length;
            ptr = blockInfo.base;
        }
    }

    bool cpu_in_list = false;

    std::set<GpuAgentInt*> whitelist_gpus;
    std::vector<uint32_t> whitelist_nodes;
    for (uint32_t i = 0; i < num_agents; ++i) {
        core::IAgent* agent = agents[i]; // core::Agent::Object(agents[i]);
        if (agent == NULL) {
            return ERROR_INVALID_AGENT;
        }

        if (agent->agent_type() == core::IAgent::kGpu) {
            whitelist_nodes.push_back(agent->node_id());
            whitelist_gpus.insert(reinterpret_cast<GpuAgentInt*>(agent));
        } else {
            cpu_in_list = true;
        }
    }

    if (whitelist_nodes.size() == 0 && IsSystem()) {
        assert(cpu_in_list);
        // This is a system region and only CPU agents in the whitelist.
        // Remove old mappings.
        MemPoolMgr::MakeKfdMemoryUnresident(ptr);
        return SUCCESS;
    }

    // If this is a local memory region, the owning gpu always needs to be in
    // the whitelist.
    // if (IsPublic() &&
    if (IsLocalMemory() && std::find(whitelist_nodes.begin(), whitelist_nodes.end(), owner()->node_id()) == whitelist_nodes.end()) {
        whitelist_nodes.push_back(owner()->node_id());
        whitelist_gpus.insert(reinterpret_cast<GpuAgentInt*>(owner()));
    }

    HsaMemMapFlags map_flag = map_flag_;
    map_flag.ui32.HostAccess |= (cpu_in_list) ? 1 : 0;

    {
        ScopedAcquire<KernelMutex> lock(&Runtime::runtime_singleton_->memory_lock_);
        uint64_t alternate_va = 0;
        if (!MemPoolMgr::MakeKfdMemoryResident(
                whitelist_nodes.size(), &whitelist_nodes[0], ptr,
                size, &alternate_va, map_flag)) {
            return ERROR_OUT_OF_RESOURCES;
        }
    }

    lock.Release();
    /* TODO
  for (GpuDeviceInt* gpu : whitelist_gpus) {
    gpu->PreloadBlits();
  }
*/
    return SUCCESS;
}

status_t MemPoolMgr::CanMigrate(const MemPoolMgr& dst,
    bool& result) const
{
    // TODO: not implemented yet.
    result = false;
    return ERROR_OUT_OF_RESOURCES;
}

status_t MemPoolMgr::Migrate(uint32_t flag, const void* ptr) const
{
    // TODO: not implemented yet.
    return ERROR_OUT_OF_RESOURCES;
}

status_t MemPoolMgr::Lock(uint32_t num_agents, core::IAgent** agents,
    void* host_ptr, size_t size,
    void** agent_ptr) const
{
    if (!IsSystem()) {
        return ERROR;
    }

    if (full_profile()) { // FIXME
        // For APU, any host pointer is always accessible by the gpu.
        *agent_ptr = host_ptr;
        return SUCCESS;
    }

    std::set<core::IAgent*> whitelist_gpus;
    std::vector<HSAuint32> whitelist_nodes;
    if (num_agents == 0 || agents == NULL) {
        // Map to all GPU agents.
        whitelist_nodes = Runtime::runtime_singleton_->gpu_ids();

        whitelist_gpus.insert(
            Runtime::runtime_singleton_->gpu_agents().begin(),
            Runtime::runtime_singleton_->gpu_agents().end());
    } else {
        for (uint32_t i = 0; i < num_agents; ++i) {
            core::IAgent* agent = agents[i]; // core::Agent::Object(agents[i]);
            if (agent == NULL) {
                return ERROR_INVALID_AGENT;
            }

            if (agent->agent_type() == core::IAgent::kGpu) {
                whitelist_nodes.push_back(agent->node_id());
                whitelist_gpus.insert(agent);
            }
        }
    }

    if (whitelist_nodes.size() == 0) {
        // No GPU agents in the whitelist. So no need to register and map since the
        // platform only has CPUs.
        *agent_ptr = host_ptr;
        return SUCCESS;
    }

    // Call kernel driver to register and pin the memory.
    if (RegisterMemory(host_ptr, size, mem_flag_)) {
        uint64_t alternate_va = 0;
        if (MakeKfdMemoryResident(whitelist_nodes.size(), &whitelist_nodes[0],
                host_ptr, size, &alternate_va, map_flag_)) {
            if (alternate_va != 0) {
                *agent_ptr = reinterpret_cast<void*>(alternate_va);
            } else {
                *agent_ptr = host_ptr;
            }
            /*
      for (auto gpu : whitelist_gpus) {
        static_cast<GpuDeviceInt*>(gpu)->PreloadBlits();
      }
*/
            return SUCCESS;
        }
        MemPoolMgr::DeregisterMemory(host_ptr);
        return ERROR_OUT_OF_RESOURCES;
    }

    return ERROR;
}

status_t MemPoolMgr::Unlock(void* host_ptr) const
{
    if (!IsSystem()) {
        return ERROR;
    }

    if (full_profile()) {
        return SUCCESS;
    }

    MakeKfdMemoryUnresident(host_ptr);
    DeregisterMemory(host_ptr);

    return SUCCESS;
}

status_t MemPoolMgr::AssignAgent(void* ptr, size_t size,
    const core::IAgent& agent,
    hsa_access_permission_t access) const
{
    return SUCCESS;
}

void* MemPoolMgr::BlockAllocator::alloc(size_t request_size, size_t& allocated_size) const
{
    assert(request_size <= block_size() && "BlockAllocator alloc request exceeds block size.");

    void* ret;
    size_t bsize = block_size();
    status_t err = region_.Allocate(
        bsize, core::IMemPoolMgr::AllocateRestrict | core::IMemPoolMgr::AllocateDirect, &ret);
    if (err != SUCCESS) {
        // throw ::hcs::hsa_exception(err, "MemPoolMgr::BlockAllocator::alloc failed.");
        throw co_exception(err, "MemPoolMgr::BlockAllocator::alloc failed.");
    }
    assert(ret != nullptr && "Region returned nullptr on success.");

    allocated_size = block_size();
    return ret;
}

