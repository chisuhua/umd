#include "LoaderContext.h"
#include "umd.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <utility>
#include <sys/mman.h>

bool IsDebuggerRegistered()
{
    return false;
    // Leaving code commented as it will be used later on
    //return ((IRuntime::runtime_singleton_->flag().emulate_aql()) &&
    //        (0 !=
    //         IRuntime::runtime_singleton_->flag().tools_lib_names().size()));
}

class SegmentMemory {
public:
    virtual ~SegmentMemory() { }
    virtual void* Address(size_t offset = 0) const = 0;
    virtual void* HostAddress(size_t offset = 0) const = 0;
    virtual bool Allocated() const = 0;
    virtual bool Allocate(size_t size, size_t align, bool zero) = 0;
    virtual bool Copy(size_t offset, const void* src, size_t size) = 0;
    virtual void Free() = 0;
    virtual bool Freeze() = 0;

protected:
    SegmentMemory(CUctx* ctx) : m_ctx(ctx) { }
    CUctx* m_ctx;

private:
    SegmentMemory(const SegmentMemory&);
    SegmentMemory& operator=(const SegmentMemory&);
};

class MallocedMemory final : public SegmentMemory {
public:
    MallocedMemory(CUctx* ctx)
        : SegmentMemory(ctx)
        , ptr_(nullptr)
        , size_(0)
    {
    }
    ~MallocedMemory() { }

    void* Address(size_t offset = 0) const override
    {
        assert(this->Allocated());
        return (char*)ptr_ + offset;
    }
    void* HostAddress(size_t offset = 0) const override
    {
        return this->Address(offset);
    }
    bool Allocated() const override
    {
        return nullptr != ptr_;
    }

    bool Allocate(size_t size, size_t align, bool zero) override;
    bool Copy(size_t offset, const void* src, size_t size) override;
    void Free() override;
    bool Freeze() override;

private:
    MallocedMemory(const MallocedMemory&);
    MallocedMemory& operator=(const MallocedMemory&);

    void* ptr_;
    size_t size_;
};

bool MallocedMemory::Allocate(size_t size, size_t align, bool zero)
{
    assert(!this->Allocated());
    assert(0 < size);
    assert(0 < align && 0 == (align & (align - 1)));
    ptr_ = _aligned_malloc(size, align);
    if (nullptr == ptr_) {
        return false;
    }
    if (SUCCESS != Umd::get(m_ctx)->memory_register(ptr_, size)) {
        _aligned_free(ptr_);
        ptr_ = nullptr;
        return false;
    }
    if (zero) {
        memset(ptr_, 0x0, size);
    }
    size_ = size;
    return true;
}

bool MallocedMemory::Copy(size_t offset, const void* src, size_t size)
{
    assert(this->Allocated());
    assert(nullptr != src);
    assert(0 < size);
    memcpy(this->Address(offset), src, size);
    return true;
}

void MallocedMemory::Free()
{
    assert(this->Allocated());
    Umd::get(m_ctx)->memory_deregister(ptr_, size_);
    _aligned_free(ptr_);
    ptr_ = nullptr;
    size_ = 0;
}

bool MallocedMemory::Freeze()
{
    assert(this->Allocated());
    return true;
}

class MappedMemory final : public SegmentMemory {
public:
    MappedMemory(CUctx* ctx)
        : SegmentMemory(ctx)
        , ptr_(nullptr)
        , size_(0)
    {
    }
    ~MappedMemory() { }

    void* Address(size_t offset = 0) const override
    {
        assert(this->Allocated());
        return (char*)ptr_ + offset;
    }
    void* HostAddress(size_t offset = 0) const override
    {
        return this->Address(offset);
    }
    bool Allocated() const override
    {
        return nullptr != ptr_;
    }

    bool Allocate(size_t size, size_t align, bool zero) override;
    bool Copy(size_t offset, const void* src, size_t size) override;
    void Free() override;
    bool Freeze() override;

private:
    MappedMemory(const MappedMemory&);
    MappedMemory& operator=(const MappedMemory&);

    void* ptr_;
    size_t size_;
};

bool MappedMemory::Allocate(size_t size, size_t align, bool zero)
{
    assert(!this->Allocated());
    assert(0 < size);
    assert(0 < align && 0 == (align & (align - 1)));
#if defined(_WIN32) || defined(_WIN64)
    ptr_ = (void*)VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
    ptr_ =  mmap(nullptr, size, PROT_EXEC | PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_NORESERVE | MAP_PRIVATE, -1, 0);
#endif // _WIN32 || _WIN64
    if (nullptr == ptr_) {
        return false;
    }
    assert(0 == ((uintptr_t)ptr_) % align);
    if (SUCCESS != Umd::get(m_ctx)->memory_register(ptr_, size)) {
#if defined(_WIN32) || defined(_WIN64)
        VirtualFree(ptr_, size, MEM_DECOMMIT);
        VirtualFree(ptr_, 0, MEM_RELEASE);
#else
        munmap(ptr_, size);
#endif // _WIN32 || _WIN64
        ptr_ = nullptr;
        return false;
    }
    if (zero) {
        memset(ptr_, 0x0, size);
    }
    size_ = size;
    return true;
}

bool MappedMemory::Copy(size_t offset, const void* src, size_t size)
{
    assert(this->Allocated());
    assert(nullptr != src);
    assert(0 < size);
    memcpy(this->Address(offset), src, size);
    return true;
}

void MappedMemory::Free()
{
    assert(this->Allocated());
    Umd::get(m_ctx)->memory_deregister(ptr_, size_);
#if defined(_WIN32) || defined(_WIN64)
    VirtualFree(ptr_, size_, MEM_DECOMMIT);
    VirtualFree(ptr_, 0, MEM_RELEASE);
#else
    munmap(ptr_, size_);
#endif // _WIN32 || _WIN64
    ptr_ = nullptr;
    size_ = 0;
}

bool MappedMemory::Freeze()
{
    assert(this->Allocated());
    return true;
}

class RegionMemory final : public SegmentMemory {
public:
    RegionMemory(IMemRegion* region, CUctx *ctx)
        : SegmentMemory(ctx)
        , region_(region)
        , ptr_(nullptr)
        , host_ptr_(nullptr)
        , size_(0)
    {
    }
    ~RegionMemory() { }

    void* Address(size_t offset = 0) const override
    {
        assert(this->Allocated());
        return (char*)ptr_ + offset;
    }
    void* HostAddress(size_t offset = 0) const override
    {
        assert(this->Allocated());
        return (char*)host_ptr_ + offset;
    }
    bool Allocated() const override
    {
        return nullptr != ptr_;
    }

    bool Allocate(size_t size, size_t align, bool zero) override;
    bool Copy(size_t offset, const void* src, size_t size) override;
    void Free() override;
    bool Freeze() override;

private:
    RegionMemory(const RegionMemory&);
    RegionMemory& operator=(const RegionMemory&);

    IMemRegion* region_;
    void* ptr_;
    void* host_ptr_;
    size_t size_;
};

bool RegionMemory::Allocate(size_t size, size_t align, bool zero)
{
    assert(!this->Allocated());
    assert(0 < size);
    assert(0 < align && 0 == (align & (align - 1)));
    if (SUCCESS != Umd::get(m_ctx)->memory_allocate(size, &ptr_, region_)) {
        ptr_ = nullptr;
        return false;
    }
    assert(0 == ((uintptr_t)ptr_) % align);
    if (SUCCESS != Umd::get(m_ctx)->memory_allocate(size, &host_ptr_, Umd::get(m_ctx)->get_system_memregion())) {
        Umd::get(m_ctx)->memory_free(ptr_);
        ptr_ = nullptr;
        host_ptr_ = nullptr;
        return false;
    }
    if (zero) {
        memset(host_ptr_, 0x0, size);
    }
    size_ = size;
    return true;
}

bool RegionMemory::Copy(size_t offset, const void* src, size_t size)
{
    assert(this->Allocated() && nullptr != host_ptr_);
    assert(nullptr != src);
    assert(0 < size);
    memcpy((char*)host_ptr_ + offset, src, size);
    return true;
}

void RegionMemory::Free()
{
    assert(this->Allocated());
    Umd::get(m_ctx)->memory_free(ptr_);
    if (nullptr != host_ptr_) {
        Umd::get(m_ctx)->memory_free(host_ptr_);
    }
    ptr_ = nullptr;
    host_ptr_ = nullptr;
    size_ = 0;
}

bool RegionMemory::Freeze()
{
    assert(this->Allocated() && nullptr != host_ptr_);
    Umd::get(m_ctx)->free_memregion(region_);
/*
    IAgent* agent = region_->owner();
    if (agent != NULL && agent->agent_type() == IAgent::kGpu) {
        // if (SUCCESS != agent->DmaCopy(ptr_, host_ptr_, size_, DMA_D2H)) {
        if (SUCCESS != agent->DmaCopy(ptr_, host_ptr_, size_)) {
            return false;
        }
    } else {
        memcpy(ptr_, host_ptr_, size_);
    }
*/
    return true;
}

void* LoaderContext::SegmentAlloc(amdgpu_hsa_elf_segment_t segment,
    IAgent* agent,
    size_t size,
    size_t align,
    bool zero)
{
    assert(0 < size);
    assert(0 < align && 0 == (align & (align - 1)));

    SegmentMemory* mem = nullptr;
    switch (segment) {
    case AMDGPU_HSA_SEGMENT_GLOBAL_AGENT:
    case AMDGPU_HSA_SEGMENT_READONLY_AGENT: {
        mem = new (std::nothrow) RegionMemory(Umd::get(m_ctx)->get_device_memregion(agent), m_ctx);
        break;
    }
    case AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM: {
        mem = new (std::nothrow) RegionMemory(Umd::get(m_ctx)->get_system_memregion(), m_ctx);
        break;
    }
    case AMDGPU_HSA_SEGMENT_CODE_AGENT: {
        profile_t agent_profile;
        mem = new (std::nothrow) RegionMemory(IsDebuggerRegistered() ? Umd::get(m_ctx)->get_system_memregion() : Umd::get(m_ctx)->get_device_memregion(agent), m_ctx);
        /*
        if (SUCCESS != hsa_agent_get_info(agent, HSA_AGENT_INFO_PROFILE, &agent_profile)) {
            return nullptr;
        }

        switch (agent_profile) {
        case HSA_PROFILE_BASE:
            mem = new (std::nothrow) RegionMemory(IsDebuggerRegistered() ? RegionMemory::System() : RegionMemory::AgentLocal(agent));
            break;
        case HSA_PROFILE_FULL:
            mem = new (std::nothrow) MappedMemory();
            break;
        default:
            assert(false);
        }
        */

        // Invalidate agent caches which may hold lines of the new allocation.
        // ((GpuAgentInt*)IAgent::Object(agent))->InvalidateCodeCaches();
        // (dynamic_cast<GpuAgentInt*>(agent))->InvalidateCodeCaches();

        break;
    }
    default:
        assert(false);
    }

    if (nullptr == mem) {
        return nullptr;
    }

    if (!mem->Allocate(size, align, zero)) {
        delete mem;
        return nullptr;
    }

    return mem;
}

bool LoaderContext::SegmentCopy(amdgpu_hsa_elf_segment_t segment, // not used.
    IAgent* agent, // not used.
    void* dst,
    size_t offset,
    const void* src,
    size_t size)
{
    assert(nullptr != dst);
    return ((SegmentMemory*)dst)->Copy(offset, src, size);
}

void LoaderContext::SegmentFree(amdgpu_hsa_elf_segment_t segment, // not used.
    IAgent* agent, // not used.
    void* seg,
    size_t size) // not used.
{
    assert(nullptr != seg);
    SegmentMemory* mem = (SegmentMemory*)seg;
    mem->Free();
    delete mem;
    mem = nullptr;
}

void* LoaderContext::SegmentAddress(amdgpu_hsa_elf_segment_t segment, // not used.
    IAgent* agent, // not used.
    void* seg,
    size_t offset)
{
    assert(nullptr != seg);
    return ((SegmentMemory*)seg)->Address(offset);
}

void* LoaderContext::SegmentHostAddress(amdgpu_hsa_elf_segment_t segment, // not used.
    IAgent* agent, // not used.
    void* seg,
    size_t offset)
{
    assert(nullptr != seg);
    return ((SegmentMemory*)seg)->HostAddress(offset);
}

bool LoaderContext::SegmentFreeze(amdgpu_hsa_elf_segment_t segment, // not used.
    IAgent* agent, // not used.
    void* seg,
    size_t size) // not used.
{
    assert(nullptr != seg);
    return ((SegmentMemory*)seg)->Freeze();
}

