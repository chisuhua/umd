#include <assert.h>
#include "Umd.h"
#include "../../libcuda/CUctx.h"
#include "KernelDispInfo.h"
#include "program/Program.h"
#include "program/loader/Loader.h"
#include "platform/IPlatform.h"

class IMemRegion;
class IAgent;
struct dim3
{
    unsigned int x, y, z;
};

loader::Executable* Umd::load_program(const std::string& file) {
    return LoadProgram(file, m_ctx, m_ctx->get_agent());
};

void Umd::set_kernel_disp(const std::string& kernel_name, loader::Executable* exec, DispatchInfo** disp_info, dim3 gridDim, dim3 blockDim, uint64_t param_addr) {
    loader::Symbol * sym = exec->GetSymbol((kernel_name + ".kd").c_str(), m_ctx->get_agent());
    DispatchInfo* dinfo = new DispatchInfo;
    sym->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, (void*)&dinfo->kernel_prog_addr);
    sym->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CTRL, (void*)&dinfo->kernel_ctrl);
    sym->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_MODE, (void*)&dinfo->kernel_mode);
    dinfo->grid_dim_x = gridDim.x;
    dinfo->grid_dim_y = gridDim.y;
    dinfo->grid_dim_z = gridDim.z;
    dinfo->block_dim_x = blockDim.x;
    dinfo->block_dim_y = blockDim.y;
    dinfo->block_dim_z = blockDim.z;
    dinfo->kernel_param_addr = param_addr;
    *disp_info = dinfo;
}

status_t Umd::memory_register(void* address, size_t size) {
    return m_platform->memory_register(address, size);
}

status_t Umd::memory_deregister(void* address, size_t size) {
    return m_platform->memory_deregister(address, size);
}

status_t Umd::memory_allocate(size_t size, void** ptr, IMemRegion *region) {
    return m_platform->memory_allocate(size, ptr, region);
}

status_t Umd::memory_copy(void* dst, const void* src, size_t count, UmdMemcpyKind kind) {
    return m_platform->memory_copy(dst, src, count, kind);
}

status_t Umd::memory_free(void* ptr) {
    return m_platform->memory_free(ptr);
}

/*
status_t Umd::memory_copy(void* ptr) {
    assert(false);
}
*/

IMemRegion* Umd::get_system_memregion() {
    return m_platform->get_system_memregion();
}

IMemRegion* Umd::get_device_memregion(IAgent* agent) {
    return m_platform->get_device_memregion(agent);
}

status_t Umd::free_memregion(IMemRegion *region) {
    return m_platform->free_memregion(region);
}
