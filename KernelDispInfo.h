// we expect block block level in compute unit
// for grid level, it have gridStartX/Y/Z is used by CP
#define SHADER_ABI_USER_DATA_REGISTER_NUM_MAX 32
#include <stdint.h>

// FIXME it need to changed acording to kernel_ctrl
#if 0
union SHADER_ABI_KERNEL_CONTROL {
    struct {
        uint16_t    grid_dim_x_en       : 1;
        uint16_t    grid_dim_y_en       : 1;
        uint16_t    grid_dim_z_en       : 1;
        uint16_t    block_dim_x_en        : 1;
        uint16_t    block_dim_y_en        : 1;
        uint16_t    block_dim_z_en        : 1;
        uint16_t    block_idx_x_en      : 1;
        uint16_t    block_idx_y_en      : 1;
        uint16_t    block_idx_z_en      : 1;
        uint16_t    start_thread_idx_en   : 1;
        uint16_t    user_sreg_num       : 6;
    } bits;
    uint16_t    val;
};
#endif

union SHADER_ABI_KERNEL_MODE {
    struct {
        uint32_t    fp_rndmode          : 3;
        uint32_t    i_rndmode           : 2;
        uint32_t    fp_denorm_flush     : 2;
        uint32_t    saturation          : 3;
        uint32_t    exception_en        : 8;
        uint32_t    relu                : 1;
        uint32_t    nan                 : 1;
        uint32_t    vmem_ooo            : 1;
        uint32_t    saturation_fp64     : 1;
        uint32_t    rsvd_23_22          : 1;
        uint32_t    trap_exception      : 1;
        uint32_t    debug_en            : 1;
        uint32_t    trap_en             : 1;
        uint32_t    rsvd_32_27          : 5;
    } bits;
    uint32_t    val;
};

union SHADER_ABI_KERNEL_RESOURCE {
    struct {
        uint32_t    vreg_number         : 9;
        uint32_t    sreg_number         : 9;
        uint32_t    shared_memory_size  : 12;
        uint32_t    treg_en             : 1;
        uint32_t    rsvd_31             : 1;
    } bits;
    uint32_t        val;
};

union SHADER_ABI_THREADBLOCK_DIM {
    struct {
        uint32_t    x   : 12;
        uint32_t    y   : 12;
        uint32_t    z   : 8;
    } bits;
    uint32_t    val;
};

struct DispatchInfo {
    uint64_t    kernel_prog_addr;
    uint64_t    kernel_name_addr;
    uint64_t    kernel_param_addr;
    uint64_t    kernel_param_size;
    uint64_t    private_mem_addr;
    uint64_t    start_pC;
    uint32_t    grid_dim_x;
    uint32_t    grid_dim_y;
    uint32_t    grid_dim_z;
    uint32_t    block_idx;
    uint32_t    block_idy;
    uint32_t    block_idz;
    uint16_t    block_dim_x;
    uint16_t    block_dim_y;
    uint16_t    block_dim_z;
    // SHADER_ABI_KERNEL_CONTROL kernel_ctrl;
    uint32_t    kernel_ctrl;
    SHADER_ABI_KERNEL_MODE    kernel_mode;
    SHADER_ABI_KERNEL_RESOURCE  kernel_resource;
    // SHADER_ABI_THREADBLOCK_DIM block_dim;
    uint32_t    userSreg[SHADER_ABI_USER_DATA_REGISTER_NUM_MAX];
    uint32_t    shared_memsize;
    uint32_t    private_memsize;
    uint32_t    bar_used;
    int cmem;
    int gmem;
    int vreg_used;
    int sreg_used;
};


