#include "../Umd.h"
#include "../IPlatform.h"


class UmdCuda : public IPlatform {
public:
    UmdCuda() { }
    ~UmdCuda() = default;

    status_t memory_register(void* address, size_t size);
    status_t memory_deregister(void* address, size_t size);
    status_t memory_allocate(size_t size, void** ptr, IMemRegion *region = nullptr);
    status_t memory_free(void* ptr);
    IMemRegion* get_system_memregion();
    IMemRegion* get_device_memregion(IAgent* agent);
    status_t free_memregion(IMemRegion *region);
};
