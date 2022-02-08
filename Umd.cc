#include "Umd.h"
#include "program/Program.h"
#include <assert.h>

class IMemRegion;
class IAgent;

loader::Executable* Umd::load_program(const std::string& file, CUctx* ctx, IAgent* agent) {
    return LoadProgram(file, ctx, agent);
};

status_t Umd::memory_register(void* address, size_t size) {
    assert(false);
}

status_t Umd::memory_deregister(void* address, size_t size) {
    assert(false);
}

status_t Umd::memory_allocate(size_t size, void** ptr, IMemRegion *region) {
    assert(false);
}

status_t Umd::memory_free(void* ptr) {
    assert(false);
}

IMemRegion* Umd::get_system_memregion() {
    assert(false);
}

IMemRegion* Umd::get_device_memregion(IAgent* agent) {
    assert(false);
}

status_t Umd::free_memregion(IMemRegion *region) {
    assert(false);
}
