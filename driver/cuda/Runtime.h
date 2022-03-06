#include "runtime/IRuntime.h"

class rtCuda {
    virtual init();
    Device* getCurrentDevice() {
    void setCurrentDevice(unsigned int index) {
    rt::HostQueue* getQueue(hipStream_t stream) {
    int getDeviceID(rt::Context& ctx) {
    rt::HostQueue* getNullStream() {
}
