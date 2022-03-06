

namespace cuda {
thread_local Device* g_device = nullptr;
thread_local std::stack<Device*> g_ctxtStack;
Device* host_device = nullptr;

void init() {
  if (!rt::Runtime::initialized()) {
    rt::Runtime::init();
  }
  const std::vector<rt::Device*>& devices = rt::Device::getDevices(CL_DEVICE_TYPE_GPU, false);

  for (unsigned int i=0; i<devices.size(); i++) {
    const std::vector<rt::Device*> device(1, devices[i]);
    rt::Context* context = new rt::Context(device, rt::Context::Info());
    if (!context) return;

    // Enable active wait on the device by default
    devices[i]->SetActiveWait(true);

    if (context && CL_SUCCESS != context->create(nullptr)) {
      context->release();
    } else {
      g_devices.push_back(new Device(context, i));
    }
  }

  rt::Context* hContext = new rt::Context(devices, rt::Context::Info());

  if (!hContext) return;

  if (CL_SUCCESS != hContext->create(nullptr)) {
    hContext->release();
  }
  host_device = new Device(hContext, -1);

  ProgramState::instance().init();
  // PlatformState::instance().init();
}

Device* getCurrentDevice() {
  return g_device;
}

void setCurrentDevice(unsigned int index) {
  assert(index<g_devices.size());
  g_device = g_devices[index];
}

rt::HostQueue* getQueue(hipStream_t stream) {
 if (stream == nullptr) {
    return getNullStream();
  } else {
    constexpr bool WaitNullStreamOnly = true;
    rt::HostQueue* queue = reinterpret_cast<hip::Stream*>(stream)->asHostQueue();
    if (!(reinterpret_cast<hip::Stream*>(stream)->Flags() & hipStreamNonBlocking)) {
      iHipWaitActiveStreams(queue, WaitNullStreamOnly);
    }
    return queue;
  }
}

// ================================================================================================
rt::HostQueue* getNullStream(rt::Context& ctx) {
  for (auto& it : g_devices) {
    if (it->asContext() == &ctx) {
      return it->NullStream();
    }
  }
  // If it's a pure SVM allocation with system memory access, then it shouldn't matter which device
  // runtime selects by default
  if (hip::host_device->asContext() == &ctx) {
    // Return current...
    return getNullStream();
  }
  return nullptr;
}

// ================================================================================================
int getDeviceID(rt::Context& ctx) {
  for (auto& it : g_devices) {
    if (it->asContext() == &ctx) {
      return it->deviceId();
    }
  }
  return -1;
}

// ================================================================================================
rt::HostQueue* getNullStream() {
  Device* device = getCurrentDevice();
  return device ? device->NullStream() : nullptr;
}

};





}
