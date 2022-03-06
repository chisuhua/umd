#pragma once

#include "Device.hpp"
#include "top.hpp"
#include <vector>
#include <unordered_map>


namespace rt {

class Context : public RuntimeObject {
public:
  std::vector<Device*> devices_;
private:
  std::vector<Signal*> signal_pool_;
  std::vector<bool> signal_pool_flag_;
  unsigned int signal_cursor_;
  std::mutex signalPoolMutex;

  void releaseSignal(Signal* signal, int signalIndex) ;
  Signal* getSignal() ;

 public:

  //! Context info structure
  struct Info {
    uint flags_;                     //!< Context info flags
    void* hDev_[LastDeviceFlagIdx];  //!< Device object reference
    void* hCtx_;                     //!< Context object reference
    size_t propertiesSize_;          //!< Size of the original properties in bytes
  };

  struct DeviceQueueInfo {
    DeviceQueue* defDeviceQueue_;  //!< Default device queue
    uint deviceQueueCnt_;          //!< The number of device queues
    DeviceQueueInfo() : defDeviceQueue_(NULL), deviceQueueCnt_(0) {}
  };

 private:
  // Copying a Context is not allowed
  Context(const Context&);
  Context& operator=(const Context&);

 protected:
  //! Context destructor
  ~Context();

 public:
  //! Default constructor
  Context(const std::vector<Device*>& devices,  //!< List of all devices
          const Info& info                      //!< Context info structure
          );
  //! Compare two Context instances.
  bool operator==(const Context& rhs) const { return this == &rhs; }
  bool operator!=(const Context& rhs) const { return !(*this == rhs); }
  /*! Creates the context
   *
   *  \return An errcode if runtime fails the context creation,
   *          CL_SUCCESS otherwise
   */
  int create(const intptr_t* properties  //!< Original context properties
             );
  /**
   * Allocate host memory using either a custom device allocator or a generic
   * OS allocator
   */
  void* hostAlloc(size_t size, size_t alignment, bool atomics = false) const;
   /**
   * Release host memory
   * @param ptr Pointer allocated using ::hostAlloc. If the pointer has been
   * allocated elsewhere, the behavior is undefined
   */
  void hostFree(void* ptr) const;

  /**
   * Allocate SVM buffer
   *
   */
  void* svmAlloc(size_t size, size_t alignment, cl_svm_mem_flags flags = CL_MEM_READ_WRITE,
                 const amd::Device* curDev = nullptr);

  /**
   * Release SVM buffer
   */
  void svmFree(void* ptr) const;

  //! Return the devices associated with this context.
  const std::vector<Device*>& devices() const { return devices_; }

  //! Return the SVM capable devices associated with this context.
  const std::vector<Device*>& svmDevices() const { return svmAllocDevice_; }

  //! Returns true if the given device is associated with this context.
  bool containsDevice(const Device* device) const;

  //! Returns the context info structure
  const Info& info() const { return info_; }

  void setInfo(Info info) {
    info_ = info;
    return;
  }

  //! Returns a pointer to the OpenGL context
  GLFunctions* glenv() const { return glenv_; }

  //! Returns context lock for the serialized access to the context
  Monitor& lock() { return ctxLock_; }

  //! Returns TRUE if runtime succesfully added a device queue
  DeviceQueue* defDeviceQueue(const Device& dev) const;

  //! Returns TRUE if runtime succesfully added a device queue
  bool isDevQueuePossible(const Device& dev);

  //! Returns TRUE if runtime succesfully added a device queue
  void addDeviceQueue(const Device& dev,   //!< Device object
                      DeviceQueue* queue,  //!< Device queue
                      bool defDevQueue     //!< Added device queue will be the default queue
                      );

  //! Removes a device queue from the list of queues
  void removeDeviceQueue(const Device& dev,  //!< Device object
                         DeviceQueue* queue  //!< Device queue
                         );

  //! Set the default device queue
  void setDefDeviceQueue(const Device& dev, DeviceQueue* queue)
      { deviceQueues_[&dev].defDeviceQueue_ = queue; };

 private:
  Info info_;                            //!< Context info structure
  // cl_context_properties* properties_;    //!< Original properties
  GLFunctions* glenv_;                   //!< OpenGL context
  Device* customHostAllocDevice_;        //!< Device responsible for host allocations
  std::vector<Device*> svmAllocDevice_;  //!< Devices can support SVM allocations
  std::unordered_map<const Device*, DeviceQueueInfo> deviceQueues_;  //!< Device queues mapping
  mutable Monitor ctxLock_;                                          //!< Lock for the context access
};


}  // namespace rt



