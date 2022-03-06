
namespace amd {
//! MemoryObject map lookup  class
class MemObjMap : public AllStatic {
 public:
  size_t size();  //!< obtain the size of the container
  void AddMemObj(const void* k,
                        amd::Memory* v);  //!< add the host mem pointer and buffer in the container
  void RemoveMemObj(const void* k);  //!< Remove an entry of mem object from the container
  amd::Memory* FindMemObj(
      const void* k);  //!< find the mem object based on the input pointer
  void UpdateAccess(amd::Device *peerDev);
  void Purge(amd::Device* dev); //!< Purge all user allocated memories on the given device
 private:
  std::map<uintptr_t, amd::Memory*>
      MemObjMap_;                      //!< the mem object<->hostptr information container
  amd::Monitor AllocatedLock_;  //!< amd monitor locker
};
}
