#include "elfio/elfio.hpp"
#include "utils/thread/monitor.hpp"

class IAgent;
namespace loader {
    class Executable;
}

namespace impl {

struct Symbol {
    std::string name;
    ELFIO::Elf64_Addr value = 0;
    ELFIO::Elf_Xword size = 0;
    ELFIO::Elf_Half sect_idx = 0;
    std::uint8_t bind = 0;
    std::uint8_t type = 0;
    std::uint8_t other = 0;
};


inline
Symbol read_symbol(const ELFIO::symbol_section_accessor& section,
                   unsigned int idx);

template<typename P>
inline
ELFIO::section* find_section_if(ELFIO::elfio& reader, P p) {
    const auto it = std::find_if(
        reader.sections.begin(), reader.sections.end(), std::move(p));

    return it != reader.sections.end() ? *it : nullptr;
}

}

void associate_code_object_symbols_with_host_allocation(
    const ELFIO::elfio& reader,
    ELFIO::section* code_object_dynsym,
    // hsa_agent_t agent,
    IAgent* agent,
    loader::Executable* executable);

class ProgramState {
  utils::Monitor lock_{"Guards ProgramState globals", true};

  /* Singleton object */
  static ProgramState* platform_;
  ProgramState() {}
  ~ProgramState() {}

public:
  void init();
#if 0
  //Dynamic Code Objects functions
  hipError_t loadModule(hipModule_t* module, const char* fname, const void* image = nullptr);
  hipError_t unloadModule(hipModule_t hmod);

  hipError_t getDynFunc(hipFunction_t *hfunc, hipModule_t hmod, const char* func_name);
  hipError_t getDynGlobalVar(const char* hostVar, hipModule_t hmod, hipDeviceptr_t* dev_ptr,
                             size_t* size_ptr);
  hipError_t getDynTexRef(const char* hostVar, hipModule_t hmod, textureReference** texRef);

  hipError_t registerTexRef(textureReference* texRef, hipModule_t hmod, std::string name);
  hipError_t getDynTexGlobalVar(textureReference* texRef, hipDeviceptr_t* dev_ptr,
                                size_t* size_ptr);
#endif
  /* Singleton instance */
  static ProgramState& instance() {
    if (platform_ == nullptr) {
       // __hipRegisterFatBinary() will call this when app starts, thus
       // there is no multiple entry issue here.
       platform_ =  new ProgramState();
    }
    return *platform_;
  }
#if 0
  //Static Code Objects functions
  hip::FatBinaryInfo** addFatBinary(const void* data);
  hipError_t removeFatBinary(hip::FatBinaryInfo** module);
  hipError_t digestFatBinary(const void* data, hip::FatBinaryInfo*& programs);

  hipError_t registerStatFunction(const void* hostFunction, hip::Function* func);
  hipError_t registerStatGlobalVar(const void* hostVar, hip::Var* var);
  hipError_t registerStatManagedVar(hip::Var* var);


  hipError_t getStatFunc(hipFunction_t* hfunc, const void* hostFunction, int deviceId);
  hipError_t getStatFuncAttr(hipFuncAttributes* func_attr, const void* hostFunction, int deviceId);
  hipError_t getStatGlobalVar(const void* hostVar, int deviceId, hipDeviceptr_t* dev_ptr,
                              size_t* size_ptr);

  hipError_t initStatManagedVarDevicePtr(int deviceId);

  //Exec Functions
  void setupArgument(const void *arg, size_t size, size_t offset);
  void configureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream);
  void popExec(ihipExec_t& exec);

private:
  //Dynamic Code Object map, keyin module to get the corresponding object
  std::unordered_map<hipModule_t, hip::DynCO*> dynCO_map_;
  hip::StatCO statCO_; //Static Code object var
  std::unordered_map<textureReference*, std::pair<hipModule_t, std::string>> texRef_map_;
#endif
  bool initialized_{false};
};
