#include "elfio/elfio.hpp"

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

