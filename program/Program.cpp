#include "loader/CodeObject.h"
#include "loader/Loader.h"
#include "elfio/elfio.hpp"
#include "Program.h"


struct Symbol {
    std::string name;
    ELFIO::Elf64_Addr value = 0;
    ELFIO::Elf_Xword size = 0;
    ELFIO::Elf_Half sect_idx = 0;
    std::uint8_t bind = 0;
    std::uint8_t type = 0;
    std::uint8_t other = 0;
};

template<typename P>
inline
ELFIO::section* find_section_if(ELFIO::elfio& reader, P p) {
    const auto it = std::find_if(
        reader.sections.begin(), reader.sections.end(), std::move(p));

    return it != reader.sections.end() ? *it : nullptr;
}

inline
Symbol read_symbol(const ELFIO::symbol_section_accessor& section,
                   unsigned int idx) {
    assert(idx < section.get_symbols_num());

    Symbol r;
    section.get_symbol(
        idx, r.name, r.value, r.size, r.bind, r.type, r.sect_idx, r.other);

    return r;
}

inline
const std::unordered_map<
    std::string,
    std::pair<ELFIO::Elf64_Addr, ELFIO::Elf_Xword>>& symbol_addresses() {
    static std::unordered_map<
        std::string, std::pair<ELFIO::Elf64_Addr, ELFIO::Elf_Xword>> r;
    static std::once_flag f;

    std::call_once(f, []() {
        dl_iterate_phdr([](dl_phdr_info* info, std::size_t, void*) {
            ELFIO::elfio tmp;
            const auto elf =
                info->dlpi_addr ? info->dlpi_name : "/proc/self/exe";

            if (!tmp.load(elf)) return 0;

            auto it = find_section_if(tmp, [](const ELFIO::section* x) {
                return x->get_type() == SHT_SYMTAB;
            });

            if (!it) return 0;

            const ELFIO::symbol_section_accessor symtab{tmp, it};

            for (auto i = 0u; i != symtab.get_symbols_num(); ++i) {
                auto s = read_symbol(symtab, i);

                if (s.type != STT_OBJECT || s.sect_idx == SHN_UNDEF) continue;

                const auto addr = s.value + info->dlpi_addr;
                r.emplace(std::move(s.name), std::make_pair(addr, s.size));
            }

            return 0;
        }, nullptr);
    });

    return r;
}

inline
std::unordered_map<std::string, void*>& globals() {
    static std::unordered_map<std::string, void*> r;
    static std::once_flag f;

    std::call_once(f, []() { r.reserve(symbol_addresses().size()); });

    return r;
}


inline
std::vector<std::string> copy_names_of_undefined_symbols(
    const ELFIO::symbol_section_accessor& section) {
    std::vector<std::string> r;

    for (auto i = 0u; i != section.get_symbols_num(); ++i) {
        // TODO: this is boyscout code, caching the temporaries
        //       may be of worth.
        auto tmp = read_symbol(section, i);
        if (tmp.sect_idx != SHN_UNDEF || tmp.name.empty()) continue;

        r.push_back(std::move(tmp.name));
    }

    return r;
}

inline
void associate_code_object_symbols_with_host_allocation(
    const ELFIO::elfio& reader,
    ELFIO::section* code_object_dynsym,
    // hsa_agent_t agent,
    IAgent* agent,
    loader::Executable* executable) {
    if (!code_object_dynsym) return;

    const auto undefined_symbols = copy_names_of_undefined_symbols(
        ELFIO::symbol_section_accessor{reader, code_object_dynsym});

    for (auto&& x : undefined_symbols) {
        if (globals().find(x) != globals().cend()) return;

        const auto it1 = symbol_addresses().find(x);

        if (it1 == symbol_addresses().cend()) {
            throw(std::runtime_error{
                "Global symbol: " + x + " is undefined."});
        }

        static std::mutex mtx;
        std::lock_guard<std::mutex> lck{mtx};

        if (globals().find(x) != globals().cend()) return;

        globals().emplace(x, (void*)(it1->second.first));
        void* p = nullptr;
        /* FIXME
        hsa_amd_memory_lock(
            reinterpret_cast<void*>(it1->second.first),
            it1->second.second,
            nullptr,  // All agents.
            0,
            &p);
            */

        loader::Symbol* sym = executable->GetSymbol(x.c_str(), agent);
        p = reinterpret_cast<void*>(loader::Symbol::Handle(sym).handle);
        // hsa_executable_agent_global_variable_define(
        //   executable, agent, x.c_str(), p);
    }
}


void load_code_object_and_freeze_executable(
    const std::string& file, IAgent* agent, loader::Executable* executable) {
    // TODO: the following sequence is inefficient, should be refactored
    //       into a single load of the file and subsequent ELFIO
    //       processing.
    static const auto cor_deleter = [](CodeObjectReader* p) {
        if (!p) return;

        CodeObjectReader::Destroy(p);
        delete p;
    };

    using RAII_code_reader = std::unique_ptr<CodeObjectReader, decltype(cor_deleter)>;

    if (file.empty()) return;

    RAII_code_reader tmp{CodeObjectReader::CreateFromMemory(file.data(), file.size()),
        cor_deleter};


    tmp.get()->LoadExecutable(executable, agent, nullptr, nullptr);
    // hsa_executable_load_agent_code_object(
    //    executable, agent, *tmp, nullptr, nullptr);

    // hsa_executable_freeze(executable, nullptr);
    executable->Freeze(nullptr);

    static std::vector<RAII_code_reader> code_readers;
    static std::mutex mtx;

    std::lock_guard<std::mutex> lck{mtx};
    code_readers.push_back(move(tmp));
}

inline
loader::Executable* load_executable(const std::string& file,
                                 loader::Executable* executable,
                                 IAgent* agent) {
    ELFIO::elfio reader;
    std::stringstream tmp{file};

    if (!reader.load(tmp)) {
        assert("Fail on elf read");
    }

    const auto code_object_dynsym = find_section_if(
        reader, [](const ELFIO::section* x) {
            return x->get_type() == SHT_DYNSYM;
    });

    associate_code_object_symbols_with_host_allocation(reader,
                                                       code_object_dynsym,
                                                       agent, executable);

    load_code_object_and_freeze_executable(file, agent, executable);

    return executable;
}

loader::Executable* LoadProgram(const std::string& file, CUctx* ctx, IAgent* agent ) {
    static Program* program_loader = nullptr;
    if (program_loader == nullptr) {
       program_loader = new Program(ctx);
    }

    return load_executable(file, program_loader->executable(), agent);
}

