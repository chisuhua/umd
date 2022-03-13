#include "loader/CodeObject.h"
#include "loader/CodeObjectReader.h"
#include "loader/Loader.h"
#include "elfio/elfio.hpp"
#include "Program.h"
#include "utils/debug.hpp"

using namespace ELFIO;

void load_code_object_and_freeze_executable(
    std::ifstream& istream, IAgent* agent, loader::Executable* executable) {
    // TODO: the following sequence is inefficient, should be refactored
    //       into a single load of the file and subsequent ELFIO
    //       processing.
    static const auto cor_deleter = [](CodeObjectReader* p) {
        if (!p) return;

        CodeObjectReader::Destroy(p);
        delete p;
    };

    using RAII_code_reader = std::unique_ptr<CodeObjectReader, decltype(cor_deleter)>;

    istream.seekg(0, std::ios::end);
    size_t file_size = istream.tellg();
    char *file_data = new char[file_size];
    istream.seekg(0, std::ios::beg);
    istream.read(file_data, file_size);

    RAII_code_reader tmp{CodeObjectReader::CreateFromMemory(file_data, file_size),
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
    std::ifstream stream;
    stream.open(file.c_str(), std::ios::in | std::ios::binary);
    if (!stream) {
        fatal("can't open elf file");
    }

    ELFIO::elfio reader;
    // std::stringstream tmp{file};

    if (!reader.load(stream)) {
        fatal("Fail on elf read");
    }

    const auto code_object_dynsym = impl::find_section_if(
        reader, [](const ELFIO::section* x) {
            return x->get_type() == SHT_DYNSYM;
    });

    associate_code_object_symbols_with_host_allocation(reader,
                                                       code_object_dynsym,
                                                       agent, executable);

    load_code_object_and_freeze_executable(stream, agent, executable);

    return executable;
}

loader::Executable* LoadProgram(const std::string& file, IContext* ctx, IAgent* agent ) {
    static Program* program_loader = nullptr;
    if (program_loader == nullptr) {
       program_loader = new Program(ctx);
    }

    return load_executable(file, program_loader->executable(), agent);
}

