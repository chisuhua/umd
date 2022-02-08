#include "loader/CodeObject.h"
#include "loader/Loader.h"
#include "loader_api.h"
#include <memory.h>

status_t hsa_executable_load_program_code_object(
    hsa_executable_t executable,
    hsa_code_object_reader_t code_object_reader,
    const char* options,
    hsa_loaded_code_object_t* loaded_code_object)
{

    Executable* exec = Executable::Object(executable);
    if (!exec) {
        return ERROR_INVALID_EXECUTABLE;
    }

    CodeObjectReader* wrapper = CodeObjectReader::Object(
        code_object_reader);
    if (!wrapper) {
        return ERROR_INVALID_CODE_OBJECT_READER;
    }

    hsa_code_object_t code_object = { reinterpret_cast<uint64_t>(wrapper->code_object_memory) };
    return exec->LoadCodeObject(
        { 0 }, code_object, options, loaded_code_object);
    CATCH;
}

