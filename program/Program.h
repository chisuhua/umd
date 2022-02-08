#pragma once

#include <map>
#include <vector>
#include "loader/CodeObject.h"
#include "loader/LoaderContext.h"
#include "loader/Loader.h"
// #include "util/utils.h"
class CUctx;

class Program {
public:
    loader::Loader* loader() { return loader_; }

    loader::Executable* executable() { return exec_; }

    LoaderContext* loader_context() { return loader_context_; }

    code::CodeObjectManager* code_manager() { return &code_manager_; }

    Program(CUctx* ctx)
    {
        loader_context_ = new LoaderContext(ctx);
        loader_ = loader::Loader::Create(loader_context_);
        exec_ = loader_->CreateExecutable(HSA_PROFILE_BASE, "", HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT);
    };

    Program(const Program&);

    Program& operator=(const Program&);

    ~Program()
    {
        loader::Loader::Destroy(loader_);
        loader_ = nullptr;
    }

    /// @brief Load Platform Open connection to kernel driver.
    status_t Load();

    /// @brief Close Platform connection to kernel driver and cleanup resources.
    void Unload();

protected:
    // Loader instance.
    loader::Loader* loader_;

    loader::Executable* exec_;

    // Loader context.
    LoaderContext *loader_context_;

    // Code object manager.
    code::CodeObjectManager code_manager_;

    // Holds reference count to program object.
    std::atomic<uint32_t> ref_count_;
};

loader::Executable* LoadProgram(const std::string& file, CUctx* ctx, IAgent* agent = nullptr) ;
