#pragma once
#include "../../libcuda/CUctx.h"
#include "platform/IContext.h"

namespace drv {
  class Context : public ::IContext {
    public:
      Context(::CUctx* ctx) : IContext(ctx->get_umd_name())
      {
        real_ = ctx;
        // this->setPlatformName("platlibcuda");
      }
      ::CUctx *get() { return real_; };
      ::CUctx *real_;
  };
}
