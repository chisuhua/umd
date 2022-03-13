#pragma once
#include "../../libcuda/CUctx.h"
#include "platform/IContext.h"

namespace drv {
  class CUctx : public ::IContext {
    public:
      CUctx(::CUctx* ctx) {
        real_ = ctx;
        // this->setPlatformName("platlibcuda");
      }
      ::CUctx *get() { return real_; };
      ::CUctx *real_;
  };
}
