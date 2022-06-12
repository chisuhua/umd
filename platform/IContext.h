#pragma once
#include <memory>
#include <string>

class IAgent;

#if 0
static std::string platform_name () {
    std::string name;
    char *buff = getenv("UMD");
    if (buff) {
        name = buff;
    } else {
        name = "platlibcuda";
    }
    return name;
};
#endif

class IContext {
public:
  IContext(std::string umd_platname) : name_(umd_platname)
  {}

  static std::string platformName(IContext *ctx) {
    assert(ctx != nullptr);
    return ctx->getPlatformName();
/*
    if (ctx == nullptr) {
      return platform_name();
    } else {
      return ctx->getPlatformName();
    }
*/
  }

  std::string getPlatformName() {
#if 0
    if (name_ == "") {
      if (umd_mode <= 1) {
          name_ = "platlibcuda";
      } else if (umd_mode == 2) {
          name_ = "platlibgem5cuda";
      } else if (umd_mode == 3) {
          name_ = "platlibgem5umd";
      } else if (umd_mode == 4) {
          name_ = "platlibgem5kmd";
      } else {
         assert (false || "umd > 1");
      }
      /*
      name_ = platform_name();
      */
    }
#endif
    return name_;
  }

  void setPlatformName(std::string name) {
      name_ = name;
  }

  IContext() {
  }

  virtual ~IContext() {};

  IAgent* get_agent() {
    return m_agent;
  }

  void set_agent(IAgent* agent) {
    m_agent = agent;
  }
public:
  IAgent* m_agent;
  std::string name_ {""};
};
