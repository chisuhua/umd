#pragma once
#include <memory>
#include <string>

class IAgent;

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

class IContext {
public:
  static std::string platformName(IContext *ctx) {
    if (ctx == nullptr) {
      return platform_name();
    } else {
      return ctx->getPlatformName();
    }
  }

  std::string getPlatformName() {
    if (name_ == "") {
      name_ = platform_name();
    }
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

  IAgent* m_agent;
  std::string name_ {""};
};
