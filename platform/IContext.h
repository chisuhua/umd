#pragma once
#include <memory>
#include <string>

class IAgent;
class IContext {
public:
  std::string& getPlatformName() {
      if (name_ == "") {
        char *buff = getenv("UMD");
        if (buff) {
            name_ = buff;
        } else {
            name_ = "platlibcuda";
        }
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
