//
// Created by PikachuHy on 2023/7/15.
//

#pragma once
#include "pscm/logger/Logger.hpp"
#include "pscm/misc/SourceLocation.h"
#include <string>

namespace pscm {
namespace logger {

class Appender {
public:
  virtual int append(Event& event) = 0;
};

class ConsoleAppender : public Appender {
public:
  int append(Event& event) override;
};

} // namespace logger
} // namespace pscm
