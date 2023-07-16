//
// Created by PikachuHy on 2023/7/16.
//

#pragma once
#include "pscm/logger/Logger.h"
#include "pscm/misc/SourceLocation.h"

namespace pscm {
namespace logger {

struct Event {
  Logger::Level level;
  std::string msg;
  SourceLocation loc;
};

} // namespace logger
} // namespace pscm
