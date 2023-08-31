module;
#include "pscm/logger/Appender.h"
#include "pscm/logger/Logger.hpp"
export module pscm.logger;

export namespace pscm::logger {
using pscm::logger::Appender;
using pscm::logger::ConsoleAppender;
using pscm::logger::Event;
using pscm::logger::Logger;
} // namespace pscm::logger
