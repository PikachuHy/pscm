module;
#include "pscm/logger/Appender.h"
#include "pscm/logger/Logger.h"
export module pscm.logger;
import pscm.icu;

export namespace pscm::logger {
using pscm::logger::_setup_formattable;
using pscm::logger::Appender;
using pscm::logger::ConsoleAppender;
using pscm::logger::Event;
using pscm::logger::Logger;
} // namespace pscm::logger
