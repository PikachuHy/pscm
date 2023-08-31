//
// Created by PikachuHy on 2023/7/15.
//
#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm.logger;
import fmt;
#else
#include "pscm/logger/Appender.h"
#include <chrono>
#include <iostream>
#include <spdlog/fmt/chrono.h>
#include <spdlog/fmt/fmt.h>
#endif

namespace fmt{
template <>
class formatter<pscm::logger::Logger::Level> {
public:
  constexpr auto parse(format_parse_context& ctx) {
    // PSCM_THROW_EXCEPTION("not supported now");
    auto i = ctx.begin();
    return i;
  }

  auto format(const pscm::logger::Logger::Level level, format_context& ctx) const {
    switch (level) {
    case pscm::logger::Logger::Level::NONE:
      break;
    case pscm::logger::Logger::Level::FATAL:
      return fmt::format_to(ctx.out(), "{}{}{}{}", on_red, white, "FATAL", reset, reset);
    case pscm::logger::Logger::Level::ERROR_:
      return fmt::format_to(ctx.out(), "{}{}{}", red, "ERROR", reset);
    case pscm::logger::Logger::Level::WARN:
      return fmt::format_to(ctx.out(), "{}{}{}", yellow, "WARN", reset);
    case pscm::logger::Logger::Level::INFO:
      return fmt::format_to(ctx.out(), "{}{}{}", green, "INFO", reset);
    case pscm::logger::Logger::Level::DEBUG_:
      return fmt::format_to(ctx.out(), "{}{}{}", cyan, "DEBUG", reset);
    case pscm::logger::Logger::Level::TRACE:
      return fmt::format_to(ctx.out(), "{}", "TRACE");
    }
    return fmt::format_to(ctx.out(), "{}", "");
  }

  // Formatting codes
  const std::string reset = "\033[m";
  const std::string bold = "\033[1m";
  const std::string dark = "\033[2m";
  const std::string underline = "\033[4m";
  const std::string blink = "\033[5m";
  const std::string reverse = "\033[7m";
  const std::string concealed = "\033[8m";
  const std::string clear_line = "\033[K";

  // Foreground colors
  const std::string black = "\033[30m";
  const std::string red = "\033[31m";
  const std::string green = "\033[32m";
  const std::string yellow = "\033[33m";
  const std::string blue = "\033[34m";
  const std::string magenta = "\033[35m";
  const std::string cyan = "\033[36m";
  const std::string white = "\033[37m";

  /// Background colors
  const std::string on_black = "\033[40m";
  const std::string on_red = "\033[41m";
  const std::string on_green = "\033[42m";
  const std::string on_yellow = "\033[43m";
  const std::string on_blue = "\033[44m";
  const std::string on_magenta = "\033[45m";
  const std::string on_cyan = "\033[46m";
  const std::string on_white = "\033[47m";

  /// Bold colors
  const std::string yellow_bold = "\033[33m\033[1m";
  const std::string red_bold = "\033[31m\033[1m";
  const std::string bold_on_red = "\033[1m\033[41m";
};

template <>
class formatter<pscm::UString> {
public:
  constexpr auto parse(format_parse_context& ctx) {
    // PSCM_THROW_EXCEPTION("not supported now");
    auto i = ctx.begin();
    return i;
  }

  auto format(const pscm::UString& str, format_context& ctx) const {
    std::string utf8;
    str.toUTF8String(utf8);
    return fmt::format_to(ctx.out(), "{}", utf8);
  }
};

}

namespace pscm {
namespace logger {

int ConsoleAppender::append(Event& event) {
  if (event.level == Logger::Level::NONE) {
    return 0;
  }
  auto now = std::chrono::system_clock::now();
  // {:%Y-%m-%d :%H:%M:%S}
  std::string log = fmt::format("[{}] [{}] [{}] {}", now, event.level, event.loc.to_string(), event.msg);
  std::cout << log << std::endl;
  return 0;
}
} // namespace logger
} // namespace pscm