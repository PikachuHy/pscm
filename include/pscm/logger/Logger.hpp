//
// Created by PikachuHy on 2023/7/15.
//

#pragma once
#include "pscm/misc/SourceLocation.h"
#include "pscm/Displayable.h"
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pscm {
namespace logger {

void _setup_formattable(UFormattable& res, const UString& txt);
void _setup_formattable(UFormattable& res, std::int64_t num);
void _setup_formattable(UFormattable& res, std::int32_t num);
void _setup_formattable(UFormattable& res, const void* txt);
template<Displayable ObjT>
void _setup_formattable(UFormattable& res, ObjT obj){
  res.setString(pscm::to_string(obj));
};

class Appender;
struct Event;
class Logger;

template<typename Obj>
concept Formattable = requires (UFormattable& res, Obj obj) {
  pscm::logger::_setup_formattable(res, obj);
};
class Logger {
public:
  // avoid DEBUG macro and ERROR macro conflict
  enum class Level { NONE, FATAL, ERROR_, WARN, INFO, DEBUG_, TRACE };
  static Logger *root_logger();
  static std::unordered_map<std::string, Logger *>& logger_map();
  static Logger *get_logger(std::string name);
  void log(Logger::Level level, UString msg, SourceLocation loc = {});
  void add_appender(Appender *appender);
  template<typename... MsgT>
  void log(Logger::Level level, const UString format_, SourceLocation loc, MsgT... msg)
  {
  if (is_level_enabled(level)) {
    Event event;
    event.level = level;
    event.loc = loc;

    constexpr std::size_t arity = sizeof...(msg);
    UErrorCode status = U_ZERO_ERROR;
    UFormattable msgf[arity];
    _to_formattable(msgf, msg...);
    UFormatter::format(format_, msgf, arity, event.msg, status);
    assert(!U_FAILURE(status)); // 这里可能返回默认环境（-127）
    this->_log(event);
  }
}

  Level level() const {
    return level_;
  }

  void set_level(Level level) {
    this->level_ = level;
  }

  const std::string& name() const {
    return name_;
  }

  const Logger *parent() const {
    return parent_;
  }

  const std::vector<Logger *>& children() const {
    return children_;
  }

  const std::unordered_set<Appender *>& appender_set() const {
    return appender_set_;
  }

  [[nodiscard]] bool is_level_enabled(Level level) const;

private:
  Logger(std::string name, Level level = Level::INFO, Logger *parent = nullptr);
  void _log(Event& event);
  template<std::size_t size, Formattable Obj, typename... MsgT>
  void _to_formattable(UFormattable (&res)[size], Obj obj,  MsgT... rest){
    constexpr std::size_t index = size - 1 - sizeof...(rest);
    _setup_formattable(res[index], obj);
    _to_formattable(res, rest...);
  };
  template<std::size_t size, Formattable Obj>
  void _to_formattable(UFormattable (&res)[size], Obj obj){
    constexpr std::size_t index = size - 1;
    _setup_formattable(res[index], obj);
  };
  

private:
  std::unordered_set<Appender *> appender_set_;
  std::string name_;
  Level level_;
  Logger *parent_ = nullptr;
  std::vector<Logger *> children_;
};

struct Event {
  Logger::Level level;
  UString msg;
  SourceLocation loc;
};
} // namespace logger
} // namespace pscm