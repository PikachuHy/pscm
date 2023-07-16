//
// Created by PikachuHy on 2023/7/15.
//

#pragma once
#include "pscm/misc/SourceLocation.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pscm {
namespace logger {
class Appender;
struct Event;

class Logger {
public:
  // avoid DEBUG macro conflict
  enum class Level { NONE, FATAL, ERROR, WARN, INFO, DEBUG_, TRACE };
  static Logger *root_logger();
  static std::unordered_map<std::string, Logger *>& logger_map();
  static Logger *get_logger(std::string name);
  void log(Logger::Level level, std::string msg, SourceLocation loc = {});
  void add_appender(Appender *appender);

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

private:
  std::unordered_set<Appender *> appender_set_;
  std::string name_;
  Level level_;
  Logger *parent_ = nullptr;
  std::vector<Logger *> children_;
};

} // namespace logger
} // namespace pscm