//
// Created by PikachuHy on 2023/7/15.
//

#include "pscm/logger/Logger.hpp"
#include "pscm/logger/Appender.h"
#include "unicode/msgfmt.h"
#include <cassert>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace pscm {
namespace logger {

Logger::Logger(std::string name, pscm::logger::Logger::Level level, pscm::logger::Logger *parent)
    : name_(std::move(name))
    , level_(level)
    , parent_(parent) {
}

Logger *Logger::root_logger() {
  static Logger logger("", Level::INFO);
  return &logger;
}

std::unordered_map<std::string, Logger *>& Logger::logger_map() {
  static std::unordered_map<std::string, Logger *> map;
  return map;
}

Logger *Logger::get_logger(std::string name) {
  Logger *ret = nullptr;
  auto it = logger_map().find(name);
  if (it != logger_map().end()) {
    ret = it->second;
  }
  if (ret != nullptr) {
    return ret;
  }
  std::string parent_name;
  auto pos = name.find_last_of('.');
  Logger *parent;
  if (pos == std::string::npos) {
    parent = root_logger();
  }
  else {
    parent_name = name.substr(0, pos);
    parent = get_logger(parent_name);
  }
  ret = new Logger(name, parent->level(), parent);
  parent->children_.push_back(ret);
  logger_map()[name] = ret;
  return ret;
}

void Logger::log(Logger::Level level, UString msg, SourceLocation loc) {
  Event event;
  event.level = level;
  event.msg = std::move(msg);
  event.loc = loc;
  this->_log(event);
}

void Logger::add_appender(Appender *appender) {
  appender_set_.insert(appender);
}

void Logger::_log(Event& event) {
  if (parent_) {
    parent_->_log(event);
  }

  for (auto appender : appender_set_) {
    appender->append(event);
  }
}

bool Logger::is_level_enabled(Level level) const {
  if (level == Level::NONE) {
    return false;
  }
  return level <= level_;
}

void _setup_formattable(UFormattable& res, const UString& txt){
  res.setString(txt);
};
void _setup_formattable(UFormattable& res, const void* ptr){
  res.setString(pscm::to_string(ptr));
};
void _setup_formattable(UFormattable& res, std::int64_t num){
  res.setInt64(num);
};
void _setup_formattable(UFormattable& res, std::int32_t num){
  res.setLong(num);
};
} // namespace logger
} // namespace pscm