#pragma once
#include "pscm/logger/Logger.h"
#define PSCM_LOGGER_LOG(logger_, level, format_, args...)                                                              \
  {                                                                                                                    \
    if (logger_->is_level_enabled(level)) {                                                                            \
      logger_->log(level, fmt::format(format_, ##args));                                                               \
    }                                                                                                                  \
  }

#define PSCM_FATAL(format_, args...) PSCM_LOGGER_LOG(logger_, pscm::logger::Logger::Level::FATAL, format_, ##args)
#define PSCM_ERROR(format_, args...) PSCM_LOGGER_LOG(logger_, pscm::logger::Logger::Level::ERROR, format_, ##args)
#define PSCM_WARN(format_, args...) PSCM_LOGGER_LOG(logger_, pscm::logger::Logger::Level::WARN, format_, ##args)
#define PSCM_INFO(format_, args...) PSCM_LOGGER_LOG(logger_, pscm::logger::Logger::Level::INFO, format_, ##args)
#define PSCM_DEBUG(format_, args...) PSCM_LOGGER_LOG(logger_, pscm::logger::Logger::Level::DEBUG_, format_, ##args)
#define PSCM_TRACE(format_, args...) PSCM_LOGGER_LOG(logger_, pscm::logger::Logger::Level::TRACE, format_, ##args)

#define PSCM_INLINE_LOG_DECLARE(name) static pscm::logger::Logger *logger_ = pscm::logger::Logger::get_logger(name)