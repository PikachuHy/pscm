#pragma once
#include "pscm/logger/Logger.h"
#define PSCM_LOGGER_LOG(logger_, level, format_, ...) (logger_)->log(level, format_, {}, ##__VA_ARGS__);

#define PSCM_FATAL(format_, ...) PSCM_LOGGER_LOG(logger_, pscm::logger::Level::FATAL, format_, ##__VA_ARGS__)
#define PSCM_ERROR(format_, ...) PSCM_LOGGER_LOG(logger_, pscm::logger::Level::ERROR_, format_, ##__VA_ARGS__)
#define PSCM_WARN(format_, ...) PSCM_LOGGER_LOG(logger_, pscm::logger::Level::WARN, format_, ##__VA_ARGS__)
#define PSCM_INFO(format_, ...) PSCM_LOGGER_LOG(logger_, pscm::logger::Level::INFO, format_, ##__VA_ARGS__)
#define PSCM_DEBUG(format_, ...) PSCM_LOGGER_LOG(logger_, pscm::logger::Level::DEBUG_, format_, ##__VA_ARGS__)
#define PSCM_TRACE(format_, ...) PSCM_LOGGER_LOG(logger_, pscm::logger::Level::TRACE, format_, ##__VA_ARGS__)

#define PSCM_INLINE_LOG_DECLARE(name) static pscm::logger::Logger *logger_ = pscm::logger::Logger::get_logger(name)
