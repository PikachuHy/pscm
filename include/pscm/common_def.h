//
// Created by PikachuHy on 2023/3/4.
//

#pragma once
#include "pscm/Exception.h"
#include <cassert>
#include <spdlog/spdlog.h>
#define PSCM_THROW_EXCEPTION(msg)                                                                                      \
  SPDLOG_ERROR("Exception occurred here: {}", msg);                                                                    \
  throw ::pscm::Exception(msg)
#define PSCM_ASSERT(e)                                                                                                 \
  if (!(e)) {                                                                                                          \
    SPDLOG_ERROR("ASSERT FAILED here: {}", #e);                                                                        \
    assert(e);                                                                                                         \
  }                                                                                                                    \
  else {                                                                                                               \
    (void)0;                                                                                                           \
  }                                                                                                                    \
  (void *)0

#define PSCM_ASSERT_WITH_LOC(e, loc)                                                                                   \
  if (!(e)) {                                                                                                          \
    SPDLOG_ERROR("ASSERT FAILED here: {}, call from {}", #e, loc.to_string());                                         \
    assert(e);                                                                                                         \
  }                                                                                                                    \
  else {                                                                                                               \
    (void)0;                                                                                                           \
  }                                                                                                                    \
  (void *)0
