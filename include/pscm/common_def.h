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

#define PSCM_CONCAT2(a, b) PSCM_CONCAT2_INNER(a, b)
#define PSCM_CONCAT2_INNER(a, b) a##b

#define PSCM_DEFINE_BUILTIN_PROC_INNER(func_name, api_name)                                                            \
  func_name(Cell args);                                                                                                \
  static ApiManager PSCM_CONCAT2(__api_manager_, func_name)(func_name, api_name);                                      \
  Cell func_name(Cell args)

#define PSCM_DEFINE_BUILTIN_PROC(module, name)                                                                         \
  Cell PSCM_DEFINE_BUILTIN_PROC_INNER(PSCM_CONCAT2(_##module##__proc__, __COUNTER__), name)
