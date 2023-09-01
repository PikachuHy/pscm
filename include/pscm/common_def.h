//
// Created by PikachuHy on 2023/3/4.
//

#pragma once
#include "pscm/Exception.h"
#include "pscm/Logger.h"
#include <cassert>
#define PSCM_THROW_EXCEPTION(msg)                                                                                      \
  PSCM_ERROR("Exception occurred here: {0}", (msg));                                                                      \
  throw ::pscm::Exception(msg)
#define PSCM_ASSERT(e)                                                                                                 \
  if (!(e)) {                                                                                                          \
    PSCM_ERROR("ASSERT FAILED here: {0}", #e);                                                                          \
    assert(e);                                                                                                         \
  }                                                                                                                    \
  else {                                                                                                               \
    (void)0;                                                                                                           \
  }                                                                                                                    \
  (void *)0

#define PSCM_ASSERT_WITH_LOC(e, loc)                                                                                   \
  if (!(e)) {                                                                                                          \
    PSCM_ERROR("ASSERT FAILED here: {0}, call from {1}", #e, loc.to_string());                                           \
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

#define PSCM_DEFINE_BUILTIN_MACRO_INNER(func_name, api_name, label)                                                    \
  func_name(SchemeProxy scm, SymbolTable *env, Cell args);                                                             \
  static ApiManager PSCM_CONCAT2(__api_manager_, func_name)(func_name, api_name, label);                               \
  Cell func_name(SchemeProxy scm, SymbolTable *env, Cell args)

#define PSCM_DEFINE_BUILTIN_MACRO(module, name, label)                                                                 \
  Cell PSCM_DEFINE_BUILTIN_MACRO_INNER(PSCM_CONCAT2(_##module##__macro__, __COUNTER__), name, label)

#define PSCM_DEFINE_BUILTIN_MACRO_PROC_WRAPPER_INNER(func_name, api_name, label, proc_args)                            \
  func_name(SchemeProxy scm, SymbolTable *env, Cell args);                                                             \
  static ApiManager PSCM_CONCAT2(__api_manager_, func_name)(func_name, api_name, label, proc_args);                    \
  Cell func_name(SchemeProxy scm, SymbolTable *env, Cell args)

#define PSCM_DEFINE_BUILTIN_MACRO_PROC_WRAPPER(module, name, label, args)                                              \
  Cell PSCM_DEFINE_BUILTIN_MACRO_PROC_WRAPPER_INNER(PSCM_CONCAT2(_##module##__macro__, __COUNTER__), name, label, UString(args))

#ifdef PSCM_USE_CXX20_MODULES
#define PSCM_CXX20_MODULES_DEFAULT_ARG_COMPAT = {}
#else
#define PSCM_CXX20_MODULES_DEFAULT_ARG_COMPAT
#endif