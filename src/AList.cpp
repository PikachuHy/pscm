#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/ApiManager.h"
#include "pscm/Cell.h"
#include "pscm/Exception.h"
#include "pscm/Pair.h"
#include "pscm/SchemeProxy.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <spdlog/fmt/fmt.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#endif
using namespace std::string_literals;

namespace pscm {
PSCM_INLINE_LOG_DECLARE("pscm.core.AList");

AList::AList() {
}

PSCM_DEFINE_BUILTIN_PROC(AList, "acons") {
  auto key = car(args);
  auto value = cadr(args);
  auto alist = caddr(args);
  auto entry = cons(key, value);
  return cons(entry, alist);
}

PSCM_DEFINE_BUILTIN_PROC(AList, "assoc-set!") {
  auto alist = car(args);
  auto key = cadr(args);
  auto value = caddr(args);
  auto it = alist;
  while (it.is_pair()) {
    auto entry = car(it);
    PSCM_ASSERT(entry.is_pair());
    if (car(entry) == key) {
      entry.to_pair()->second = value;
      return alist;
    }
    it = cdr(it);
  }
  auto entry = cons(key, value);
  return cons(entry, alist);
}

PSCM_DEFINE_BUILTIN_PROC(AList, "assq-set!") {
  auto alist = car(args);
  auto key = cadr(args);
  auto value = caddr(args);
  auto it = alist;
  while (it.is_pair()) {
    auto entry = car(it);
    PSCM_ASSERT(entry.is_pair());
    if (car(entry).is_eq(key).to_bool()) {
      entry.to_pair()->second = value;
      return alist;
    }
    it = cdr(it);
  }
  auto entry = cons(key, value);
  return cons(entry, alist);
}

PSCM_DEFINE_BUILTIN_PROC(AList, "assv-set!") {
  auto alist = car(args);
  auto key = cadr(args);
  auto value = caddr(args);
  auto it = alist;
  while (it.is_pair()) {
    auto entry = car(it);
    PSCM_ASSERT(entry.is_pair());
    if (car(entry).is_eqv(key).to_bool()) {
      entry.to_pair()->second = value;
      return alist;
    }
    it = cdr(it);
  }
  auto entry = cons(key, value);
  return cons(entry, alist);
}

PSCM_DEFINE_BUILTIN_PROC(AList, "assoc-remove!") {
  auto alist = car(args);
  auto key = cadr(args);
  auto it = alist;
  auto last = cons(nil, alist);
  auto ret = last;
  while (it.is_pair()) {
    auto entry = car(it);
    PSCM_ASSERT(entry.is_pair());
    if (car(entry) == key) {
      last->second = cdr(it);
      return ret->second;
    }
    last = it.to_pair();
    it = cdr(it);
  }
  return ret->second;
}

PSCM_DEFINE_BUILTIN_PROC(AList, "assq-remove!") {
  auto alist = car(args);
  auto key = cadr(args);
  auto it = alist;
  auto last = cons(nil, alist);
  auto ret = last;
  while (it.is_pair()) {
    auto entry = car(it);
    PSCM_ASSERT(entry.is_pair());
    if (car(entry).is_eq(key).to_bool()) {
      last->second = cdr(it);
      return ret->second;
    }
    last = it.to_pair();
    it = cdr(it);
  }
  return ret->second;
}

PSCM_DEFINE_BUILTIN_PROC(AList, "assv-remove!") {
  auto alist = car(args);
  auto key = cadr(args);
  auto it = alist;
  auto last = cons(nil, alist);
  auto ret = last;
  while (it.is_pair()) {
    auto entry = car(it);
    PSCM_ASSERT(entry.is_pair());
    if (car(entry).is_eqv(key).to_bool()) {
      last->second = cdr(it);
      return ret->second;
    }
    last = it.to_pair();
    it = cdr(it);
  }
  return ret->second;
}

PSCM_DEFINE_BUILTIN_PROC(AList, "assq") {
  auto obj = car(args);
  auto list = cadr(args);
  while (!list.is_nil()) {
    if (obj.is_eq(caar(list)).to_bool()) {
      return car(list);
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

PSCM_DEFINE_BUILTIN_PROC(AList, "assv") {
  auto obj = car(args);
  auto list = cadr(args);
  while (!list.is_nil()) {
    if (obj.is_eqv(caar(list)).to_bool()) {
      return car(list);
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

PSCM_DEFINE_BUILTIN_PROC(AList, "assoc") {
  auto obj = car(args);
  auto list = cadr(args);
  while (!list.is_nil()) {
    if (obj == caar(list)) {
      return car(list);
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

PSCM_DEFINE_BUILTIN_PROC(AList, "assq-ref") {
  auto list = car(args);
  auto obj = cadr(args);
  while (!list.is_nil()) {
    if (obj.is_eq(caar(list)).to_bool()) {
      return cdar(list);
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

PSCM_DEFINE_BUILTIN_PROC(AList, "assv-ref") {
  auto list = car(args);
  auto obj = cadr(args);
  while (!list.is_nil()) {
    if (obj.is_eqv(caar(list)).to_bool()) {
      return cdar(list);
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

PSCM_DEFINE_BUILTIN_PROC(AList, "assoc-ref") {
  auto list = car(args);
  auto obj = cadr(args);
  while (!list.is_nil()) {
    if (obj == caar(list)) {
      return cdar(list);
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

} // namespace pscm