//
// Created by PikachuHy on 2023/2/25.
//
#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/Pair.h"
#include "pscm/ApiManager.h"
#include "pscm/Cell.h"
#include "pscm/Exception.h"
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

PSCM_INLINE_LOG_DECLARE("pscm.core.Parser");

Cell car(Cell c, SourceLocation loc) {
  if (!c.is_pair()) {
    PSCM_THROW_EXCEPTION(loc.to_string() + ", Cell is not Pair: " + c.to_string());
  }
  auto p = c.to_pair();
  return p->first;
}

Cell caar(Cell c, SourceLocation loc) {
  return car(car(c, loc), loc);
}

Cell caaar(Cell c, SourceLocation loc) {
  return car(car(car(c, loc), loc), loc);
}

Cell cdr(Cell c, SourceLocation loc) {
  if (!c.is_pair()) {
    PSCM_THROW_EXCEPTION(loc.to_string() + ", Cell is not Pair: " + c.to_string());
  }
  auto p = c.to_pair();
  return p->second;
}

Cell cdar(Cell c, SourceLocation loc) {
  return cdr(car(c, loc), loc);
}

Cell cadr(Cell c, SourceLocation loc) {
  return car(cdr(c, loc), loc);
}

Cell cadar(Cell c, SourceLocation loc) {
  return car(cdr(car(c, loc), loc), loc);
}

Cell caadr(Cell c, SourceLocation loc) {
  return car(car(cdr(c, loc), loc), loc);
}

Cell cdadr(Cell c, SourceLocation loc) {
  return cdr(car(cdr(c, loc), loc), loc);
}

Cell cddr(Cell c, SourceLocation loc) {
  if (!c.is_pair()) {
    PSCM_THROW_EXCEPTION(loc.to_string() + ", Cell is not Pair: " + c.to_string());
  }
  auto p = c.to_pair();
  if (!p->second.is_pair()) {
    PSCM_THROW_EXCEPTION("Cell is not Pair: " + p->second.to_string() + " " + loc.to_string());
  }
  p = p->second.to_pair();
  return p->second;
}

Cell cdddr(Cell c, SourceLocation loc) {
  return cdr(cddr(c, loc), loc);
}

Cell caddr(Cell c, SourceLocation loc) {
  return car(cdr(cdr(c, loc), loc), loc);
}

Cell caddar(Cell c, SourceLocation loc) {
  return car(cdr(cdr(car(c, loc), loc), loc), loc);
}

Cell cadddr(Cell c, SourceLocation loc) {
  return car(cdr(cdr(cdr(c, loc), loc), loc), loc);
}

bool Pair::operator==(const Pair& rhs) const {
  return first == rhs.first && second == rhs.second;
}

bool Pair::operator!=(const Pair& rhs) const {
  return !(rhs == *this);
}

int list_length(Cell expr) {
  int len = 0;
  while (!expr.is_nil()) {
    len++;
    expr = cdr(expr);
  }
  return len;
}

PSCM_DEFINE_BUILTIN_PROC(List, "cons") {
  auto a = car(args);
  auto b = cadr(args);
  return cons(a, b);
}

PSCM_DEFINE_BUILTIN_PROC(List, "car") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return car(arg);
}

PSCM_DEFINE_BUILTIN_PROC(List, "caar") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return caar(arg);
}

PSCM_DEFINE_BUILTIN_PROC(List, "cdadr") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return cdadr(arg);
}

PSCM_DEFINE_BUILTIN_PROC(List, "cdr") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return cdr(arg);
}

PSCM_DEFINE_BUILTIN_PROC(List, "cdar") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return cdar(arg);
}

PSCM_DEFINE_BUILTIN_PROC(List, "cadr") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return cadr(arg);
}

PSCM_DEFINE_BUILTIN_PROC(List, "cadar") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return cadar(arg);
}

PSCM_DEFINE_BUILTIN_PROC(List, "cddr") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return cddr(arg);
}

PSCM_DEFINE_BUILTIN_PROC(List, "caddr") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return caddr(arg);
}

PSCM_DEFINE_BUILTIN_PROC(List, "caddar") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return caddar(arg);
}

PSCM_DEFINE_BUILTIN_PROC(List, "list") {
  PSCM_ASSERT(args.is_pair() || args.is_nil());
  return args;
}

PSCM_DEFINE_BUILTIN_PROC(List, "list?") {
  PSCM_ASSERT(args.is_pair());
  std::unordered_set<Pair *> p_set;
  auto arg = car(args);
  while (arg.is_pair()) {
    if (p_set.find(arg.to_pair()) != p_set.end()) {
      return Cell::bool_false();
    }
    p_set.insert(arg.to_pair());
    arg = cdr(arg);
  }
  return arg.is_nil() ? Cell::bool_true() : Cell::bool_false();
}

PSCM_DEFINE_BUILTIN_PROC(List, "pair?") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return Cell(arg.is_pair());
}

void set_car(Cell list, Cell val) {
  PSCM_ASSERT(list.is_pair());
  list.to_pair()->first = val;
}

void set_cdr(Cell list, Cell val) {
  PSCM_ASSERT(list.is_pair());
  list.to_pair()->second = val;
}

PSCM_DEFINE_BUILTIN_PROC(List, "set-car!") {
  PSCM_ASSERT(args.is_pair());
  auto pair = car(args);
  auto obj = cadr(args);
  if (!pair.is_pair()) {
    PSCM_THROW_EXCEPTION("Invalid set-car! args: " + args.to_string());
  }
  set_car(pair, obj);
  return Cell::none();
}

PSCM_DEFINE_BUILTIN_PROC(List, "set-cdr!") {
  PSCM_ASSERT(args.is_pair());
  auto pair = car(args);
  auto obj = cadr(args);
  if (!pair.is_pair()) {
    PSCM_THROW_EXCEPTION("Invalid set-cdr! args: " + args.to_string());
  }
  set_cdr(pair, obj);
  return Cell::none();
}

PSCM_DEFINE_BUILTIN_PROC(List, "null?") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return Cell(arg.is_nil());
}

PSCM_DEFINE_BUILTIN_PROC(List, "length") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  int count = 0;
  for_each(
      [&count](Cell expr, auto) {
        count++;
      },
      arg);
  return new Number(count);
}

PSCM_DEFINE_BUILTIN_PROC(List, "append") {
  if (args.is_nil()) {
    return nil;
  }
  auto pair = cons(nil, nil);
  auto it = pair;
  auto loop_var = args;
  while (loop_var.is_pair()) {
    auto list = car(loop_var);
    if (list.is_nil() || list.is_pair()) {
      auto p = list;
      while (p.is_pair()) {
        auto new_pair = cons(car(p), nil);
        if (!it->second.is_nil()) {
          PSCM_THROW_EXCEPTION("Wrong type argument in position 2 (expecting empty list): " + it->second.to_string());
        }
        it->second = new_pair;
        it = new_pair;
        p = cdr(p);
      }
      if (!p.is_nil()) {
        if (!it->second.is_nil()) {
          PSCM_THROW_EXCEPTION("Wrong type argument in position 2 (expecting empty list): " + it->second.to_string());
        }
        it->second = p;
      }
    }
    else {
      if (cdr(loop_var).is_nil()) {
        if (!it->second.is_nil()) {
          PSCM_THROW_EXCEPTION("Wrong type argument in position 2 (expecting empty list): " + it->second.to_string());
        }
        it->second = list;
      }
      else {
        PSCM_THROW_EXCEPTION("Wrong type argument in position 1 (expecting empty list): " + list.to_string());
      }
    }
    loop_var = cdr(loop_var);
  }
  return pair->second;
}

PSCM_DEFINE_BUILTIN_PROC(List, "reverse") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return reverse_argl(arg);
}

PSCM_DEFINE_BUILTIN_PROC(List, "reverse!") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return reverse_argl(arg);
}

PSCM_DEFINE_BUILTIN_PROC(List, "list-ref") {
  PSCM_ASSERT(args.is_pair());
  auto list = car(args);
  auto k = cadr(args);
  if (!k.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type (expecting exact integer): " + k.to_string());
  }
  auto num = k.to_num();
  if (!num->is_int()) {
    PSCM_THROW_EXCEPTION("Wrong type (expecting exact integer): " + k.to_string());
  }
  auto n = num->to_int();
  Cell ret;
  int i = 0;
  for_each(
      [&ret, &i, n](Cell expr, auto) {
        if (i == n) {
          ret = expr;
        }
        i++;
      },
      list);
  return ret;
}

PSCM_DEFINE_BUILTIN_PROC(List, "list-head") {
  PSCM_ASSERT(args.is_pair());
  auto list = car(args);
  auto k = cadr(args);
  if (!k.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type (expecting exact integer): " + k.to_string());
  }
  auto num = k.to_num();
  if (!num->is_int()) {
    PSCM_THROW_EXCEPTION("Wrong type (expecting exact integer): " + k.to_string());
  }
  auto n = num->to_int();
  auto max_len = list_length(list);
  if (max_len < n) {
    PSCM_THROW_EXCEPTION("out of range: " + pscm::to_string(max_len) + ", but got " + pscm::to_string(n));
  }
  auto ret = cons(nil, nil);
  auto it = ret;
  for (int i = 0; i < n; i++) {
    auto new_pair = cons(car(list), nil);
    it->second = new_pair;
    it = new_pair;
    list = cdr(list);
  }
  return ret->second;
}

PSCM_DEFINE_BUILTIN_PROC(List, "list-tail") {
  PSCM_ASSERT(args.is_pair());
  auto list = car(args);
  auto k = cadr(args);
  if (!k.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type (expecting exact integer): " + k.to_string());
  }
  auto num = k.to_num();
  if (!num->is_int()) {
    PSCM_THROW_EXCEPTION("Wrong type (expecting exact integer): " + k.to_string());
  }
  auto n = num->to_int();
  auto max_len = list_length(list);
  if (max_len < n) {
    PSCM_THROW_EXCEPTION("out of range: " + pscm::to_string(max_len) + ", but got " + pscm::to_string(n));
  }
  auto ret = cons(nil, nil);
  auto it = ret;
  for (int i = 0; i < max_len - n; i++) {
    list = cdr(list);
  }
  for (int i = 0; i < n; i++) {
    auto new_pair = cons(car(list), nil);
    it->second = new_pair;
    it = new_pair;
    list = cdr(list);
  }
  return ret->second;
}

PSCM_DEFINE_BUILTIN_PROC(List, "last-pair") {
  PSCM_ASSERT(args.is_pair());
  auto list = car(args);
  if (list.is_nil()) {
    return nil;
  }
  auto max_len = list_length(list);
  for (int i = 0; i < max_len - 1; i++) {
    list = cdr(list);
  }
  return cons(car(list), nil);
}

PSCM_DEFINE_BUILTIN_MACRO_PROC_WRAPPER(List, "for-each", Label::APPLY_FOR_EACH, "(proc . lists)") {
  PSCM_ASSERT(args.is_pair());
  Cell ret;
  auto proc = car(args);
  PSCM_ASSERT(proc.is_sym());
  auto lists = cdr(args);
  PSCM_ASSERT(lists.is_sym());
  proc = env->get(proc.to_sym());
  lists = env->get(lists.to_sym());
  int len = 0;
  ret = for_each(
      [&len](auto, auto) {
        len++;
      },
      lists);
  switch (len) {
  case 0: {
    break;
  }
  case 1: {
    ret = for_each(
        [&scm, env, proc](Cell expr, auto loc) {
          [[maybe_unused]] auto ret = scm.eval(env, cons(proc, list(list(quote, expr))));
        },
        car(lists));
    break;
  }
  case 2: {
    ret = for_each(
        [&scm, env, proc](Cell expr1, Cell expr2, auto loc) {
          [[maybe_unused]] auto ret = scm.eval(env, cons(proc, list(list(quote, expr1), list(quote, expr2))));
        },
        car(lists), cadr(lists));
    break;
  }
  default: {
    PSCM_THROW_EXCEPTION("not supported now");
  }
  }
  return list(quote, ret);
}

PSCM_DEFINE_BUILTIN_MACRO_PROC_WRAPPER(List, "map", Label::APPLY_MAP, "(proc . lists)") {
  PSCM_ASSERT(args.is_pair());
  Cell ret;
  auto proc = car(args);
  PSCM_ASSERT(proc.is_sym());
  auto lists = cdr(args);
  PSCM_INFO("lists: {0}", lists);
  PSCM_ASSERT(lists.is_sym());
  proc = env->get(proc.to_sym());
  lists = env->get(lists.to_sym());
  int len = 0;
  ret = for_each(
      [&len](auto, auto) {
        len++;
      },
      lists);
  switch (len) {
  case 0: {
    break;
  }
  case 1: {
    ret = map(
        [&scm, env, proc](Cell expr, auto loc) {
          return scm.eval(env, cons(proc, list(list(quote, expr))));
        },
        car(lists));
    break;
  }
  case 2: {
    ret = map(
        [&scm, env, proc](Cell expr1, Cell expr2, auto loc) {
          return scm.eval(env, cons(proc, list(list(quote, expr1), list(quote, expr2))));
        },
        car(lists), cadr(lists));
    break;
  }
  default: {
    PSCM_THROW_EXCEPTION("not supported now");
  }
  }
  return list(quote, ret);
}

Cell memq(Cell args) {
  auto obj = car(args);
  auto list = cadr(args);
  while (!list.is_nil()) {
    if (obj.is_eq(car(list)).to_bool()) {
      return list;
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

Cell memv(Cell args) {
  auto obj = car(args);
  auto list = cadr(args);
  while (!list.is_nil()) {
    if (obj.is_eqv(car(list)).to_bool()) {
      return list;
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

Cell member(Cell args) {
  auto obj = car(args);
  auto list = cadr(args);
  while (!list.is_nil()) {
    if (obj == car(list)) {
      return list;
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

static Cell scm_mem(Cell args, Cell::ScmCmp cmp_func) {
  auto obj = car(args);
  auto list = cadr(args);
  while (!list.is_nil()) {
    if (cmp_func(obj, car(list))) {
      return list;
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

PSCM_DEFINE_BUILTIN_PROC(Pair, "memq") {
  return scm_mem(args, Cell::is_eq);
}

PSCM_DEFINE_BUILTIN_PROC(Pair, "memv") {
  return scm_mem(args, Cell::is_eqv);
}

PSCM_DEFINE_BUILTIN_PROC(Pair, "member") {
  return scm_mem(args, Cell::is_equal);
}
} // namespace pscm