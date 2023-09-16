//
// Created by PikachuHy on 2023/3/17.
//

#pragma once
#include "pscm/Cell.h"
#include "pscm/Logger.h"
#include "pscm/Number.h"
#include "pscm/Pair.h"
#include "pscm/Symbol.h"
#include "pscm/common_def.h"
#include <concepts>
#ifdef PSCM_USE_CXX20_MODULES
#include <fmt/core.h>
#include <fmt/format.h>
#else
#include <spdlog/fmt/fmt.h>
#endif
namespace pscm {
inline Cell list(Cell t) {
  return cons(t, nil);
}

inline Cell list(int32_t t) {
  return cons(new Number(t), nil);
}

inline Cell list(Symbol t) {
  return cons(new Symbol(t), nil);
}

template <typename... Args>
Cell list(int32_t t, Args... args);

template <typename... Args>
inline Cell list(Cell t, Args... args) {
  return cons(t, list(args...));
}

template <typename... Args>
inline Cell list(int32_t t, Args... args) {
  return cons(new Number(t), list(args...));
}

template <typename... Args>
inline Cell list(Symbol t, Args... args) {
  return cons(new Symbol(t), list(args...));
}

Cell reverse_argl(Cell argl);

template <typename Func>
Cell for_each(Func f, Cell list, SourceLocation loc = {}) {
  while (!list.is_nil()) {
    auto item = car(list);
    f(item, loc);
    auto l = cdr(list);
    list = l;
  }
  return Cell::none();
}

template <typename Func>
Cell for_each(Func f, Cell list1, Cell list2, SourceLocation loc = {}) {
  while (!list1.is_nil() && !list2.is_nil()) {
    auto item1 = car(list1);
    auto item2 = car(list2);
    f(item1, item2, loc);
    list1 = cdr(list1);
    list2 = cdr(list2);
  }
  PSCM_INLINE_LOG_DECLARE("pscm.core.for_each");
  PSCM_ASSERT(list1.is_nil() && list2.is_nil());
  return Cell::none();
}

template <typename Func>
Cell map(Func f, Cell list, SourceLocation loc = {}) {
  auto ret = cons(nil, nil);
  auto p = ret;
  while (!list.is_nil()) {
    auto item = car(list, loc);
    auto val = f(item, loc);
    auto new_item = cons(val, nil);
    p->second = new_item;
    p = new_item;
    auto l = cdr(list);
    list = l;
  }
  return ret->second;
}

template <typename Func>
Cell map(Func f, Cell list1, Cell list2, SourceLocation loc = {}) {
  auto ret = cons(nil, nil);
  auto p = ret;
  while ((!list1.is_nil() && !list2.is_nil())) {
    auto item1 = car(list1);
    auto item2 = car(list2);
    auto val = f(item1, item2, loc);
    auto new_item = cons(val, nil);
    p->second = new_item;
    p = new_item;
    list1 = cdr(list1);
    list2 = cdr(list2);
  }
  PSCM_INLINE_LOG_DECLARE("pscm.core.map");
  PSCM_ASSERT(list1.is_nil() && list2.is_nil());
  return ret->second;
}

} // namespace pscm

template <>
class fmt::formatter<pscm::Cell> {
public:
  constexpr auto parse(format_parse_context& ctx) {
    // PSCM_THROW_EXCEPTION("not supported now");
    auto i = ctx.begin();
    return i;
  }

  auto format(const pscm::Cell& cell, format_context& ctx) const {
    std::string str;
    cell.to_string().toUTF8String(str);
    return format_to(ctx.out(), "{}", str);
  }
};
