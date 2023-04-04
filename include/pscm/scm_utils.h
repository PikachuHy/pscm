//
// Created by PikachuHy on 2023/3/17.
//

#pragma once
#include "pscm/Cell.h"
#include "pscm/Number.h"
#include "pscm/Pair.h"
#include "pscm/Symbol.h"
#include "pscm/common_def.h"
#include <spdlog/spdlog.h>

namespace pscm {
template <typename T>
Cell list(T t) {
  if constexpr (std::same_as<T, Cell>) {
    return cons(t, nil);
  }
  else if constexpr (std::same_as<T, int32_t>) {
    return cons(new Number(t), nil);
  }
  else if constexpr (std::same_as<T, Symbol>) {
    return cons(new Symbol(t), nil);
  }
  else if constexpr (std::is_pointer_v<T>) {
    using U = std::remove_pointer_t<T>;
    return cons(t, nil);
  }
  else {
    static_assert(!sizeof(T), "not supported now");
  }
}

template <typename T, typename... Args>
Cell list(T t, Args... args) {
  if constexpr (std::same_as<T, Cell>) {
    return cons(t, list(args...));
  }
  else if constexpr (std::same_as<T, int32_t>) {
    return cons(new Number(t), list(args...));
  }
  else if constexpr (std::same_as<T, Symbol>) {
    return cons(new Symbol(t), list(args...));
  }
  else if constexpr (std::is_pointer_v<T>) {
    using U = std::remove_pointer_t<T>;
    return cons(t, list(args...));
  }
  else {
    static_assert(!sizeof(T), "not supported now");
  }
}

Cell reverse_argl(Cell argl);

Cell map(auto f, Cell list, SourceLocation loc = {}) {
  auto ret = cons(nil, nil);
  auto p = ret;
  while (!list.is_nil()) {
    auto item = car(list);
    auto val = f(item, loc);
    auto new_item = cons(val, nil);
    p->second = new_item;
    p = new_item;
    auto l = cdr(list);
    list = l;
  }
  return ret->second;
}

Cell map(auto f, Cell list1, Cell list2, SourceLocation loc = {}) {
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
  PSCM_ASSERT(list1.is_nil() && list2.is_nil());
  return ret->second;
}

} // namespace pscm

template <>
class fmt::formatter<pscm::Cell> {
public:
  auto parse(format_parse_context& ctx) {
    // PSCM_THROW_EXCEPTION("not supported now");
    auto i = ctx.begin();
    return i;
  }

  auto format(const pscm::Cell& cell, auto& ctx) const {
    return format_to(ctx.out(), "{}", cell.to_string());
  }
};
