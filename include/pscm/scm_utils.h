//
// Created by PikachuHy on 2023/3/17.
//

#pragma once
#include "pscm/Cell.h"
#include "pscm/Number.h"
#include "pscm/Pair.h"
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
    return cons(&t, nil);
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
    return cons(&t, list(args...));
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
