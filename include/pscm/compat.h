#pragma once
#ifdef _MSVC_LANG
#if _MSVC_LANG <= 201402L
#define PSCM_STD_COMPAT 1
#else
#define PSCM_STD_COMPAT 0
#endif
#else
#if __cplusplus <= 201402L
#define PSCM_STD_COMPAT 1
#else
#define PSCM_STD_COMPAT 0
#endif
#endif
#if PSCM_STD_COMPAT
#include <mpark/variant.hpp>
#include <nonstd/string_view.hpp>
#include <spdlog/spdlog.h>
#include <tl/optional.hpp>

namespace pscm {
using StringView = nonstd::string_view;

} // namespace pscm

template <>
class fmt::formatter<pscm::StringView> {
public:
  auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
    // PSCM_THROW_EXCEPTION("not supported now");
    auto i = ctx.begin();
    return i;
  }

  auto format(const pscm::StringView& s, format_context& ctx) const -> format_context::iterator {
    return format_to(ctx.out(), "{}", std::string(s));
  }
};

namespace std {

inline int gcd(int a, int b) {
  return b > 0 ? gcd(b, a % b) : a;
}

inline int lcm(int a, int b) {
  auto t = gcd(a, b);
  auto aa = a / t;
  auto bb = b / t;
  return aa * bb * t;
}

template <typename... Ts>
using optional = tl::optional<Ts...>;
static constexpr tl::nullopt_t nullopt{ tl::nullopt_t::do_not_use{}, tl::nullopt_t::do_not_use{} };
template <typename... Ts>
using variant = mpark::variant<Ts...>;
using monostate = mpark::monostate;
template <std::size_t I, typename T>
using variant_alternative_t = mpark::variant_alternative_t<I, T>;

template <std::size_t I, typename... Ts>
inline constexpr variant_alternative_t<I, variant<Ts...>>& get(variant<Ts...>& v) {
  return mpark::detail::generic_get<I>(v);
}

template <std::size_t I, typename... Ts>
inline constexpr variant_alternative_t<I, variant<Ts...>>&& get(variant<Ts...>&& v) {
  return mpark::detail::generic_get<I>(mpark::lib::move(v));
}

template <std::size_t I, typename... Ts>
inline constexpr const variant_alternative_t<I, variant<Ts...>>& get(const variant<Ts...>& v) {
  return mpark::detail::generic_get<I>(v);
}

template <std::size_t I, typename... Ts>
inline constexpr const variant_alternative_t<I, variant<Ts...>>&& get(const variant<Ts...>&& v) {
  return mpark::detail::generic_get<I>(mpark::lib::move(v));
}

template <typename T, typename... Ts>
inline constexpr T& get(variant<Ts...>& v) {
  return get<mpark::detail::find_index_checked<T, Ts...>::value>(v);
}

template <typename T, typename... Ts>
inline constexpr T&& get(variant<Ts...>&& v) {
  return get<mpark::detail::find_index_checked<T, Ts...>::value>(mpark::lib::move(v));
}

template <typename T, typename... Ts>
inline constexpr const T& get(const variant<Ts...>& v) {
  return get<mpark::detail::find_index_checked<T, Ts...>::value>(v);
}

template <typename T, typename... Ts>
inline constexpr const T&& get(const variant<Ts...>&& v) {
  return get<mpark::detail::find_index_checked<T, Ts...>::value>(mpark::lib::move(v));
}

} // namespace std
#else
#include <string_view>

namespace pscm {
using StringView = std::string_view;
}
#endif