#pragma once
#if __cplusplus <= 201402L
#include <mpark/variant.hpp>
#include <tl/optional.hpp>

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
#endif