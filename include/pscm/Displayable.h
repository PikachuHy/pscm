//
// Created by PikachuHy on 2023/3/12.
//

#pragma once
#include "pscm/misc/ICUCompat.h"
#include "unicode/unistr.h"
#if defined(WASM_PLATFORM)
#else
#include "unicode/ustream.h"
#endif
#include <iostream>
#include <type_traits>
#include <utility>

namespace pscm {

template <typename T>
concept DisplayableObj = requires(T a) {
  { a.to_string() } -> std::same_as<UString>;
};

template <DisplayableObj T>
const UString to_string(T obj) {
  return obj.to_string();
};

template <typename T>
concept Displayable = requires(T a) {
  { pscm::to_string(a) } -> std::same_as<const UString>;
};

#if defined(WASM_PLATFORM)
template <typename T>
  requires Displayable<T> && (!(std::convertible_to<T, char *> || std::same_as<T, char>))
std::ostream& operator<<(std::ostream& out, const T& obj) {
  std::string utf8;
  pscm::to_string(obj).toUTF8String(utf8);
  return out << utf8;
}
#else
template <typename T>
  requires Displayable<T> && (!(std::convertible_to<T, char *> || std::same_as<T, char>))
std::ostream& operator<<(std::ostream& out, const T& obj) {
  return out << pscm::to_string(obj);
}
#endif
} // namespace pscm
