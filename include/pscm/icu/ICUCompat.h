//
// Created by jingkaimori on 2023/8/16.
//

#pragma once
#include "msstl/charconv.hpp"
#include "unicode/chariter.h"
#include "unicode/msgfmt.h"
#include "unicode/schriter.h"
#include "unicode/uniset.h"
#include "unicode/unistr.h"
#include <cassert>
#include <charconv>
#include <climits>
#include <fstream>
// #include <numbers>
#include <string>
#include <variant>
#ifdef __ANDROID__
namespace std {
template <class _Tp>
concept integral = is_integral_v<_Tp>;

template <class _Tp>
concept floating_point = is_floating_point_v<_Tp>;
template <class _From, class _To>
concept convertible_to = is_convertible_v<_From, _To> && requires { static_cast<_To>(std::declval<_From>()); };
template <class _Tp, class _Up>
concept __same_as_impl = _IsSame<_Tp, _Up>::value;

template <class _Tp, class _Up>
concept same_as = __same_as_impl<_Tp, _Up> && __same_as_impl<_Up, _Tp>;

} // namespace std
#endif
namespace pscm {
/* There is no overload for floatpoint on MacOS, so import polyfill from mscharconv.
 */
namespace charconv {
template <typename floattype>
concept std_to_chars = requires(floattype f, char *const str) { std::to_chars(str, str + 1, f); };

template <typename floattype>
  requires std::floating_point<floattype> && std_to_chars<floattype>
std::to_chars_result to_chars(char *const str, char *const end, floattype f) {
  return std::to_chars(str, end, f);
};

template <typename floattype>
  requires(std::floating_point<floattype> && !std_to_chars<floattype>)
std::to_chars_result to_chars(char *const str, char *const end, floattype f) {
  auto [ptr, ec] = msstl::to_chars(str, end, f);
  return { ptr, ec };
};
} // namespace charconv

using UString = U_ICU_NAMESPACE::UnicodeString;
using UIterator = U_ICU_NAMESPACE::StringCharacterIterator;
using USet = U_ICU_NAMESPACE::UnicodeSet;
using UIteratorP = U_ICU_NAMESPACE::CharacterIterator *;
using UFormattable = U_ICU_NAMESPACE::Formattable;
using UFormatter = U_ICU_NAMESPACE::MessageFormat;
static const UChar32 UIteratorDone = U_ICU_NAMESPACE::CharacterIterator::DONE;

enum class FileStatus { NOT_FOUND, INTERNAL_ERROR };

enum class ParseStatus { FORMAT_ERROR, INTERNAL_ERROR };

UChar32 operator""_u(char arg);
UChar32 operator""_u(char16_t arg);
const UString operator""_u(const char *str, std::size_t len);
const UString get_const_string(const UString& src);
bool if_file_exists(UString filename);
void open_fstream(std::fstream& stream, UString path, std::ios_base::openmode = std::ios_base::in | std::ios_base::out);
const std::variant<UString, FileStatus> read_file(const UString& filename);

/**
 * Format integer to string, omit locale settings. This function should only
 * be used to print techinique numbers such line number or pointer value.
 */
template <std::integral inttype>
const UString to_programmatic_string(inttype integer, int radix = 10) {
  constexpr std::size_t buf_size = (sizeof(inttype) * CHAR_BIT) + 1;
  char buf[buf_size];
  auto res = std::to_chars(buf, buf + buf_size, integer, radix);
  assert(res.ec == std::errc());
  return UString(buf, res.ptr - buf, UString::EInvariant::kInvariant);
}

/**
 * Format integer to string, omit locale settings. This function should only
 * be used to print techinique numbers such line number or pointer value.
 */
template <std::floating_point floattype>
const UString to_programmatic_string(floattype integer) {
  constexpr std::size_t buf_size = 256;
  char buf[buf_size];
  auto res = pscm::charconv::to_chars(buf, buf + buf_size, integer);
  assert(res.ec == std::errc());
  return UString(buf, res.ptr - buf, UString::EInvariant::kInvariant);
}

const UString to_string(double num);
const UString to_string(int integer);
const UString to_string(std::int64_t integer);
const UString to_string(std::size_t integer);
const UString to_string(const void *pointer);
std::variant<double, std::int64_t, ParseStatus> double_from_string(const UString& str);
// polyfill missing ustream.h on wasm
#if defined(WASM_PLATFORM)
std::ostream& operator<<(std::ostream& out, const UString& obj);
#endif
} // namespace pscm

namespace U_ICU_NAMESPACE {
enum ICUBoundries { DONE = ForwardCharacterIterator::DONE };

UChar32 operator*(const StringCharacterIterator& iter);

StringCharacterIterator& operator++(StringCharacterIterator& iter);

bool operator!=(const StringCharacterIterator& iter, ICUBoundries);

StringCharacterIterator begin(UnicodeString str);
ICUBoundries end(UnicodeString iter);
}; // namespace U_ICU_NAMESPACE

namespace std {
template <>
struct hash<pscm::UString> {
  using result_type = std::size_t;
  using argument_type = pscm::UString;
  std::size_t operator()(const pscm::UString& rhs) const;
};

} // namespace std
