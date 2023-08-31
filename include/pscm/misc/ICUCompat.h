//
// Created by jingkaimori on 2023/8/16.
//

#pragma once
#include "unicode/chariter.h"
#include "unicode/schriter.h"
#include "unicode/uniset.h"
#include "unicode/unistr.h"
#include "unicode/msgfmt.h"
#include <fstream>
#include <string>
#include <variant>

namespace pscm {
  using UString = U_ICU_NAMESPACE::UnicodeString;
  using UIterator = U_ICU_NAMESPACE::StringCharacterIterator;
  using USet = U_ICU_NAMESPACE::UnicodeSet;
  using UIteratorP = U_ICU_NAMESPACE::CharacterIterator*;
  using UFormattable = U_ICU_NAMESPACE::Formattable;
  using UFormatter = U_ICU_NAMESPACE::MessageFormat;
  static const UChar32 UIteratorDone = U_ICU_NAMESPACE::CharacterIterator::DONE;

  enum class FileStatus {
    NOT_FOUND,
    INTERNAL_ERROR
  };

  enum class ParseStatus {
    FORMAT_ERROR,
    INTERNAL_ERROR
  };

  UChar32 operator""_u(char arg);
  UChar32 operator""_u(char16_t arg);
  const UString operator""_u(const char* str, std::size_t len);
  const UString get_const_string(const UString & src);
  bool if_file_exists(UString filename);
  void open_fstream(
    std::fstream& stream,
    UString path,
    std::ios_base::openmode = std::ios_base::in | std::ios_base::out
  );
  const std::variant<UString, FileStatus> read_file(
    const UString& filename);

  const UString to_string(int integer);
  const UString to_string(const void* pointer);
  std::variant<double, ParseStatus> double_from_string(const UString& str);
} // namespace pscm

namespace U_ICU_NAMESPACE  {
  enum ICUBoundries {
    DONE = ForwardCharacterIterator::DONE
  };
  UChar32 operator*(StringCharacterIterator iter);

  StringCharacterIterator & operator++(StringCharacterIterator iter);

  bool operator!=(StringCharacterIterator iter, ICUBoundries);

  StringCharacterIterator & begin(UnicodeString str);
  ICUBoundries end(UnicodeString iter);
};

namespace std {
template <>
struct hash<pscm::UString> {
  using result_type = std::size_t;
  using argument_type = pscm::UString;
  std::size_t operator()(const pscm::UString& rhs) const;
};

} // namespace std
