//
// Created by jingkaimori on 2023/8/16.
//
#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm.compat;
import pscm.icu;
#else
#include "pscm/common_def.h"
#include "pscm/icu/ICUCompat.h"
#include "unicode/numfmt.h"
#include "unicode/unistr.h"
#include <iostream>
#if PSCM_STD_COMPAT
#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif
#endif
namespace pscm {

PSCM_INLINE_LOG_DECLARE("pscm.core.ICU");

namespace icu {
class Formatter {
public:
  static auto& instance() {
    static Formatter formatter;
    return formatter;
  }

  auto format(const auto& val, auto& res) {
    formatter_->format(val, res);
  }

  auto parse(auto& str, auto& ress, auto& stat) {
    formatter_->parse(str, ress, stat);
  }

private:
  Formatter() {
    const auto locale = U_ICU_NAMESPACE::Locale::getDefault();
    formatter_ =
        U_ICU_NAMESPACE::NumberFormat::createInstance(locale, UNumberFormatStyle::UNUM_DECIMAL, formatter_stat);
    if (!(U_SUCCESS(formatter_stat))) {
      std::string msg = u_errorName(formatter_stat);
      std::cout << msg << std::endl;
    }
    PSCM_ASSERT(U_SUCCESS(formatter_stat));
  }

  UErrorCode formatter_stat = U_ZERO_ERROR;
  U_ICU_NAMESPACE::NumberFormat *formatter_;
};
} // namespace icu

// static UErrorCode formatter_stat = U_ZERO_ERROR;
// static const auto locale = U_ICU_NAMESPACE::Locale::getDefault();
// static auto formatter =
//     U_ICU_NAMESPACE::NumberFormat::createInstance(locale, UNumberFormatStyle::UNUM_DECIMAL, formatter_stat);

const UString operator""_u(const char *arg, std::size_t len) {
  return UString::fromUTF8(U_ICU_NAMESPACE::StringPiece(arg, len));
}

UChar32 operator""_u(char arg) {
  return static_cast<UChar32>(arg);
}

UChar32 operator""_u(char16_t arg) {
  return static_cast<UChar32>(arg);
}

const UString to_string(int integer) {
  // PSCM_ASSERT(U_SUCCESS(formatter_stat));
  UString res;
  icu::Formatter::instance().format(integer, res);
  return res;
}

const UString to_string(std::int64_t integer) {
  // PSCM_ASSERT(U_SUCCESS(formatter_stat));
  UString res;
  icu::Formatter::instance().format(integer, res);
  return res;
}

const UString to_string(std::size_t integer) {
  // PSCM_ASSERT(U_SUCCESS(formatter_stat));
  UString res;
  icu::Formatter::instance().format(static_cast<std::int64_t>(integer), res);
  return res;
}

const UString to_string(double num) {
  // PSCM_ASSERT(U_SUCCESS(formatter_stat));
  UString res;
  icu::Formatter::instance().format(num, res);
  return res;
}

const UString to_string(const void *ptr) {
  return to_programmatic_string<std::int64_t>(reinterpret_cast<std::int64_t>(ptr));
}

std::variant<double, std::int64_t, ParseStatus> double_from_string(const UString& str) {
  // PSCM_ASSERT(U_SUCCESS(formatter_stat));
  U_ICU_NAMESPACE::Formattable ress;
  UErrorCode stat = U_ZERO_ERROR;
  icu::Formatter::instance().parse(str, ress, stat);
  if (U_SUCCESS(stat)) {
    switch (ress.getType()) {
    case U_ICU_NAMESPACE::Formattable::kDouble:
      return ress.getDouble();
    case U_ICU_NAMESPACE::Formattable::kLong:
      return ress.getLong();
    case U_ICU_NAMESPACE::Formattable::kInt64:
      return ress.getInt64();
    default:
      return ParseStatus::FORMAT_ERROR;
    }
  }
  else {
    if (stat == U_INVALID_FORMAT_ERROR) {
      return ParseStatus::FORMAT_ERROR;
    }
    else {
      return ParseStatus::INTERNAL_ERROR;
    }
  }
}

void open_fstream(std::fstream& stream, UString path, std::ios_base::openmode mode) {
  std::string path_utf8;
  path.toUTF8String(path_utf8);
  stream.open(path_utf8, mode);
  stream.seekg(0, std::ios_base::beg);
  PSCM_INFO("path: {0}, file opened: {1}", path, stream.is_open())
}

bool if_file_exists(UString fname) {
  std::string path_utf8;
  fname.toUTF8String(path_utf8);
  return fs::exists(path_utf8);
}

const UString get_const_string(const UString& src) {
  UString res(false, src.getBuffer(), src.length());
  return res;
}

const std::variant<UString, FileStatus> read_file(const UString& filename) {
  if (!if_file_exists(filename)) {
    PSCM_ERROR("file not found: {0}", filename);
    return FileStatus::NOT_FOUND;
  }
  std::fstream ifs;
  open_fstream(ifs, filename, std::ios::in);
  if (!ifs.is_open()) {
    PSCM_ERROR("load file {0} error", filename);
    return FileStatus::INTERNAL_ERROR;
  }
  ifs.seekg(0, ifs.end);
  auto sz = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  std::string code;
  code.resize(sz);
  ifs.read((char *)code.data(), sz);
  // last series of codepoint in buffer may be NUL, so let icu detect size
  UString res(code.data());
  return res;
}

#if defined(WASM_PLATFORM)
std::ostream& operator<<(std::ostream& f_, const UString& obj) {
  std::string utf8;
  obj.toUTF8String(utf8);
  f_ << utf8;
  return f_;
}
#endif
} // namespace pscm

namespace U_ICU_NAMESPACE {
UChar32 operator*(const StringCharacterIterator& iter) {
  return iter.current32();
}

StringCharacterIterator& operator++(StringCharacterIterator& iter) {
  iter.next32();
  return iter;
}

bool operator!=(const StringCharacterIterator& iter, ICUBoundries) {
  return iter.current32() != DONE;
}

StringCharacterIterator begin(UnicodeString str) {
  return StringCharacterIterator(str);
}

ICUBoundries end(UnicodeString iter) {
  return ICUBoundries::DONE;
}
}; // namespace U_ICU_NAMESPACE

namespace std {
std::size_t hash<pscm::UString>::operator()(const pscm::UString& cell) const {
  return cell.hashCode();
};

} // namespace std
