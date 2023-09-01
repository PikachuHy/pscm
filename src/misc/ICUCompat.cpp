//
// Created by jingkaimori on 2023/8/16.
//

#pragma once
#include "pscm/misc/ICUCompat.h"
#include "pscm/common_def.h"
#include "unicode/unistr.h"
#include "unicode/numfmt.h"
#include <charconv>
#if PSCM_STD_COMPAT
#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

namespace pscm {

PSCM_INLINE_LOG_DECLARE("pscm.core.Module");

  static UErrorCode formatter_stat = U_ZERO_ERROR;
  static const auto locale = U_ICU_NAMESPACE::Locale::getDefault();
  static auto formatter = U_ICU_NAMESPACE::NumberFormat::createInstance(locale, UNumberFormatStyle::UNUM_DECIMAL, formatter_stat);

  const UString operator""_u(const char* arg, std::size_t len){
    return UString::fromUTF8(U_ICU_NAMESPACE::StringPiece (arg, len));
  }

  UChar32 operator""_u(char arg){
    return static_cast<UChar32>(arg);
  }

  UChar32 operator""_u(char16_t arg){
    return static_cast<UChar32>(arg);
  }

  const UString to_string(int integer){
    PSCM_ASSERT(U_SUCCESS(formatter_stat));
    UString res;
    formatter->format(integer, res);
    return res;
  }

  const UString to_string(const void* integer){
    PSCM_ASSERT(U_SUCCESS(formatter_stat));
    char buf[16];
    auto res = std::to_chars(buf, buf + 16, reinterpret_cast<int64_t>(integer), 16);
    UString ustr(buf, res.ptr - buf, UString::EInvariant::kInvariant);
    return ustr;
  }

  std::variant<double, ParseStatus> double_from_string(const UString& str){
    PSCM_ASSERT(U_SUCCESS(formatter_stat));
    U_ICU_NAMESPACE::Formattable ress;
    UErrorCode stat = U_ZERO_ERROR;
    formatter->parse(str, ress, stat);
    if (U_SUCCESS(stat))
    {
      return ress.getDouble();
    }else
    {
      if (stat == U_INVALID_FORMAT_ERROR)
      {
        return ParseStatus::FORMAT_ERROR;
      }else
      {
        return ParseStatus::INTERNAL_ERROR;
      }
    }
  }

  void open_fstream(
    std::fstream& stream,
    UString path,
    std::ios_base::openmode
  ){
    std::string path_utf8;
    path.toUTF8String(path_utf8);
    stream.open(path_utf8);
  }

  bool if_file_exists(UString fname){
    std::string path_utf8;
    fname.toUTF8String(path_utf8);
    return fs::exists(path_utf8);
  }

  const UString get_const_string(const UString & src){
    UString res(false, src.getBuffer(), src.length());
    return res;
  }

  const std::variant<UString, FileStatus> read_file(const UString& filename){
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
    UString res(code.data(), sz);
    return res;
  }
} // namespace pscm

namespace U_ICU_NAMESPACE  {
  UChar32 operator*(StringCharacterIterator iter){
    return iter.current32();
  }

  StringCharacterIterator & operator++(StringCharacterIterator iter){
    iter.next32();
    return iter;
  }

  bool operator!=(StringCharacterIterator iter, ICUBoundries){
    return iter.current32() != DONE;
  }

  StringCharacterIterator & begin(UnicodeString str){
    StringCharacterIterator siter(str);
    return siter;
  }
  ICUBoundries end(UnicodeString iter){
    return ICUBoundries::DONE;
  }
};

namespace std {
std::size_t hash<pscm::UString>::operator()(const pscm::UString& cell) const {
  return cell.hashCode();
};

} // namespace std
