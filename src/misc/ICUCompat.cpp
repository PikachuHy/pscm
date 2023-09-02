//
// Created by jingkaimori on 2023/8/16.
//

#pragma once
#include "pscm/misc/ICUCompat.h"
#include "pscm/common_def.h"
#include "unicode/unistr.h"
#include "unicode/numfmt.h"
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

  const UString to_string(std::int64_t integer){
    PSCM_ASSERT(U_SUCCESS(formatter_stat));
    UString res;
    formatter->format(integer, res);
    return res;
  }

  const UString to_string(std::size_t integer){
    PSCM_ASSERT(U_SUCCESS(formatter_stat));
    UString res;
    formatter->format(static_cast<std::int64_t>(integer), res);
    return res;
  }

  const UString to_string(double num){
    PSCM_ASSERT(U_SUCCESS(formatter_stat));
    UString res;
    formatter->format(num, res);
    return res;
  }

  const UString to_string(const void* ptr){
    return to_programmatic_string<std::int64_t, 16>(reinterpret_cast<std::int64_t>(ptr));
  }

  std::variant<double, std::int64_t, ParseStatus> double_from_string(const UString& str){
    PSCM_ASSERT(U_SUCCESS(formatter_stat));
    U_ICU_NAMESPACE::Formattable ress;
    UErrorCode stat = U_ZERO_ERROR;
    formatter->parse(str, ress, stat);
    if (U_SUCCESS(stat))
    {
      switch (ress.getType() )
      {
      case U_ICU_NAMESPACE::Formattable::kDouble:
        return ress.getDouble();
      case U_ICU_NAMESPACE::Formattable::kLong:
        return ress.getLong();
      case U_ICU_NAMESPACE::Formattable::kInt64:
        return ress.getInt64();
      default:
        return ParseStatus::FORMAT_ERROR;
      }
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
    std::ios_base::openmode mode
  ){
    std::string path_utf8;
    path.toUTF8String(path_utf8);
    stream.open(path_utf8, mode);
    PSCM_INFO("path: {0}, file opened: {1}", path, stream.is_open())
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
  UChar32 operator*(const StringCharacterIterator& iter){
    return iter.current32();
  }

  StringCharacterIterator & operator++(StringCharacterIterator& iter){
    iter.next32();
    return iter;
  }

  bool operator!=(const StringCharacterIterator& iter, ICUBoundries){
    return iter.current32() != DONE;
  }

  StringCharacterIterator begin(UnicodeString str){
    return StringCharacterIterator(str);
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
