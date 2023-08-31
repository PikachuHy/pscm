//
// Created by PikachuHy on 2023/3/20.
//
#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/Str.h"
#include "pscm/Port.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <iostream>
#endif
namespace pscm {
PSCM_INLINE_LOG_DECLARE("pscm.core.String");
void String::display(Port& port) const {
  for (auto ch : data_) {
    port.write_char(ch);
  }
}

std::ostream& operator<<(std::ostream& os, const String& s) {
  return os << s.to_string();
}

UString String::to_string() const{
  UString data(data_);
  data.findAndReplace("\"", "\\\"");
  UString os;
  os += '"';
  os += data;
  os += '"';
  return os;

}

bool String::operator==(const String& rhs) const {
  return data_ == rhs.data_;
}

bool String::operator<(const String& rhs) const {
  return data_ < rhs.data_;
}

bool String::operator>(const String& rhs) const {
  return data_ > rhs.data_;
}

bool String::operator<=(const String& rhs) const {
  return data_ <= rhs.data_;
}

bool String::operator>=(const String& rhs) const {
  return data_ >= rhs.data_;
}

void String::set(std::size_t idx, UChar32 ch) {
  int32_t index = data_.moveIndex32(0, idx);
  bool pos_is_s = U_IS_SURROGATE(data_.charAt(index));
  bool ch_is_s = U_IS_SURROGATE(ch);
  if (!(pos_is_s || ch_is_s))
  {
    // replace between single codepoints
    data_.setCharAt(index, ch);
  }else if (pos_is_s && ch_is_s)
  {
    // replace between surrogate pairs
    data_.replace(index, 2, ch);
  }else
  {
    // replace a surrogate pair with single code point or conversely
    int32_t behind = data_.moveIndex32(index,1);
    UString&& behindstr = data_.tempSubStringBetween(behind, data_.length());
    data_.truncate(index);
    data_.append(ch);
    data_.append(behindstr);
  }
}

String String::to_downcase() const {
  return String(UString(data_).toLower());
}

String String::substring(std::int64_t start, std::int64_t end) const {
  UString res;
  data_.extractBetween(start, end, res);
  return String(std::move(res));
}

void String::fill(UChar32 ch) {
  data_.setTo(ch);
}

HashCodeType String::hash_code() const {
  return data_.hashCode();
}

String operator""_str(const char *data, std::size_t len) {
  return String(operator""_u(data, len));
}
} // namespace pscm
