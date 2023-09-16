//
// Created by PikachuHy on 2023/3/20.
//

#pragma once
#include "Cell.h"
#include "unicode/unistr.h"
#include <ostream>
#include <string>

namespace pscm {
class Port;

class String {
public:
  String(UString data)
      : data_(std::move(data)) {
  }

  String(std::size_t sz, UChar32 ch)
      : data_(sz, ch, sz) {
  }

  [[nodiscard]] bool empty() const {
    return data_.isEmpty();
  }

  void display(Port& port) const;
  bool operator==(const String& rhs) const;
  bool operator<(const String& rhs) const;
  bool operator>(const String& rhs) const;
  bool operator<=(const String& rhs) const;
  bool operator>=(const String& rhs) const;
  UString to_string() const;

  const UString& str() const {
    return data_;
  }

  [[nodiscard]] std::size_t length() const {
    return data_.length();
  }

  void set(std::size_t idx, UChar32 ch);

  [[nodiscard]] String to_downcase() const;

  [[nodiscard]] String substring(std::int64_t start, std::int64_t end) const;

  void fill(UChar32 ch);

  HashCodeType hash_code() const;

private:
  UString data_;
};

String operator""_str(const char *data, std::size_t len);

} // namespace pscm
