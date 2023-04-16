//
// Created by PikachuHy on 2023/3/20.
//

#pragma once
#include <ostream>
#include <string>

namespace pscm {

class String {
public:
  String(std::string data)
      : data_(std::move(data)) {
  }

  String(std::size_t sz, char ch) {
    data_.resize(sz);
    std::fill(data_.begin(), data_.end(), ch);
  }

  [[nodiscard]] bool empty() const {
    return data_.empty();
  }

  void display() const;
  friend std::ostream& operator<<(std::ostream& os, const String& s);
  bool operator==(const String& rhs) const;
  bool operator<(const String& rhs) const;
  bool operator>(const String& rhs) const;
  bool operator<=(const String& rhs) const;
  bool operator>=(const String& rhs) const;

  std::string_view str() const {
    return data_;
  }

  [[nodiscard]] std::size_t length() const {
    return data_.size();
  }

  void set(std::size_t idx, char ch);

  [[nodiscard]] String to_downcase() const;

  [[nodiscard]] String substring(std::int64_t start, std::int64_t end) const;

  void fill(char ch);

private:
  std::string data_;
};

String operator""_str(const char *data, std::size_t len);

} // namespace pscm
