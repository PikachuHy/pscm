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

  [[nodiscard]] bool empty() const {
    return data_.empty();
  }

  void display() const;
  friend std::ostream& operator<<(std::ostream& os, const String& s);
  bool operator==(const String& rhs) const;

  std::string_view str() const {
    return data_;
  }

private:
  std::string data_;
};

String operator""_str(const char *data, std::size_t len);

} // namespace pscm
