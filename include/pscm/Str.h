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

  void display() const;
  friend std::ostream& operator<<(std::ostream& os, const String& s);
  bool operator==(const String& rhs) const;

private:
  std::string data_;
};

} // namespace pscm
