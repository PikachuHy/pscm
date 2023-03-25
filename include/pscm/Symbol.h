//
// Created by PikachuHy on 2023/2/25.
//

#pragma once
#include <string>
#include <string_view>

namespace pscm {

class Symbol {
public:
  Symbol(std::string_view sym)
      : name_(sym.data(), sym.size()) {
  }

  std::string_view name() const {
    return name_;
  }

  bool operator==(const Symbol& sym) const;

private:
  std::string name_;
};

Symbol operator""_sym(const char *data, std::size_t len);
extern Symbol callcc;
extern Symbol call_with_values;
extern Symbol values;
extern Symbol cond_else;
} // namespace pscm
