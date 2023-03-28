//
// Created by PikachuHy on 2023/2/25.
//

#include "pscm/Symbol.h"
#include <ostream>

namespace pscm {
Symbol callcc("call-with-current-continuation");
Symbol call_with_values("call-with-values");
Symbol values("values");
Symbol cond_else("else");

std::ostream& operator<<(std::ostream& out, const Symbol& sym) {
  auto name = sym.name_;
  if (name.find(' ') != std::string::npos) {
    out << "#";
    out << "{";
    for (auto ch : name) {
      if (ch == ' ') {
        out << "\\";
      }
      out << ch;
    }
    out << "}";
    out << "#";
    return out;
  }
  return out << name;
}

bool Symbol::operator==(const Symbol& sym) const {
  return name_ == sym.name_;
}

Symbol operator""_sym(const char *data, std::size_t len) {
  return Symbol(std::string_view(data, len));
}

Symbol *gensym() {
  static int index = 0;
  auto sym = new Symbol(" g" + std::to_string(index++));
  return sym;
}
} // namespace pscm