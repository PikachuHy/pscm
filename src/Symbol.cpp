//
// Created by PikachuHy on 2023/2/25.
//

#include "pscm/Symbol.h"

namespace pscm {
Symbol callcc("call-with-current-continuation");
Symbol call_with_values("call-with-values");
Symbol values("values");
Symbol cond_else("else");

bool Symbol::operator==(const Symbol& sym) const {
  return name_ == sym.name_;
}

Symbol operator""_sym(const char *data, std::size_t len) {
  return Symbol(std::string_view(data, len));
}
} // namespace pscm