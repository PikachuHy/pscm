//
// Created by PikachuHy on 2023/3/19.
//

#include "pscm/Char.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <string>
using namespace std::string_literals;

namespace pscm {
static Char ch_at("@");
static Char ch_star("*");

Cell Char::from(char ch) {
  if (ch == '@') {
    return &ch_at;
  }
  else if (ch == '*') {
    return &ch_star;
  }
  else {
    PSCM_THROW_EXCEPTION("unsupported char: "s + ch);
  }
}

std::ostream& operator<<(std::ostream& out, const Char& ch) {
  PSCM_ASSERT(!ch.ch_.empty());
  return out << "#\\" << ch.ch_;
}

void Char::display() const {
  PSCM_ASSERT(!ch_.empty());
  std::cout << ch_;
  std::cout.flush();
}
} // namespace pscm