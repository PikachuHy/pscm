//
// Created by PikachuHy on 2023/3/4.
//

#pragma once
#include "pscm/Cell.h"
#include <functional>
#include <string>
#include <variant>

namespace pscm {

class Function {
public:
  Function(std::string name, Cell::ScmFunc f)
      : name_(std::move(name))
      , f_(f) {
  }

  Cell call(Cell args);
  friend std::ostream& operator<<(std::ostream& out, const Function& func);

private:
  std::string name_;
  std::variant<std::monostate, Cell::ScmFunc> f_;
};

} // namespace pscm
