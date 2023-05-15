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
  typedef Cell (*ScmFunc2)(Cell, SourceLocation);

  Function(std::string name, Cell::ScmFunc f)
      : name_(std::move(name))
      , f_(f) {
  }

  Function(std::string name, ScmFunc2 f)
      : name_(std::move(name))
      , f_(f) {
  }

  Cell call(Cell args, SourceLocation loc = {});
  friend std::ostream& operator<<(std::ostream& out, const Function& func);

  std::string_view name() const {
    return name_;
  }

private:
  std::string name_;
  std::variant<std::monostate, Cell::ScmFunc, ScmFunc2> f_;
};

} // namespace pscm
