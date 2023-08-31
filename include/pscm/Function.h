//
// Created by PikachuHy on 2023/3/4.
//

#pragma once
#include "pscm/Cell.h"
#include "pscm/misc/ICUCompat.h"
#include <functional>
#include <string>
#include <variant>

namespace pscm {

class Function {
public:
  typedef Cell (*ScmFunc2)(Cell, SourceLocation);

  Function(UString name, Cell::ScmFunc f)
      : name_(std::move(name))
      , f_(f) {
  }

  Function(UString name, ScmFunc2 f)
      : name_(std::move(name))
      , f_(f) {
  }

  Cell call(Cell args, SourceLocation loc = {});
  UString to_string() const;

  const UString name() const {
    return get_const_string(name_);
  }

private:
  UString name_;
  std::variant<std::monostate, Cell::ScmFunc, ScmFunc2> f_;
};

} // namespace pscm
