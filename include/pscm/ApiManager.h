#pragma once
#include "pscm/Cell.h"
#include <variant>
#include <vector>

namespace pscm {
class ApiManager {
public:
  ApiManager(Cell::ScmFunc f, UString name, SourceLocation loc = {});

  ApiManager(Cell::ScmMacro2 f, UString name, Label label, SourceLocation loc = {});

  ApiManager(Cell::ScmMacro2 f, UString name, Label label, UString args, SourceLocation loc = {});

  static void install_api(SymbolTable *env);
  static std::vector<ApiManager *>& api_list();
  static SymbolTable *private_env();

  bool is_func() const {
    return f_.index() == 0;
  }

  bool is_proc() const {
    return f_.index() == 2;
  }

  bool is_macro() const {
    return f_.index() == 1;
  }

private:
  static std::vector<ApiManager *> list_;
  std::variant<Cell::ScmFunc, Cell::ScmMacro2, Procedure *> f_;
  Label label_;
  UString name_;
  SourceLocation loc_;
};

} // namespace pscm