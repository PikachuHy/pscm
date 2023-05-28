#include "pscm/Cell.h"
#include <vector>

namespace pscm {
class ApiManager {
public:
  ApiManager(Cell::ScmFunc f, std::string name, SourceLocation loc = {});

  ApiManager(Cell::ScmMacro2 f, std::string name, Label label, SourceLocation loc = {});

  ApiManager(Cell::ScmMacro2 f, std::string name, Label label, const char *args, SourceLocation loc = {});

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
  std::string name_;
  SourceLocation loc_;
};

} // namespace pscm