#include "pscm/Cell.h"
#include <vector>

namespace pscm {
class ApiManager {
public:
  ApiManager(Cell::ScmFunc f, std::string name, SourceLocation loc = {});

  static void install_proc(SymbolTable *env);
  static std::vector<ApiManager *>& api_list();

private:
  static std::vector<ApiManager *> list_;
  Cell::ScmFunc f_;
  std::string name_;
  SourceLocation loc_;
};
} // namespace pscm