#include "pscm/Keyword.h"
#include "pscm/ApiManager.h"
#include "pscm/Pair.h"
#include "pscm/Symbol.h"
#include "pscm/common_def.h"

namespace pscm {
std::ostream& operator<<(std::ostream& os, const Keyword& keyword) {
  PSCM_ASSERT(keyword.sym_);
  os << '#';
  os << keyword.sym_->name();
  return os;
}

PSCM_DEFINE_BUILTIN_PROC(Keyword, "keyword?") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return Cell(arg.is_keyword());
}
} // namespace pscm