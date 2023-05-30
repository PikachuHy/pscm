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

bool operator==(const Keyword& lhs, const Keyword& rhs) {
  if (lhs.sym_ == rhs.sym_) {
    return true;
  }
  return *lhs.sym_ == *rhs.sym_;
}

HashCodeType Keyword::hash_code() const {
  return sym_->hash_code();
}

PSCM_DEFINE_BUILTIN_PROC(Keyword, "keyword?") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return Cell(arg.is_keyword());
}
} // namespace pscm