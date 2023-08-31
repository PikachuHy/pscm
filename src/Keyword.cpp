#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/Keyword.h"
#include "pscm/ApiManager.h"
#include "pscm/Pair.h"
#include "pscm/Symbol.h"
#include "pscm/common_def.h"
#include <spdlog/fmt/fmt.h>
#endif
namespace pscm {
PSCM_INLINE_LOG_DECLARE("pscm.core.Keyword");

UString Keyword::to_string() const{
  PSCM_ASSERT(sym_);
  UString res;
  res += '#';
  res += sym_->name();
  return res;
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

PSCM_DEFINE_BUILTIN_PROC(Keyword, "keyword->symbol") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_keyword());
  auto keyword = arg.to_keyword();
  return keyword->sym();
}
} // namespace pscm