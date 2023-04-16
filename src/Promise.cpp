#include "pscm/Promise.h"
#include "pscm/Procedure.h"
#include "pscm/scm_utils.h"

namespace pscm {
std::ostream& operator<<(std::ostream& out, const Promise& promise) {
  out << "#";
  out << "<";
  out << "promise";
  out << " ";
  auto proc = promise.proc();
  PSCM_ASSERT(proc);
  out << *proc;
  out << ">";
  return out;
}

void Promise::set_result(Cell ret) {
  ret_ = ret;
}

Cell Promise::result() const {
  PSCM_ASSERT(ret_.has_value());
  return ret_.value();
}
} // namespace pscm