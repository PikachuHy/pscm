#include "pscm/Module.h"
#include "pscm/ApiManager.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"

namespace pscm {
std::ostream& operator<<(std::ostream& os, const Module& m) {
  os << '#';
  os << '<';
  os << "module";
  os << ' ';
  if (!m.name_.is_none()) {
    os << m.name_;
  }
  os << ' ';
  os << &m;
  os << '>';
  return os;
}

PSCM_DEFINE_BUILTIN_PROC(Module, "current-module") {
  return new Module(Cell::none());
}

PSCM_DEFINE_BUILTIN_PROC(Module, "set-current-module") {
  return Cell::none();
}

PSCM_DEFINE_BUILTIN_PROC(Module, "module-ref") {
  PSCM_ASSERT(args.is_pair());
  auto module = car(args);
  auto name = cadr(args);
  PSCM_ASSERT(module.is_module());
  PSCM_ASSERT(name.is_sym());
  auto sym = name.to_symbol();
  if (*sym == "%module-public-interface"_sym) {
    return Cell::nil();
  }
  PSCM_THROW_EXCEPTION("not suppoted now");
}

PSCM_DEFINE_BUILTIN_PROC(Module, "module-map") {
  PSCM_ASSERT(args.is_pair());
  return Cell::nil();
}

PSCM_DEFINE_BUILTIN_PROC(Module, "re-export") {
  return Cell::none();
}
} // namespace pscm