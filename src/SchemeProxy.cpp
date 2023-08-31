#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/SchemeProxy.h"
#include "pscm/Scheme.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#endif
namespace pscm {
PSCM_INLINE_LOG_DECLARE("pscm.core.SchemeProxy");

Module *SchemeProxy::current_module() const {
  return scm_.current_module();
};

void SchemeProxy::set_current_module(Module *m) {
  scm_.current_module_ = m;
}

Cell SchemeProxy::eval(SymbolTable *env, Cell expr) {
  return scm_.eval(env, expr);
}

bool SchemeProxy::load(const UString& filename) {
  return scm_.load(filename);
}

Module *SchemeProxy::create_module(Cell module_name) {
  PSCM_ASSERT(module_name.is_pair());
  PSCM_ASSERT(scm_.module_map_.find(module_name) == scm_.module_map_.end());
  auto m = scm_.create_module(module_name);
  scm_.module_map_[module_name] = m;
  scm_.current_module_ = m;
  return m;
}

bool SchemeProxy::has_module(Cell module_name) const {
  return scm_.module_map_.find(module_name) != scm_.module_map_.end();
}

Module *SchemeProxy::get_module(Cell module_name) const {
  return scm_.module_map_.at(module_name);
}

void SchemeProxy::load_module(const UString& filename, Cell module_name) {
  scm_.load_module(filename, module_name);
}

void SchemeProxy::vau_hack(Symbol *sym, Cell value) {
  scm_.vau_hack_env_->insert(sym, value);
}
} // namespace pscm