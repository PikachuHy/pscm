#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/Promise.h"
#include "pscm/ApiManager.h"
#include "pscm/Procedure.h"
#include "pscm/SchemeProxy.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#endif
namespace pscm {
PSCM_INLINE_LOG_DECLARE("pscm.core.Promise");
UString Promise::to_string() const{
  UString out;
  out += "#";
  out += "<";
  out += "promise";
  out += " ";
  auto proc_ = proc();
  PSCM_ASSERT(proc_);
  out += proc_->to_string();
  out += ">";
  return out;
}

void Promise::set_result(Cell ret) {
  ret_ = ret;
}

Cell Promise::result() const {
  PSCM_ASSERT(ret_.has_value());
  return ret_.value();
}

PSCM_DEFINE_BUILTIN_MACRO(Promise, "delay", Label::APPLY_DELAY) {
  PSCM_ASSERT(args.is_pair());
  Cell expr = car(args);
  auto proc = new Procedure(nullptr, nil, list(expr), env);
  return new Promise(proc);
}

PSCM_DEFINE_BUILTIN_MACRO_PROC_WRAPPER(Promise, "force", Label::APPLY_FORCE, "(promise)") {
  PSCM_ASSERT(args.is_pair());
  Cell promise = car(args);
  PSCM_ASSERT(promise.is_sym());
  promise = env->get(promise.to_sym());
  PSCM_ASSERT(promise.is_promise());
  auto p = promise.to_promise();
  Cell ret;
  if (p->ready()) {
    ret = p->result();
  }
  else {
    auto proc = p->proc();
    ret = scm.eval(env, list(proc));
    p->set_result(ret);
  }
  return list(quote, ret);
}

} // namespace pscm