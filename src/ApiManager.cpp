#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/ApiManager.h"
#include "pscm/Function.h"
#include "pscm/Macro.h"
#include "pscm/Pair.h"
#include "pscm/Parser.h"
#include "pscm/Procedure.h"
#include "pscm/Symbol.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/logger/Logger.hpp"
#include "pscm/scm_utils.h"
#include <spdlog/fmt/fmt.h>
#include <tuple>
#include <vector>
#endif
PSCM_INLINE_LOG_DECLARE("pscm.core.ApiManager");

namespace pscm {
std::vector<ApiManager *>& ApiManager::api_list() {
  static std::vector<ApiManager *> list;
  return list;
}

SymbolTable *ApiManager::private_env() {
  static SymbolTable env;
  return &env;
}

ApiManager::ApiManager(Cell::ScmFunc f, UString name, SourceLocation loc)
    : f_(f)
    , name_(name)
    , loc_(loc) {
  ApiManager::api_list().push_back(this);
}

ApiManager::ApiManager(Cell::ScmMacro2 f, UString name, Label label, SourceLocation loc)
    : f_(f)
    , label_(label)
    , name_(name)
    , loc_(loc) {
  ApiManager::api_list().push_back(this);
}

ApiManager::ApiManager(Cell::ScmMacro2 f, UString name, Label label, UString args, SourceLocation loc)
    : label_(label)
    , name_(name)
    , loc_(loc) {
  auto proc_args = Parser(args).parse();
  PSCM_ASSERT(!proc_args.is_none());
  auto builtin_sym = new Symbol("builtin_" + name);
  auto builtin_macro = new Macro(name, label, f);
  ApiManager::private_env()->insert(builtin_sym, builtin_macro);
  Cell proc_body = cons(builtin_macro, proc_args);
  proc_body = list(proc_body);
  auto proc = new Procedure(new Symbol(name), proc_args, proc_body, ApiManager::private_env());
  f_ = proc;
  ApiManager::api_list().push_back(this);
}

void ApiManager::install_api(SymbolTable *env) {
  PSCM_TRACE("api count: {0}", ApiManager::api_list().size());
  for (const auto& api_manager : ApiManager::api_list()) {
    auto name = api_manager->name_;
    auto loc = api_manager->loc_;
    auto sym = new Symbol(name);
    if (env->contains(sym)) {
      PSCM_ERROR("replication api defined in: {0}", loc.to_string());
      PSCM_THROW_EXCEPTION("replication api install, previous is " + env->get(sym).to_string());
    }
    PSCM_TRACE("insert {0} from {0}", name, loc.to_string());
    if (api_manager->is_func()) {
      auto f = std::get<0>(api_manager->f_);
      env->insert(sym, new Function(name, f));
    }
    else if (api_manager->is_macro()) {
      auto f = std::get<1>(api_manager->f_);
      env->insert(sym, new Macro(name, api_manager->label_, f));
    }
    else if (api_manager->is_proc()) {
      auto f = std::get<2>(api_manager->f_);
      env->insert(sym, f);
    }
  }
}

} // namespace pscm