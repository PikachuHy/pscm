#include "pscm/ApiManager.h"
#include "pscm/Function.h"
#include "pscm/Symbol.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include <tuple>
#include <vector>

namespace pscm {
std::vector<ApiManager *>& ApiManager::api_list() {
  static std::vector<ApiManager *> list;
  return list;
}

ApiManager::ApiManager(Cell::ScmFunc f, std::string name, SourceLocation loc)
    : f_(f)
    , name_(name)
    , loc_(loc) {
  ApiManager::api_list().push_back(this);
}

void ApiManager::install_proc(SymbolTable *env) {
  SPDLOG_INFO("proc count: {}", ApiManager::api_list().size());
  for (const auto& api_manager : ApiManager::api_list()) {
    auto f = api_manager->f_;
    auto name = api_manager->name_;
    auto loc = api_manager->loc_;
    auto sym = new Symbol(name);
    if (env->contains(sym)) {
      SPDLOG_ERROR("replication procedure defined in: {}", loc.to_string());
      PSCM_THROW_EXCEPTION("replication procedure install, previous is " + env->get(sym).to_string());
    }
    SPDLOG_INFO("insert {} from {}", name, loc.to_string());
    env->insert(sym, new Function(name, f));
  }
}

} // namespace pscm