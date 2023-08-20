//
// Created by PikachuHy on 2023/7/22.
//
module;
#include <pscm/common_def.h>
module pscm.build;
import :DepsScanner;
import :BuildVariables;
import :Rule;
import :Expander;
import std;
import fmt;
import subprocess;
import nlohmann.json;

namespace pscm::build {

auto DepsScanner::scan(CompileBuildVariables variables, std::string_view cwd) -> ModuleDep {
  PSCM_INLINE_LOG_DECLARE("pscm.build.Action");
  std::vector<std::string> args;
  const auto& toolchain = CppBinaryRule::toolchain();
  args.push_back(toolchain.deps_scanner());
  args.push_back("-format=p1689");
  args.push_back("--");
  args.push_back(toolchain.cc());
  Expander expander(args);
  expander.expand_includes(ctx_->includes());
  expander.expand_includes(variables.get_includes());
  expander.expand_defines(ctx_->defines());
  expander.expand_defines(variables.get_defines());
  expander.expand_opts(variables.get_opts());
  expander.expand_modules(variables.modules());
  args.push_back("-x");
  args.push_back("c++-module");
  expander.expand_source_file(variables.get_source_file());
  expander.expand_output_file(variables.get_output_file());
  PSCM_DEBUG("RUN: {}", args);
  auto process =
      subprocess::run(args, subprocess::RunBuilder().cout(subprocess::PipeOption::pipe).cwd(std::string(cwd)));
  if (process.returncode != 0) {
    PSCM_ERROR("ERROR: return code: {}", process.returncode);
    std::exit(1);
  }
  // std::cout << "captured: " << process.cout << '\n';
  auto data = nlohmann::json::parse(process.cout);
  auto rules = data["rules"];
  PSCM_ASSERT(rules.size() == 1);
  auto rule = rules[0];
  ModuleDep dep{};
  auto requireModules = rule["requires"];
  for (auto item : requireModules) {
    dep.add_require_module(item["logical-name"]);
  }
  auto providesModules = rule["provides"];
  PSCM_ASSERT(providesModules.size() <= 1);
  if (!providesModules.empty()) {
    auto item = providesModules[0];
    if (item["is-interface"].get<bool>()) {
      dep.set_provide_module(item["logical-name"]);
    }
  }
  return dep;
}
} // namespace pscm::build