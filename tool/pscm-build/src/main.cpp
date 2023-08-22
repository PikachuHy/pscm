module;
#include <pscm/Logger.h>
#include <pscm/common_def.h>
export module pscm.build;
import :Rule;
import :RuleContext;
import :Artifact;
import :Label;
import pscm;
import std;
import fmt;
import glob;
using namespace pscm::build;
using namespace std::string_literals;
PSCM_INLINE_LOG_DECLARE("pscm.build");

// when run pscm-build
// the repo.pscm file must exist
// which indicate the repo root path
// for example,
// /somepath/repo.pscm
// /somepath/foo/bar/repo.pscm
// if run pscm-build on /somepath, the repo root path is /somepath
// if run pscm-build on /somepath/foo, the repo root path is /somepath
// if run pscm-build on /somepath/foo/bar, the repo root path is /somepath/foo/bar
std::optional<std::string> find_repo() {
  auto repo_path = fs::current_path();
  std::string repo_path_str = repo_path.c_str();
  while (repo_path_str != "/") {
    auto filename = repo_path_str + "/repo.pscm";
    if (fs::exists(filename)) {
      return repo_path_str;
    }
    repo_path = repo_path.parent_path();
    repo_path_str = repo_path.c_str();
  }
  return std::nullopt;
}

int build_target(std::string_view repo_path, Label label) {
  std::string filename = fmt::format("{}/{}/build.pscm", repo_path, label.package());
  if (!fs::exists(filename)) {
    PSCM_ERROR("no such file: {}", filename);
    std::exit(1);
  }
  PSCM_INFO("load {}", filename);
  std::fstream ifs;
  ifs.open(filename, std::ios::in);
  if (!ifs.is_open()) {
    PSCM_ERROR("open {} failed", filename);
    return 2;
  }

  ifs.seekg(0, ifs.end);
  auto sz = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  std::string code;
  code.resize(sz);
  ifs.read((char *)code.data(), sz);
  std::unordered_map<pscm::build::Label, Rule *> rule_map;
  using namespace pscm;
  try {
    RuleContext ctx(repo_path, label.repo(), label.package());
    Parser parser(code, filename);
    Cell expr = parser.next();
    while (!expr.is_none()) {
      if (expr.is_pair() && car(expr).is_sym()) {
        auto rule_name = car(expr);
        if (rule_name == "cpp_library"_sym) {
          auto rule = _cpp_library_impl(ctx, cdr(expr));
          PSCM_INFO("parse {} {}", rule_name, rule->name());
          rule_map[pscm::build::Label(label.repo(), label.package(), rule->name())] = rule;
        }
        else if (rule_name == "cpp_binary"_sym || rule_name == "cpp_test"_sym) {
          auto rule = _cpp_binary_impl(ctx, cdr(expr));
          PSCM_INFO("parse {} {}", rule_name, rule->name());
          rule_map[pscm::build::Label(label.repo(), label.package(), rule->name())] = rule;
        }
        else {
          PSCM_INFO("rule {} not supported now", rule_name);
        }
      }
      expr = parser.next();
    }
    RuleRunner runner(repo_path, rule_map);
    runner.run(label);
  }

  catch (Exception& ex) {
    PSCM_ERROR("load file {} error", filename);
  }
  return 0;
}

void show_usage() {
  std::cout << R"(
Usage: pscm-build <command> <options> ...

Availabel commands:
  build                Builds the specified targets.
  clean                Removes output files.
)";
}

export int main(int argc, char **argv) {
  pscm::logger::Logger::root_logger()->add_appender(new pscm::logger::ConsoleAppender());
  auto repo_path = find_repo();
  if (!repo_path.has_value()) {
    PSCM_ERROR("repo.pscm not found");
    std::exit(1);
  }
  PSCM_INFO("find repo: {}", repo_path);
  std::string target;
  if (argc >= 2) {
    if (argv[1] == "clean"s) {
      auto build_path = repo_path.value() + "/pscm-build-bin"s;
      PSCM_INFO("clean build directory: {}", build_path);
      fs::remove_all(build_path);
      return 0;
    }
    else if (argv[1] == "build"s) {
      if (argc <= 2) {
        PSCM_ERROR("incomplete subcommand build, a target must be specified!");
        std::exit(1);
      }
      int arg_idx = 2;
      std::optional<Label> label;
      while (arg_idx < argc) {
        if (std::string_view(argv[arg_idx]) == "-s") {
          arg_idx++;
          pscm::logger::Logger::get_logger("pscm.build.Action")->set_level(pscm::logger::Logger::Level::DEBUG_);
        }
        else {
          target = argv[arg_idx];
          label = Label::parse(target);
          if (!label.has_value()) {
            PSCM_ERROR("bad target: {}", target);
            return -1;
          }
          std::string package;
          if (label->package().empty()) {
            std::string filename = fs::current_path().c_str() + "/build.pscm"s;
            bool ok = fs::exists(filename);
            if (!ok) {
              PSCM_ERROR("build.pscm is required!!!");
              return 1;
            }
            package = std::string(fs::path(filename).parent_path().c_str()).substr(repo_path->size());
            PSCM_INFO("package: {}", package);
            label = Label(label->repo(), package, label->name());
          }
          arg_idx++;
        }
      }
      auto ret = build_target(repo_path.value(), label.value());
      if (ret) {
        return ret;
      }
      PSCM_INFO("Done!!!");
      return 0;
    }
    else {
      PSCM_ERROR("invalid subcommand: {}, only support clean, build now", argv[1]);
      show_usage();
      return 0;
    }
  }
  else {
    PSCM_ERROR("no subcommand found");
    show_usage();
    return 0;
  }
}
