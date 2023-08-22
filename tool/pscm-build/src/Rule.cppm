//
// Created by PikachuHy on 2023/7/22.
//
module;
#include <pscm/common_def.h>
export module pscm.build:Rule;
import std;
import pscm;
import :Artifact;
import :CppToolchain;
import :CompilationContext;
import :LinkingContext;
import :Label;
import :RuleContext;

export namespace pscm::build {
class CppHelper;

class Rule {
public:
  virtual void parse(RuleContext ctx, Cell args) = 0;
  virtual Artifact *run(std::string_view repo_path, std::string_view package,
                        const std::unordered_set<Artifact *>& depset) = 0;

  const Label& label() const {
    return label_;
  }

  const std::string& name() const {
    return name_;
  }

  const std::vector<Label>& deps() const {
    return deps_;
  }

protected:
  Label label_;
  std::string name_;
  // build.pscm relative to repo.pscm
  std::string package_;
  std::vector<Label> deps_;
};

class CppRuleBase : public Rule {
public:
  static const CppToolchain& toolchain() {
    static CppToolchain cpp_toolchain = init_cpp_toolchain();
    return cpp_toolchain;
  }

  void parse(RuleContext ctx, Cell args) override;

  auto init_compilation_context(const std::unordered_set<Artifact *>& depset) -> CompilationContext {
    CompilationContext ctx;
    for (auto dep : depset) {
      if (dep->is_cpp_library_artifact()) {
        auto cpp_lib_dep = static_cast<CppLibraryArtifact *>(dep);
        ctx.merge(cpp_lib_dep->compilation_context());
      }
    }
    return ctx;
  }

  auto init_linking_context(const std::unordered_set<Artifact *>& depset) -> LinkingContext {
    LinkingContext ctx;
    for (auto dep : depset) {
      if (dep->is_cpp_library_artifact()) {
        auto cpp_lib_dep = static_cast<CppLibraryArtifact *>(dep);
        ctx.merge(cpp_lib_dep->linking_context());
      }
    }
    return ctx;
  }

protected:
  virtual void parse_attr(RuleContext ctx, Cell args);

protected:
  std::vector<std::string> srcs_;
  std::vector<std::string> copts_;
};

class CppLibraryRule : public CppRuleBase {
public:
  Artifact *run(std::string_view repo_path, std::string_view package,
                const std::unordered_set<Artifact *>& depset) override;

protected:
  void parse_attr(RuleContext ctx, Cell args) override;

private:
  std::vector<std::string> hdrs_;
  std::vector<std::string> defines_;
  std::vector<std::string> includes_;
  friend class CppHelper;
};

class CppBinaryRule : public CppRuleBase {
public:
  Artifact *run(std::string_view, std::string_view, const std::unordered_set<Artifact *>& depset) override;

protected:
  void parse_attr(RuleContext ctx, Cell args) override;

private:
  friend class CppHelper;
};

Rule *_cpp_library_impl(RuleContext ctx, Cell args) {
  auto rule = new CppLibraryRule();
  rule->parse(ctx, args);
  return rule;
}

Rule *_cpp_binary_impl(RuleContext ctx, Cell args) {
  auto rule = new CppBinaryRule();
  rule->parse(ctx, args);
  return rule;
}

class RuleRunner {
public:
  RuleRunner(std::string_view repo_path, std::unordered_map<Label, Rule *> rule_map)
      : repo_path_(repo_path)
      , rule_map_(rule_map) {
  }

  void run(Label label);

  void run_rule(Rule *rule, std::string_view package);

  void collect_dep(std::unordered_set<Artifact *>& depset, Rule *rule) {
    for (const auto& dep : rule->deps()) {
      auto r = rule_map_.at(dep);
      collect_dep(depset, r);
      auto artifact = artifact_map_.at(dep);
      depset.insert(artifact);
    }
  }

private:
  std::string_view repo_path_;
  std::unordered_map<Label, Rule *> rule_map_;
  std::unordered_map<Label, Artifact *> artifact_map_;
};

} // namespace pscm::build