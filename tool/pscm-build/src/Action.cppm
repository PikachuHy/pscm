module;
#include <pscm/common_def.h>
export module pscm.build:Action;
import std;
import pscm;
import fmt;
import :CompilationContext;
import :LinkingContext;
import :BuildVariables;
import :CppToolchain;
import :Expander;
PSCM_INLINE_LOG_DECLARE("pscm.build.Action");

export namespace pscm::build {

class Action {
public:
  void run(std::string_view cwd);
  virtual auto get_args() const -> std::vector<std::string> = 0;
  virtual auto get_output_file() const -> std::optional<std::string> = 0;
};

class CppAction : public Action {
public:
  CppAction(const CppToolchain *toolchain)
      : toolchain_(toolchain) {
  }

protected:
  const CppToolchain *toolchain_;
};

class CppCompileAction : public CppAction {
public:
  CppCompileAction(CompilationContext *ctx, CompileBuildVariables var, const CppToolchain *toolchain)
      : CppAction(toolchain)
      , ctx_(ctx)
      , var_(var) {
  }

  auto get_output_file() const -> std::optional<std::string> override {
    return var_.get_output_file();
  }

protected:
  CompilationContext *ctx_;
  CompileBuildVariables var_;
};

class CppModuleInterfaceCompileAction : public CppCompileAction {
public:
  CppModuleInterfaceCompileAction(CompilationContext *ctx, CompileBuildVariables var, const CppToolchain *toolchain)
      : CppCompileAction(ctx, var, toolchain) {
  }

  auto get_args() const -> std::vector<std::string> override {
    PSCM_ASSERT(ctx_);
    PSCM_ASSERT(toolchain_);
    std::vector<std::string> ret;
    ret.push_back("ccache");
    ret.push_back(toolchain_->cc());
    ret.push_back("--precompile");
    ret.push_back("-x");
    ret.push_back("c++-module");
    Expander expander(ret);
    expander.expand_includes(ctx_->includes());
    expander.expand_includes(var_.get_includes());
    expander.expand_defines(ctx_->defines());
    expander.expand_defines(var_.get_defines());
    expander.expand_opts(var_.get_opts());
    expander.expand_modules(var_.modules());
    expander.expand_source_file(var_.get_source_file());
    expander.expand_output_file(var_.get_output_file());
    return ret;
  }
};

class CppModuleInterfaceCodegenAction : public CppCompileAction {
public:
  CppModuleInterfaceCodegenAction(CompilationContext *ctx, CompileBuildVariables var, const CppToolchain *toolchain)
      : CppCompileAction(ctx, var, toolchain) {
  }

  auto get_args() const -> std::vector<std::string> override {
    std::vector<std::string> ret;
    ret.push_back("ccache");
    ret.push_back(toolchain_->cc());
    Expander expander(ret);
    expander.expand_defines(ctx_->defines());
    expander.expand_defines(var_.get_defines());
    expander.expand_opts(var_.get_opts());
    expander.expand_modules(var_.modules());
    expander.expand_source_file(var_.get_source_file());
    expander.expand_output_file(var_.get_output_file());
    return ret;
  }
};

class CppModuleImplementationCompileActon : public CppCompileAction {
public:
  CppModuleImplementationCompileActon(CompilationContext *ctx, CompileBuildVariables var, const CppToolchain *toolchain)
      : CppCompileAction(ctx, var, toolchain) {
  }

  auto get_args() const -> std::vector<std::string> override {
    std::vector<std::string> ret;
    ret.push_back("ccache");
    ret.push_back(toolchain_->cc());
    Expander expander(ret);
    expander.expand_includes(ctx_->includes());
    expander.expand_includes(var_.get_includes());
    expander.expand_defines(ctx_->defines());
    expander.expand_defines(var_.get_defines());
    expander.expand_opts(var_.get_opts());
    expander.expand_modules(var_.modules());
    expander.expand_source_file(var_.get_source_file());
    expander.expand_output_file(var_.get_output_file());
    return ret;
  }
};

class CppLinkAction : public CppAction {
public:
  CppLinkAction(LinkingContext *ctx, LinkBuildVariables var, const CppToolchain *toolchain)
      : CppAction(toolchain)
      , ctx_(ctx)
      , var_(var) {
  }

  auto get_output_file() const -> std::optional<std::string> override {
    return var_.get_output_file();
  }

protected:
  LinkingContext *ctx_;
  LinkBuildVariables var_;
};

class CppLinkStaticAction : public CppLinkAction {
public:
  CppLinkStaticAction(LinkingContext *ctx, LinkBuildVariables var, const CppToolchain *toolchain)
      : CppLinkAction(ctx, var, toolchain) {
  }

  auto get_args() const -> std::vector<std::string> override {
    std::vector<std::string> ret;
    ret.push_back("ccache");
    ret.push_back(toolchain_->ar());
    ret.push_back("rcsD");
    ret.push_back(var_.get_output_file());
    for (const auto& arg : var_.get_object_files()) {
      ret.push_back(arg);
    }
    for (const auto& arg : var_.get_linkopts()) {
      ret.push_back(arg);
    }
    return ret;
  }
};

class CppLinkBinaryAction : public CppLinkAction {
public:
  CppLinkBinaryAction(LinkingContext *ctx, LinkBuildVariables var, const CppToolchain *toolchain)
      : CppLinkAction(ctx, var, toolchain) {
  }

  auto get_args() const -> std::vector<std::string> override {
    std::vector<std::string> ret;
    ret.push_back("ccache");
    ret.push_back(toolchain_->cc());
    ret.push_back("-o");
    ret.push_back(var_.get_output_file());
    for (const auto& arg : var_.get_object_files()) {
      ret.push_back(arg);
    }
    for (const auto& arg : ctx_->libraries()) {
      ret.push_back(arg);
    }
    for (const auto& arg : var_.get_linkopts()) {
      ret.push_back(arg);
    }
    ret.push_back("-fuse-ld=lld");
    ret.push_back("-lc++");
    return ret;
  }
};
} // namespace pscm::build
