export module pscm.build:CompilationContext;
import std;

export namespace pscm::build {
struct ModuleInfo {
  std::string name;
  std::string bmi;
};

class CompilationContext {
public:
  class Builder {
  public:
    void add_define(const std::string& arg) {
      defines_.insert(arg);
    }

    void add_defines(const std::vector<std::string>& args) {
      defines_.insert(args.begin(), args.end());
    }

    void add_include(const std::string& arg) {
      includes_.insert(arg);
    }

    void add_includes(const std::vector<std::string>& args) {
      includes_.insert(args.begin(), args.end());
    }

    void add_module(ModuleInfo *arg) {
      modules_.insert(arg);
    }

    void add_modules(const std::vector<ModuleInfo *>& args) {
      modules_.insert(args.begin(), args.end());
    }

    CompilationContext build() {
      CompilationContext ctx{};
      ctx.defines_ = std::move(defines_);
      ctx.includes_ = std::move(includes_);
      ctx.modules_ = std::move(modules_);
      return ctx;
    }

  private:
    std::unordered_set<std::string> defines_;
    std::unordered_set<std::string> includes_;
    std::unordered_set<ModuleInfo *> modules_;
  };

public:
  auto defines() const -> const std::unordered_set<std::string>& {
    return defines_;
  }

  auto includes() const -> const std::unordered_set<std::string>& {
    return includes_;
  }

  auto modules() const -> const std::unordered_set<ModuleInfo *>& {
    return modules_;
  }

  void merge(const CompilationContext& ctx) {
    defines_.insert(ctx.defines().begin(), ctx.defines().end());
    includes_.insert(ctx.includes().begin(), ctx.includes().end());
    modules_.insert(ctx.modules().begin(), ctx.modules().end());
  }

  auto get_module_name_map() const -> std::unordered_map<std::string, ModuleInfo *> {
    std::unordered_map<std::string, ModuleInfo *> ret;
    for (auto m : modules_) {
      ret[m->name] = m;
    }
    return ret;
  }

private:
  std::unordered_set<std::string> defines_;
  std::unordered_set<std::string> includes_;
  std::unordered_set<ModuleInfo *> modules_;
};

} // namespace pscm::build
