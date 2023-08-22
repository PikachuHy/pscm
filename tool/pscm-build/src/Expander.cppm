export module pscm.build:Expander;
import std;
import fmt;
import :CompilationContext;

export namespace pscm::build {
class Expander {
public:
  Expander(std::vector<std::string>& args)
      : args_(args) {
  }

  void expand_source_file(std::string_view source_file) const {
    args_.push_back("-c");
    args_.push_back(std::string(source_file));
  }

  void expand_output_file(std::string_view output_file) const {
    args_.push_back("-o");
    args_.push_back(std::string(output_file));
  }

  void expand_includes(auto includes) {
    for (auto arg : includes) {
      args_.push_back(fmt::format("-I{}", arg));
    }
  }

  void expand_defines(auto defines) {
    for (auto arg : defines) {
      args_.push_back(fmt::format("-D{}", arg));
    }
  }

  void expand_opts(auto opts) {
    for (auto arg : opts) {
      args_.push_back(arg);
    }
  }

  void expand_modules(const std::vector<ModuleInfo>& modules) {
    for (const auto& module_info : modules) {
      args_.push_back(fmt::format("-fmodule-file={}={}", module_info.name, module_info.bmi));
    }
  }

private:
  std::vector<std::string>& args_;
};
} // namespace pscm::build
