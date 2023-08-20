//
// Created by PikachuHy on 2023/7/22.
//

export module pscm.build:DepsScanner;
import :BuildVariables;
import :CompilationContext;
import std;

export namespace pscm::build {
class ModuleDep {
public:
  void add_require_module(std::string module_name) {
    requires_.push_back(module_name);
  }

  [[nodiscard]] auto require_modules() -> const std::vector<std::string>& {
    return requires_;
  }

  [[nodiscard]] auto is_module_interface() const -> bool {
    return name_.has_value();
  }

  void set_provide_module(std::string name) {
    name_ = name;
  }

  [[nodiscard]] auto provide_module() const -> const std::optional<std::string>& {
    return name_;
  }

private:
  std::optional<std::string> name_;
  std::vector<std::string> requires_;
};

class DepsScanner {
public:
  DepsScanner(CompilationContext *ctx)
      : ctx_(ctx){};
  auto scan(CompileBuildVariables variables, std::string_view cwd) -> ModuleDep;

private:
  CompilationContext *ctx_;
};
} // namespace pscm::build