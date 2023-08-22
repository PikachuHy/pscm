//
// Created by PikachuHy on 2023/7/22.
//

export module pscm.build:BuildVariables;
import std;
import :CompilationContext;

export namespace pscm::build {
class CompileBuildVariables {
public:
  void set_source_file(std::string source_file) {
    this->source_file_ = std::move(source_file);
  }

  void set_output_file(std::string output_file) {
    this->output_file_ = std::move(output_file);
  }

  [[nodiscard]] auto get_source_file() const -> const std::string& {
    return source_file_;
  }

  [[nodiscard]] auto get_output_file() const -> const std::string& {
    return output_file_;
  }

  void set_opts(std::vector<std::string> opts) {
    this->opts_ = std::move(opts);
  }

  [[nodiscard]] auto get_opts() const -> const std::vector<std::string>& {
    return opts_;
  }

  void set_includes(std::vector<std::string> includes) {
    this->includes_ = std::move(includes);
  }

  [[nodiscard]] auto get_includes() const -> const std::vector<std::string>& {
    return includes_;
  }

  void set_defines(std::vector<std::string> defines) {
    this->defines_ = std::move(defines);
  }

  [[nodiscard]] auto get_defines() const -> const std::vector<std::string>& {
    return defines_;
  }

  void set_modules(std::vector<ModuleInfo> modules) {
    this->modules_ = std::move(modules);
  }

  [[nodiscard]] const std::vector<ModuleInfo> modules() const {
    return modules_;
  }

private:
  std::string source_file_;
  std::string output_file_;
  std::vector<std::string> opts_;
  std::vector<std::string> includes_;
  std::vector<std::string> defines_;
  std::vector<ModuleInfo> modules_;
};

class LinkBuildVariables {
public:
  void set_output_file(std::string output_file) {
    this->output_file_ = std::move(output_file);
  }

  [[nodiscard]] auto get_output_file() const -> const std::string& {
    return output_file_;
  }

  void set_linkopts(std::vector<std::string> linkopts) {
    this->linkopts_ = std::move(linkopts);
  }

  [[nodiscard]] auto get_linkopts() const -> const std::vector<std::string>& {
    return linkopts_;
  }

  void set_object_files(std::vector<std::string> object_files) {
    this->object_files_ = std::move(object_files);
  }

  [[nodiscard]] auto get_object_files() const -> const std::vector<std::string>& {
    return object_files_;
  }

private:
  std::string output_file_;
  std::vector<std::string> linkopts_;
  std::vector<std::string> object_files_;
};
} // namespace pscm::build