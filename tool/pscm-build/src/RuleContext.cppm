
export module pscm.build:RuleContext;
import std;

export namespace pscm::build {
class RuleContext {
public:
  RuleContext(std::string_view repo_path, std::string_view repo, std::string_view package)
      : repo_path_(repo_path)
      , repo_(repo)
      , package_(package) {
  }

  [[nodiscard]] auto repo_path() const -> std::string_view {
    return repo_path_;
  }

  [[nodiscard]] auto repo() const -> std::string_view {
    return repo_;
  }

  [[nodiscard]] auto package() const -> std::string_view {
    return package_;
  }

private:
  std::string_view repo_path_;
  std::string_view repo_;
  std::string_view package_;
};
} // namespace pscm::build
