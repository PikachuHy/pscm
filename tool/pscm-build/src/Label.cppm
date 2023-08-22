module;
#include <pscm/Logger.h>
export module pscm.build:Label;
import std;
import fmt;

export namespace pscm::build {
class Label {
public:
  // label
  // @repo_name//package1/package2:foo
  // //package1/foo
  // //package1/package2:foo
  // :foo
  static std::optional<Label> parse(std::string_view s);

  Label() {
  }

  Label(std::string_view repo, std::string_view package, std::string_view name)
      : repo_(repo)
      , package_(package)
      , name_(name) {
  }

  auto repo() const -> std::string_view {
    return repo_;
  }

  auto package() const -> std::string_view {
    return package_;
  }

  auto name() const -> std::string_view {
    return name_;
  }

  friend bool operator==(const Label& lhs, const Label& rhs) {
    return lhs.repo() == rhs.repo() && lhs.package() == rhs.package() && lhs.name() == rhs.name();
  }

  auto to_string() const -> std::string {
    return fmt::format("@{}/{}:{}", repo_, package_, name_);
  }

private:
  std::string repo_;
  std::string package_;
  std::string name_;
};

} // namespace pscm::build

export namespace std {
template <>
struct hash<pscm::build::Label> {
  using result_type = std::size_t;
  using argument_type = pscm::build::Label;

  std::size_t operator()(const pscm::build::Label& label) const {
    return hash<std::string_view>()(label.name());
  }
};
} // namespace std