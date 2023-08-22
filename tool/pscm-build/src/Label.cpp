module;
#include <pscm/Logger.h>
module pscm.build;
import std;
import fmt;
import :Label;

namespace pscm::build {
PSCM_INLINE_LOG_DECLARE("pscm.build");

std::optional<Label> Label::parse(std::string_view s) {
  std::string_view repo;
  std::string_view package;
  std::string_view name;
  if (s.starts_with("@")) {
    auto idx = s.find("//");
    if (idx == std::string_view::npos) {
      PSCM_ERROR("invalid repo name: {}", s);
      return std::nullopt;
    }
    repo = s.substr(1, idx - 1);
    s = s.substr(idx);
  }
  if (s.starts_with("//")) {
    auto idx = s.find(":");
    if (idx == std::string_view::npos) {
      if (s.substr(1).empty()) {
        PSCM_ERROR("invalid package: {}", s);
        return std::nullopt;
      }
      idx = s.find_last_of("/");
      package = s.substr(1);
      name = s.substr(idx);
    }
    else {
      package = s.substr(1, idx - 1);
      name = s.substr(idx + 1);
    }
    return Label(repo, package, name);
  }
  if (s.starts_with(":")) {
    name = s.substr(1);
    return Label(repo, package, name);
  }
  if (s == "...") {
    return Label(repo, package, s);
  }
  PSCM_ERROR("invalid target: {}", s);
  return std::nullopt;
}

} // namespace pscm::build
