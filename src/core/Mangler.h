#pragma once
#include <string>

namespace pscm::core {
class ExprAST;
class Type;

class Mangler {
public:
  [[nodiscard]] std::string mangle(const std::string& callee, const std::vector<const Type *>& args) const;
};

} // namespace pscm::core
