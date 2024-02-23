#include "Procedure.h"
#include <sstream>

namespace pscm::core {
std::string Procedure::to_string() const {
  std::stringstream ss;
  ss << "#";
  ss << "<procedure ";
  if (name_) {
    ss << name_->to_string();
  }
  else {
    ss << "#f";
  }
  ss << " ";
  ss << "(";
  for (auto arg : args_) {
    ss << arg->to_string();
    ss << " ";
  }
  if (vararg_.has_value()) {
    ss << ".";
    ss << " ";
    ss << vararg_.value()->to_string();
  }
  else {
    ss.seekp(-1, std::ios::cur);
  }
  ss << ")";
  ss << ">";
  return ss.str();
}

llvm::Value *Procedure::codegen(CodegenContext& ctx) {
  return nullptr;
}

const Type *Procedure::type() const {
  return nullptr;
}

} // namespace pscm::core
