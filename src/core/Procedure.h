#pragma once

#include "Value.h"
#include <optional>
#include <vector>

namespace pscm::core {
class Value;
class SymbolValue;
class SymbolTable;

class Procedure
    : public Value
    , public ExprAST {
public:
  Procedure(SymbolValue *name, std::vector<SymbolValue *> args, std::vector<Value *> body, SymbolTable *env,
            std::optional<SymbolValue *> vararg = std::nullopt)
      : name_(name)
      , args_(std::move(args))
      , body_(std::move(body))
      , env_(env)
      , vararg_(std::move(vararg)) {
  }

  [[nodiscard]] SymbolValue *name() const {
    return name_;
  }

  void set_name(SymbolValue *name) {
    this->name_ = name;
  }

  [[nodiscard]] const std::vector<SymbolValue *>& args() const {
    return args_;
  }

  [[nodiscard]] const std::vector<Value *>& body() const {
    return body_;
  }

  [[nodiscard]] std::string to_string() const override;
  llvm::Value *codegen(CodegenContext& ctx) override;
  const Type *type() const override;

private:
  SymbolValue *name_;
  std::vector<SymbolValue *> args_;
  std::vector<Value *> body_;
  SymbolTable *env_;
  std::optional<SymbolValue *> vararg_;
};

} // namespace pscm::core
