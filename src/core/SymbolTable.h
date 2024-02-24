#pragma once
#include "Value.h"
#include <memory>

namespace pscm::core {
class Value;
class ExprAST;
class SymbolValue;
class SymbolTableImpl;

class SymbolTable {
public:
  explicit SymbolTable(SymbolTable *parent = nullptr);
  ~SymbolTable();
  [[nodiscard]] ExprAST *lookup(SymbolValue *sym) const;
  void put(SymbolValue *sym, ExprAST *value);
  [[nodiscard]] SymbolTable *parent() const;

private:
  SymbolTableImpl *impl_;
};

} // namespace pscm::core
