
#include "SymbolTable.h"

#include <unordered_map>

namespace pscm::core {
class SymbolTableImpl {
public:
  explicit SymbolTableImpl(SymbolTable *parent)
      : parent_(parent) {
  }

  void put(SymbolValue *sym, ExprAST *value) {
    sym_table_.insert_or_assign(sym->to_string(), value);
  }

  [[nodiscard]] ExprAST *lookup(SymbolValue *sym) {
    auto it = sym_table_.find(sym->to_string());
    if (it != sym_table_.end()) {
      return it->second;
    }
    if (parent_) {
      return parent_->lookup(sym);
    }
    return nullptr;
  }

  SymbolTable *parent_;
  std::unordered_map<std::string, ExprAST *> sym_table_;
};

SymbolTable::SymbolTable(SymbolTable *parent)
    : impl_(new SymbolTableImpl(parent)) {
}

SymbolTable::~SymbolTable() {
  delete impl_;
}

ExprAST *SymbolTable::lookup(SymbolValue *sym) const {
  return impl_->lookup(sym);
}

void SymbolTable::put(SymbolValue *sym, ExprAST *value) {
  impl_->put(sym, value);
}

SymbolTable *SymbolTable::parent() const {
  return impl_->parent_;
}
} // namespace pscm::core
