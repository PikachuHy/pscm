#pragma once

namespace pscm::core {
class Value;
class SymbolValue;
class AST;
class ExprAST;
class Type;
class EvaluatorImpl;

class Evaluator {
public:
  Evaluator();
  ~Evaluator();
  AST *eval(Value *expr);
  void add_proc(SymbolValue *sym, ExprAST *value);
  void push_symbol_table();
  void pop_symbol_table();
  void add_sym(SymbolValue *sym, ExprAST *value);

private:
  EvaluatorImpl *impl_;
};

} // namespace pscm::core
