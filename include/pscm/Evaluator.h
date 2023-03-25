//
// Created by PikachuHy on 2023/3/4.
//

#pragma once
#include "pscm/Cell.h"
#include <vector>

namespace pscm {
Cell add(Cell args);
Cell minus(Cell args);
Cell mul(Cell args);
Cell div(Cell args);
Cell less_than(Cell args);
Cell equal_to(Cell args);
Cell greater_than(Cell args);
Cell is_negative(Cell args);
Cell builtin_not(Cell args);
Cell display(Cell args);
Cell newline(Cell args);
Cell is_procedure(Cell args);
Cell is_boolean(Cell args);
class SymbolTable;

class Evaluator {
public:
  Evaluator();
  Cell eval(Cell expr, SymbolTable *env);

  struct Register {
    Cell expr;        // expression to evaluated
    SymbolTable *env; // evaluation environment
    Cell proc;        // procedure to be applied
    Cell argl;        // list of evaluated arguments
    Label cont;       // place to go to next
    Cell val;         // result of evaluation
    Cell unev;        // temporary register for expressions
    std::string to_string() const;
  };

  enum RegisterType { reg_expr, reg_env, reg_proc, reg_argl, reg_cont, reg_val, reg_unev };

  struct Stack {
    std::vector<Cell> expr;
    std::vector<SymbolTable *> env;
    std::vector<Cell> proc;
    std::vector<Cell> argl;
    std::vector<Label> cont;
    std::vector<Cell> val;
    std::vector<Cell> unev;
    bool empty() const;
    std::string to_string() const;
  };

private:
  void run();

private:
  friend std::ostream& operator<<(std::ostream& out, const Register& reg);
  friend std::ostream& operator<<(std::ostream& out, const RegisterType& reg);
  friend std::ostream& operator<<(std::ostream& out, const Stack& stack);
  Register reg_;
  Stack stack_;
  Label pos_;
  std::vector<RegisterType> reg_type_stack_;
  std::size_t step_ = 0;
};
} // namespace pscm
