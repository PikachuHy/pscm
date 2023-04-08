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
Cell less_or_equal_than(Cell args);
Cell equal_to(Cell args);
Cell greater_than(Cell args);
Cell greater_or_equal_than(Cell args);
Cell is_negative(Cell args);
Cell builtin_not(Cell args);
Cell display(Cell args);
Cell newline(Cell args);
Cell is_procedure(Cell args);
Cell is_boolean(Cell args);
Cell create_list(Cell args);
Cell is_list(Cell args);
Cell is_pair(Cell args);
Cell set_car(Cell args);
Cell set_cdr(Cell args);
Cell proc_cons(Cell args);
Cell proc_car(Cell args);
Cell proc_cdr(Cell args);
Cell proc_cdar(Cell args);
Cell proc_cadr(Cell args);
Cell proc_cddr(Cell args);
Cell is_eqv(Cell args);
Cell is_eq(Cell args);
Cell is_equal(Cell args);
Cell memq(Cell args);
Cell memv(Cell args);
Cell member(Cell args);
Cell is_equal(Cell args);
Cell assq(Cell args);
Cell assv(Cell args);
Cell assoc(Cell args);
Cell make_vector(Cell args);
Cell proc_vector(Cell args);
Cell vector_set(Cell args);
Cell is_zero(Cell args);
Cell is_null(Cell args);
Cell length(Cell args);
Cell append(Cell args);
Cell reverse(Cell args);
Cell list_ref(Cell args);
Cell expt(Cell args);
Cell proc_abs(Cell args);
Cell proc_sqrt(Cell args);
Cell proc_round(Cell args);
Cell inexact_to_exact(Cell args);
Cell is_symbol(Cell args);
Cell symbol_to_string(Cell args);
Cell string_to_symbol(Cell args);
Cell is_string_equal(Cell args);

class SymbolTable;
class Scheme;

class Evaluator {
public:
  Evaluator(Scheme& scm);
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
  Scheme& scm_;
};
} // namespace pscm
