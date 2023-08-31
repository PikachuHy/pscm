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
Cell is_positive(Cell args);
Cell is_negative(Cell args);
Cell is_odd(Cell args);
Cell is_even(Cell args);
Cell proc_max(Cell args);
Cell proc_min(Cell args);
Cell quotient(Cell args);
Cell remainder(Cell args);
Cell modulo(Cell args);
Cell proc_gcd(Cell args);
Cell proc_lcm(Cell args);
Cell builtin_not(Cell args);
Cell write(Cell args);
Cell display(Cell args);
Cell newline(Cell args);
Cell is_procedure(Cell args);
Cell is_boolean(Cell args);
Cell is_vector(Cell args);
Cell make_vector(Cell args);
Cell proc_vector(Cell args);
Cell vector_length(Cell args);
Cell vector_ref(Cell args);
Cell vector_set(Cell args);
Cell vector_to_list(Cell args);
Cell list_to_vector(Cell args);
Cell vector_fill(Cell args);
Cell is_zero(Cell args);
Cell proc_acos(Cell args);
Cell expt(Cell args);
Cell proc_abs(Cell args);
Cell proc_sqrt(Cell args);
Cell proc_round(Cell args);
Cell is_exact(Cell args);
Cell is_inexact(Cell args);
Cell inexact_to_exact(Cell args);
Cell is_symbol(Cell args);
Cell symbol_to_string(Cell args);
Cell string_to_symbol(Cell args);
Cell is_string(Cell args);
Cell make_string(Cell args);
Cell proc_string(Cell args);
Cell string_length(Cell args);
Cell string_ref(Cell args);
Cell string_set(Cell args);
Cell is_string_equal(Cell args);
Cell is_string_equal_case_insensitive(Cell args);
Cell is_string_less(Cell args);
Cell is_string_greater(Cell args);
Cell is_string_less_or_equal(Cell args);
Cell is_string_greater_or_equal(Cell args);
Cell is_string_less_case_insensitive(Cell args);
Cell is_string_greater_case_insensitive(Cell args);
Cell is_string_less_or_equal_case_insensitive(Cell args);
Cell is_string_greater_or_equal_case_insensitive(Cell args);
Cell proc_substring(Cell args);
Cell string_append(Cell args);
Cell string_to_list(Cell args);
Cell list_to_string(Cell args);
Cell string_copy(Cell args);
Cell string_fill(Cell args);
Cell is_number(Cell args);
Cell is_complex(Cell args);
Cell is_real(Cell args);
Cell is_integer(Cell args);
Cell is_rational(Cell args);
Cell string_to_number(Cell args);
Cell number_to_string(Cell args);
Cell is_char(Cell args);
Cell is_char_equal(Cell args);
Cell is_char_less(Cell args);
Cell is_char_greater(Cell args);
Cell is_char_less_or_equal(Cell args);
Cell is_char_greater_or_equal(Cell args);
Cell is_char_equal_case_insensitive(Cell args);
Cell is_char_less_case_insensitive(Cell args);
Cell is_char_greater_case_insensitive(Cell args);
Cell is_char_less_or_equal_case_insensitive(Cell args);
Cell is_char_greater_or_equal_case_insensitive(Cell args);
Cell is_char_alphabetic(Cell args);
Cell is_char_numeric(Cell args);
Cell is_char_whitespace(Cell args);
Cell is_char_upper_case(Cell args);
Cell is_char_lower_case(Cell args);
Cell char_to_integer(Cell args);
Cell integer_to_char(Cell args);
Cell char_upcase(Cell args);
Cell char_downcase(Cell args);
Cell call_with_input_file(Cell args);
Cell call_with_output_file(Cell args);
Cell is_input_port(Cell args);
Cell is_output_port(Cell args);
Cell current_input_port(Cell args);
Cell current_output_port(Cell args);
Cell open_input_file(Cell args);
Cell open_output_file(Cell args);
Cell close_input_port(Cell args);
Cell close_output_port(Cell args);
Cell proc_read(Cell args);
Cell read_char(Cell args);
Cell peek_char(Cell args);
Cell is_eof_object(Cell args);
Cell is_char_ready(Cell args);
Cell write_char(Cell args);
Cell load(Cell args);
Cell transcript_on(Cell args);
Cell transcript_off(Cell args);
Cell proc_exit(Cell args);

class SymbolTable;
class Scheme;

class Evaluator {
public:
  Evaluator(Scheme& scm, Evaluator *parent = nullptr);
  Cell eval(Cell expr, SymbolTable *env);

  struct Register {
    Cell expr;        // expression to evaluated
    SymbolTable *env; // evaluation environment
    Cell proc;        // procedure to be applied
    Cell argl;        // list of evaluated arguments
    Label cont;       // place to go to next
    Cell val;         // result of evaluation
    Cell unev;        // temporary register for expressions
    UString to_string() const;
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
    UString to_string() const;
  };

private:
  void run();
  bool load(const UString& filename, SymbolTable *env);
  Label eval_map_expr(Label default_pos);

private:
  Register reg_;
  Stack stack_;
  Label pos_;
  std::vector<RegisterType> reg_type_stack_;
  std::size_t step_ = 0;
  Scheme& scm_;
  Evaluator *parent_;
};

UString to_string(pscm::Evaluator::RegisterType type);
} // namespace pscm
