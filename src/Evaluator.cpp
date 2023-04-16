//
// Created by PikachuHy on 2023/3/4.
//

#include "pscm/Evaluator.h"
#include "pscm/Char.h"
#include "pscm/Continuation.h"
#include "pscm/Exception.h"
#include "pscm/Expander.h"
#include "pscm/Function.h"
#include "pscm/Macro.h"
#include "pscm/Number.h"
#include "pscm/Pair.h"
#include "pscm/Parser.h"
#include "pscm/Port.h"
#include "pscm/Procedure.h"
#include "pscm/Promise.h"
#include "pscm/Scheme.h"
#include "pscm/Str.h"
#include "pscm/Symbol.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <fstream>
#include <ostream>
#include <sstream>
#include <unordered_set>
#define PSCM_PUSH_STACK(reg_name)                                                                                      \
  SPDLOG_DEBUG("push {} stack: {}", #reg_name, stack_.reg_name.size());                                                \
  reg_type_stack_.push_back(reg_##reg_name);                                                                           \
  stack_.reg_name.push_back(reg_.reg_name)

#define PSCM_POP_STACK(reg_name)                                                                                       \
  SPDLOG_DEBUG("pop {} stack: {}", #reg_name, stack_.reg_name.size());                                                 \
  PSCM_ASSERT(!reg_type_stack_.empty());                                                                               \
  if (reg_type_stack_.back() != reg_##reg_name) {                                                                      \
    std::stringstream ss1, ss2, ss3, ss4;                                                                              \
    ss1 << reg_##reg_name;                                                                                             \
    ss2 << reg_type_stack_.back();                                                                                     \
    SPDLOG_ERROR("reg stack error, expect '{}' but got '{}'", ss1.str(), ss2.str());                                   \
    for (int i = 0; i < reg_type_stack_.size(); i++) {                                                                 \
      ss3 << reg_type_stack_[reg_type_stack_.size() - i - 1];                                                          \
      ss3 << ", ";                                                                                                     \
    }                                                                                                                  \
    for (int i = 0; i < stack_.reg_name.size(); i++) {                                                                 \
      ss4 << stack_.reg_name[stack_.reg_name.size() - i - 1];                                                          \
      ss4 << ", ";                                                                                                     \
    }                                                                                                                  \
    SPDLOG_INFO("reg stack: {}", ss3.str());                                                                           \
    SPDLOG_INFO("{} reg stack: {}", #reg_name, ss4.str());                                                             \
    PSCM_ASSERT(reg_type_stack_.back() == reg_##reg_name);                                                             \
  }                                                                                                                    \
  reg_type_stack_.pop_back();                                                                                          \
  PSCM_ASSERT(!stack_.reg_name.empty());                                                                               \
  reg_.reg_name = stack_.reg_name.back();                                                                              \
  stack_.reg_name.pop_back()

#define GOTO(label)                                                                                                    \
  pos_ = label;                                                                                                        \
  SPDLOG_INFO("GOTO label: {}", pos_);                                                                                 \
  break

#define PRINT_STEP() SPDLOG_INFO("[step: {}] label: {}", step_, pos_)

template <>
class fmt::formatter<pscm::Label> {
public:
  auto parse(format_parse_context& ctx) {
    // PSCM_THROW_EXCEPTION("not supported now");
    auto i = ctx.begin();
    return i;
  }

  auto format(const pscm::Label& pos, auto& ctx) const {
    std::stringstream ss;
    ss << pos;
    return format_to(ctx.out(), "{}", ss.str());
  }
};

namespace pscm {
extern Cell scm_quasiquote(Scheme& scm, SymbolTable *env, Cell args);

Cell add(Cell args) {
  PSCM_ASSERT(args.is_pair() || args.is_nil());
  auto ret = new Number(0);
  for_each(
      [&ret](Cell expr, auto loc) {
        PSCM_ASSERT(expr.is_num());
        auto num = expr.to_number();
        ret->inplace_add(*num);
      },
      args);
  return ret;
}

Cell minus(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  if (arg.is_nil()) {
    PSCM_THROW_EXCEPTION("wrong-number-of-args: " + args.to_string());
  }
  auto first_val = arg;
  PSCM_ASSERT(first_val.is_num());
  auto first_num = first_val.to_number();
  PSCM_ASSERT(first_num);

  args = cdr(args);
  if (args.is_nil()) {
    auto ret = new Number(0);
    ret->inplace_minus(*first_num);
    return ret;
  }
  auto second_arg = car(args);
  Number *ret = nullptr;
  if (!second_arg.is_nil()) {
    auto val = second_arg;
    PSCM_ASSERT(val.is_num());
    auto num = val.to_number();
    PSCM_ASSERT(num);
    auto tmp = *first_num - *num;
    ret = new Number(tmp);
    args = cdr(args);
  }
  if (args.is_nil()) {
    return ret;
  }
  arg = car(args);
  while (!arg.is_nil()) {
    auto val = arg;
    PSCM_ASSERT(val.is_num());
    auto num = val.to_number();
    PSCM_ASSERT(num);
    ret->inplace_minus(*num);
    args = cdr(args);
    if (args.is_nil()) {
      break;
    }
    arg = car(args);
  }
  return ret;
}

Cell mul(Cell args) {
  auto ret = new Number(1);
  if (args.is_nil()) {
    return ret;
  }
  auto arg = car(args);
  while (!arg.is_nil()) {
    auto val = arg;
    PSCM_ASSERT(val.is_num());
    auto num = val.to_number();
    PSCM_ASSERT(num);
    ret->inplace_mul(*num);
    args = cdr(args);
    if (args.is_nil()) {
      break;
    }
    arg = car(args);
  }
  return ret;
}

Cell div(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  if (arg.is_nil()) {
    PSCM_THROW_EXCEPTION("wrong-number-of-args: " + args.to_string());
  }
  auto first_val = arg;
  PSCM_ASSERT(first_val.is_num());
  auto first_num = first_val.to_number();
  PSCM_ASSERT(first_num);

  args = cdr(args);
  if (args.is_nil()) {
    auto ret = new Number(1);
    ret->inplace_div(*first_num);
    return ret;
  }
  auto second_arg = car(args);
  Number *ret = nullptr;
  if (!second_arg.is_nil()) {
    auto val = second_arg;
    PSCM_ASSERT(val.is_num());
    auto num = val.to_number();
    PSCM_ASSERT(num);
    auto tmp = *first_num / *num;
    ret = new Number(tmp);
    args = cdr(args);
  }
  if (args.is_nil()) {
    return ret;
  }
  arg = car(args);
  while (!arg.is_nil()) {
    auto val = arg;
    PSCM_ASSERT(val.is_num());
    auto num = val.to_number();
    PSCM_ASSERT(num);
    ret->inplace_div(*num);
    args = cdr(args);
    if (args.is_nil()) {
      break;
    }
    arg = car(args);
  }
  return ret;
}

Cell less_than(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto a = car(args);
  auto b = cdr(args);
  if (b.is_nil()) {
    PSCM_THROW_EXCEPTION("wrong-number-of-args: " + args.to_string());
  }
  b = car(b);
  if (!a.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type argument in position 1: " + a.to_string());
  }
  if (!b.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type argument in position 2: " + b.to_string());
  }
  auto n1 = a.to_number();
  auto n2 = b.to_number();
  PSCM_ASSERT(n1);
  PSCM_ASSERT(n2);
  return (*n1 < *n2) ? Cell::bool_true() : Cell::bool_false();
}

Cell less_or_equal_than(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto a = car(args);
  auto b = cdr(args);
  if (b.is_nil()) {
    PSCM_THROW_EXCEPTION("wrong-number-of-args: " + args.to_string());
  }
  b = car(b);
  if (!a.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type argument in position 1: " + a.to_string());
  }
  if (!b.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type argument in position 2: " + b.to_string());
  }
  auto n1 = a.to_number();
  auto n2 = b.to_number();
  PSCM_ASSERT(n1);
  PSCM_ASSERT(n2);
  return (*n1 <= *n2) ? Cell::bool_true() : Cell::bool_false();
}

Cell equal_to(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto a = car(args);
  auto b = cdr(args);
  if (b.is_nil()) {
    PSCM_THROW_EXCEPTION("wrong-number-of-args: " + args.to_string());
  }
  b = car(b);
  if (!a.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type argument in position 1: " + a.to_string());
  }
  if (!b.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type argument in position 2: " + b.to_string());
  }
  auto n1 = a.to_number();
  auto n2 = b.to_number();
  PSCM_ASSERT(n1);
  PSCM_ASSERT(n2);
  return (*n1 == *n2) ? Cell::bool_true() : Cell::bool_false();
}

Cell greater_than(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto a = car(args);
  auto b = cdr(args);
  if (b.is_nil()) {
    PSCM_THROW_EXCEPTION("wrong-number-of-args: " + args.to_string());
  }
  b = car(b);
  if (!a.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type argument in position 1: " + a.to_string());
  }
  if (!b.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type argument in position 2: " + b.to_string());
  }
  auto n1 = a.to_number();
  auto n2 = b.to_number();
  PSCM_ASSERT(n1);
  PSCM_ASSERT(n2);
  return (*n1 > *n2) ? Cell::bool_true() : Cell::bool_false();
}

Cell greater_or_equal_than(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto a = car(args);
  auto b = cdr(args);
  if (b.is_nil()) {
    PSCM_THROW_EXCEPTION("wrong-number-of-args: " + args.to_string());
  }
  b = car(b);
  if (!a.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type argument in position 1: " + a.to_string());
  }
  if (!b.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type argument in position 2: " + b.to_string());
  }
  auto n1 = a.to_number();
  auto n2 = b.to_number();
  PSCM_ASSERT(n1);
  PSCM_ASSERT(n2);
  return (*n1 >= *n2) ? Cell::bool_true() : Cell::bool_false();
}

Cell is_positive(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  if (arg.is_nil()) {
    PSCM_THROW_EXCEPTION("wrong-number-of-args: " + args.to_string());
  }
  if (!arg.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type argument in position 1: " + arg.to_string());
  }
  auto num = arg.to_number();
  return (*num > "0"_num) ? Cell::bool_true() : Cell::bool_false();
}

Cell is_negative(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  if (arg.is_nil()) {
    PSCM_THROW_EXCEPTION("wrong-number-of-args: " + args.to_string());
  }
  if (!arg.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type argument in position 1: " + arg.to_string());
  }
  auto num = arg.to_number();
  return (*num < "0"_num) ? Cell::bool_true() : Cell::bool_false();
}

Cell is_odd(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  if (arg.is_nil()) {
    PSCM_THROW_EXCEPTION("wrong-number-of-args: " + args.to_string());
  }
  if (!arg.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type argument in position 1: " + arg.to_string());
  }
  auto num = arg.to_number();
  PSCM_ASSERT(num->is_int());
  auto n = num->to_int();
  return (n % 2 == 1) ? Cell::bool_true() : Cell::bool_false();
}

Cell is_even(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  if (arg.is_nil()) {
    PSCM_THROW_EXCEPTION("wrong-number-of-args: " + args.to_string());
  }
  if (!arg.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type argument in position 1: " + arg.to_string());
  }
  auto num = arg.to_number();
  PSCM_ASSERT(num->is_int());
  auto n = num->to_int();
  return (n % 2 == 0) ? Cell::bool_true() : Cell::bool_false();
}

Cell proc_max(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_num());
  auto num = arg.to_number();
  PSCM_ASSERT(num->is_int() || num->is_float());
  bool is_int = num->is_int();
  double n = is_int ? num->to_int() : num->to_float();
  for_each(
      [&n, &is_int](auto expr, auto) {
        PSCM_ASSERT(expr.is_num());
        auto num = expr.to_number();
        PSCM_ASSERT(num->is_int() || num->is_float());
        if (!num->is_int()) {
          is_int = false;
        }
        double m = num->is_int() ? num->to_int() : num->to_float();
        if (m > n) {
          n = m;
        }
      },
      cdr(args));
  if (is_int) {
    return new Number(int64_t(n));
  }
  return new Number(n);
}

Cell proc_min(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_num());
  auto num = arg.to_number();
  PSCM_ASSERT(num->is_int() || num->is_float());
  bool is_int = num->is_int();
  double n = is_int ? num->to_int() : num->to_float();
  for_each(
      [&n, &is_int](auto expr, auto) {
        PSCM_ASSERT(expr.is_num());
        auto num = expr.to_number();
        PSCM_ASSERT(num->is_int() || num->is_float());
        if (!num->is_int()) {
          is_int = false;
        }
        double m = num->is_int() ? num->to_int() : num->to_float();
        if (m < n) {
          n = m;
        }
      },
      cdr(args));
  if (is_int) {
    return new Number(int64_t(n));
  }
  return new Number(n);
}

Cell quotient(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg1.is_num());
  PSCM_ASSERT(arg2.is_num());
  auto num1 = arg1.to_number();
  auto num2 = arg2.to_number();
  PSCM_ASSERT(num1->is_int());
  PSCM_ASSERT(num2->is_int());
  auto n1 = num1->to_int();
  auto n2 = num2->to_int();

  return new Number(n1 / n2);
}

Cell remainder(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg1.is_num());
  PSCM_ASSERT(arg2.is_num());
  auto num1 = arg1.to_number();
  auto num2 = arg2.to_number();
  PSCM_ASSERT(num1->is_int());
  PSCM_ASSERT(num2->is_int());
  auto n1 = num1->to_int();
  auto n2 = num2->to_int();
  auto m = n1 / n2;
  return new Number(n1 - n2 * m);
}

Cell modulo(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg1.is_num());
  PSCM_ASSERT(arg2.is_num());
  auto num1 = arg1.to_number();
  auto num2 = arg2.to_number();
  PSCM_ASSERT(num1->is_int());
  PSCM_ASSERT(num2->is_int());
  auto n1 = num1->to_int();
  auto n2 = num2->to_int();
  auto m = n1 / n2;
  return new Number(n1 - n2 * m);
}

Cell builtin_not(Cell args) {
  PSCM_ASSERT(args.is_pair());
  if (args.is_nil()) {
    PSCM_THROW_EXCEPTION("wrong-number-of-args: " + args.to_string());
  }
  auto arg = car(args);
  if (arg.is_bool() && !arg.to_bool()) {
    return Cell::bool_true();
  }
  return Cell::bool_false();
}

Cell display(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  if (arg.is_char()) {
    PSCM_ASSERT(arg.to_char());
    arg.to_char()->display();
  }
  else if (arg.is_num()) {
    PSCM_ASSERT(arg.to_number());
    arg.to_number()->display();
  }
  else if (arg.is_str()) {
    PSCM_ASSERT(arg.to_str());
    arg.to_str()->display();
  }
  else {
    PSCM_THROW_EXCEPTION("not supported now");
  }
  return Cell::none();
}

Cell write(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  auto port = cdr(args);
  if (!port.is_nil()) {
    port = car(port);
    PSCM_ASSERT(port.is_port());
    auto p = port.to_port();
    p->write(arg);
    return Cell::none();
  }
  if (arg.is_char()) {
    PSCM_ASSERT(arg.to_char());
    arg.to_char()->display();
  }
  else if (arg.is_num()) {
    PSCM_ASSERT(arg.to_number());
    arg.to_number()->display();
  }
  else if (arg.is_str()) {
    PSCM_ASSERT(arg.to_str());
    arg.to_str()->display();
  }
  else {
    std::cout << arg;
  }
  return Cell::none();
}

Cell newline(Cell args) {
  if (args.is_nil()) {
    std::cout << std::endl;
  }
  else {
    auto arg = car(args);
    PSCM_ASSERT(arg.is_port());
    auto port = arg.to_port();
    port->write_char('\n');
  }

  return Cell::none();
}

Cell is_procedure(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  if (arg.is_proc()) {
    return Cell::bool_true();
  }
  if (arg.is_func()) {
    return Cell::bool_true();
  }
  if (arg.is_cont()) {
    return Cell::bool_true();
  }
  return Cell::bool_false();
}

Cell is_boolean(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return arg.is_bool() ? Cell::bool_true() : Cell::bool_false();
}

Cell create_list(Cell args) {
  PSCM_ASSERT(args.is_pair() || args.is_nil());
  return args;
}

Cell is_list(Cell args) {
  PSCM_ASSERT(args.is_pair());
  std::unordered_set<Pair *> p_set;
  auto arg = car(args);
  while (arg.is_pair()) {
    if (p_set.contains(arg.to_pair())) {
      return Cell::bool_false();
    }
    p_set.insert(arg.to_pair());
    arg = cdr(arg);
  }
  return arg.is_nil() ? Cell::bool_true() : Cell::bool_false();
}

Cell is_pair(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return Cell(arg.is_pair());
}

Cell set_car(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto pair = car(args);
  auto obj = cadr(args);
  if (!pair.is_pair()) {
    PSCM_THROW_EXCEPTION("Invalid set-car! args: " + args.to_string());
  }
  pair.to_pair()->first = obj;
  return Cell::none();
}

Cell set_cdr(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto pair = car(args);
  auto obj = cadr(args);
  if (!pair.is_pair()) {
    PSCM_THROW_EXCEPTION("Invalid set-cdr! args: " + args.to_string());
  }
  pair.to_pair()->second = obj;
  return Cell::none();
}

Cell proc_cons(Cell args) {
  auto a = car(args);
  auto b = cadr(args);
  return cons(a, b);
}

Cell proc_car(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return car(arg);
}

Cell proc_cdr(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return cdr(arg);
}

Cell proc_cdar(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return cdar(arg);
}

Cell proc_cadr(Cell args) {
  PSCM_ASSERT(args.is_pair());
  SPDLOG_INFO("args: {}", args);
  auto arg = car(args);
  return cadr(arg);
}

Cell proc_cddr(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return cddr(arg);
}

Cell is_eqv(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto obj1 = car(args);
  auto obj2 = cadr(args);
  return obj1.is_eqv(obj2);
}

Cell is_eq(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto obj1 = car(args);
  auto obj2 = cadr(args);
  return obj1.is_eq(obj2);
}

Cell is_equal(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto obj1 = car(args);
  auto obj2 = cadr(args);
  return (obj1 == obj2) ? Cell::bool_true() : Cell::bool_false();
}

Cell memq(Cell args) {
  auto obj = car(args);
  auto list = cadr(args);
  while (!list.is_nil()) {
    if (obj.is_eq(car(list)).to_bool()) {
      return list;
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

Cell memv(Cell args) {
  auto obj = car(args);
  auto list = cadr(args);
  while (!list.is_nil()) {
    if (obj.is_eqv(car(list)).to_bool()) {
      return list;
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

Cell member(Cell args) {
  auto obj = car(args);
  auto list = cadr(args);
  while (!list.is_nil()) {
    if (obj == car(list)) {
      return list;
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

Cell assq(Cell args) {
  auto obj = car(args);
  auto list = cadr(args);
  while (!list.is_nil()) {
    if (obj.is_eq(caar(list)).to_bool()) {
      return car(list);
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

Cell assv(Cell args) {
  auto obj = car(args);
  auto list = cadr(args);
  while (!list.is_nil()) {
    if (obj.is_eqv(caar(list)).to_bool()) {
      return car(list);
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

Cell assoc(Cell args) {
  auto obj = car(args);
  auto list = cadr(args);
  while (!list.is_nil()) {
    if (obj == caar(list)) {
      return car(list);
    }
    list = cdr(list);
  }
  return Cell::bool_false();
}

Cell is_vector(Cell args) {
  PSCM_ASSERT(args.is_pair());
  return Cell(car(args).is_vec());
}

Cell make_vector(Cell args) {
  auto k = car(args);
  PSCM_ASSERT(k.is_num());
  auto num = k.to_number();
  PSCM_ASSERT(num->is_int());
  Cell default_value = Cell::none();
  if (!cdr(args).is_nil()) {
    default_value = cadr(args);
  }
  auto v = new Cell::Vec();
  v->resize(num->to_int());
  std::fill(v->begin(), v->end(), default_value);
  auto vec = Cell(v);
  SPDLOG_INFO("vec: {} from {}", vec, (void *)v);
  return vec;
}

Cell proc_vector(Cell args) {
  SPDLOG_INFO("args: {}", args);
  auto vec = new Cell::Vec();
  for_each(
      [vec](Cell expr, auto loc) {
        vec->push_back(expr);
      },
      args);
  auto ret = Cell(vec);
  SPDLOG_INFO("ret: {}", ret);
  return ret;
}

Cell vector_length(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_vec());
  auto vec = arg.to_vec();
  return new Number(std::int64_t(vec->size()));
}

Cell vector_ref(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  auto k = cadr(args);
  PSCM_ASSERT(arg.is_vec());
  PSCM_ASSERT(k.is_num());
  auto vec = arg.to_vec();
  auto num = k.to_number();
  PSCM_ASSERT(num->is_int());
  return vec->at(num->to_int());
}

Cell vector_set(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto vec = car(args);
  auto k = cadr(args);
  auto obj = caddr(args);
  PSCM_ASSERT(vec.is_vec());
  PSCM_ASSERT(k.is_num());
  auto num = k.to_number();
  PSCM_ASSERT(num->is_int());
  SPDLOG_INFO("vec: {} from {}", vec, (void *)vec.to_vec());
  SPDLOG_INFO("k: {} --> {}", k, obj);
  auto index = num->to_int();
  auto& v = *vec.to_vec();
  if (index >= v.size()) {
    PSCM_THROW_EXCEPTION("Value out of range: " + k.to_string());
  }
  v[index] = obj;
  SPDLOG_INFO("vec: {} from {}", vec, (void *)vec.to_vec());
  return Cell::none();
}

Cell vector_to_list(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_vec());
  auto vec = arg.to_vec();
  auto ret = cons(nil, nil);
  auto p = ret;
  for (auto e : *vec) {
    auto new_pair = cons(e, nil);
    p->second = new_pair;
    p = new_pair;
  }
  return ret->second;
}

Cell list_to_vector(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  auto vec = new Cell::Vec();
  for_each(
      [vec](Cell expr, auto) {
        vec->push_back(expr);
      },
      arg);
  return vec;
}

Cell vector_fill(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  auto obj = cadr(args);
  PSCM_ASSERT(arg.is_vec());
  auto vec = arg.to_vec();
  for (auto& e : *vec) {
    e = obj;
  }
  return Cell::none();
}

Cell is_zero(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  if (!arg.is_num()) {
    PSCM_THROW_EXCEPTION("In procedure zero? in expression " + args.to_string());
  }
  auto num = arg.to_number();
  return Cell(num->is_zero());
}

Cell is_null(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return Cell(arg.is_nil());
}

Cell length(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  int count = 0;
  for_each(
      [&count](Cell expr, auto) {
        count++;
      },
      arg);
  return new Number(count);
}

Cell append(Cell args) {
  if (args.is_nil()) {
    return nil;
  }
  auto list = car(args);
  if (list.is_nil() || list.is_pair()) {
    if (list.is_nil()) {
      return cadr(args);
    }
    auto p = list;
    while (p.is_pair() && cdr(p).is_pair()) {
      p = cdr(p);
    }
    p.to_pair()->second = cadr(args);
  }
  else {
    PSCM_THROW_EXCEPTION("Wrong type argument in position 1 (expecting empty list): " + list.to_string());
  }
  return list;
}

Cell reverse(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return reverse_argl(arg);
}

Cell list_ref(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto list = car(args);
  auto k = cadr(args);
  if (!k.is_num()) {
    PSCM_THROW_EXCEPTION("Wrong type (expecting exact integer): " + k.to_string());
  }
  auto num = k.to_number();
  if (!num->is_int()) {
    PSCM_THROW_EXCEPTION("Wrong type (expecting exact integer): " + k.to_string());
  }
  auto n = num->to_int();
  Cell ret;
  int i = 0;
  for_each(
      [&ret, &i, n](Cell expr, auto) {
        if (i == n) {
          ret = expr;
        }
        i++;
      },
      list);
  return ret;
}

Cell proc_acos(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto z1 = car(args);
  PSCM_ASSERT(z1.is_num());
  auto num1 = z1.to_number();
  PSCM_ASSERT(num1->is_int());
  double v = num1->is_int() ? num1->to_int() : num1->to_float();
  auto ret = std::acos(v);
  return new Number(ret);
}

Cell expt(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto z1 = car(args);
  auto z2 = cadr(args);
  PSCM_ASSERT(z1.is_num());
  PSCM_ASSERT(z2.is_num());
  auto num1 = z1.to_number();
  auto num2 = z2.to_number();
  PSCM_ASSERT(num1->is_int());
  PSCM_ASSERT(num2->is_int());
  auto n1 = num1->to_int();
  auto n2 = num2->to_int();
  if (n1 == 0 && n2 == 0) {
    return new Number(1);
  }
  else if (n1 == 0) {
    return new Number(0);
  }
  else if (n2 == 0) {
    return new Number(1);
  }
  else {
    auto ret = n1;
    for (int i = 0; i < n2 - 1; ++i) {
      ret *= n1;
    }
    return new Number(ret);
  }
}

Cell proc_abs(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto x = car(args);
  if (x.is_num()) {
    auto num = x.to_number();
    PSCM_ASSERT(num->is_int());
    return new Number(std::abs(num->to_int()));
  }
  PSCM_THROW_EXCEPTION("Wrong type argument in position 1: " + x.to_string());
}

Cell proc_sqrt(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto x = car(args);
  if (x.is_num()) {
    auto num = x.to_number();
    PSCM_ASSERT(num->is_int());
    auto n = num->to_int();
    n = std::sqrt(n);
    return new Number(n);
  }
  PSCM_THROW_EXCEPTION("Wrong type argument in position 1: " + x.to_string());
}

Cell proc_round(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto x = car(args);
  if (x.is_num()) {
    auto num = x.to_number();
    if (num->is_int()) {
      return x;
    }
    PSCM_ASSERT(num->is_float());
    auto n = num->to_float();
    int m = std::round(n);
    return new Number(m);
  }
  PSCM_THROW_EXCEPTION("Wrong type argument in position 1: " + x.to_string());
}

Cell is_exact(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto x = car(args);
  if (x.is_num()) {
    auto num = x.to_number();
    if (num->is_int()) {
      return Cell::bool_true();
    }
    return Cell::bool_false();
  }
  PSCM_THROW_EXCEPTION("Wrong type argument in position 1: " + x.to_string());
}

Cell is_inexact(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto x = car(args);
  if (x.is_num()) {
    auto num = x.to_number();
    if (num->is_float()) {
      return Cell::bool_true();
    }
    return Cell::bool_false();
  }
  PSCM_THROW_EXCEPTION("Wrong type argument in position 1: " + x.to_string());
}

Cell inexact_to_exact(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto x = car(args);
  if (x.is_num()) {
    auto num = x.to_number();
    if (num->is_int()) {
      return x;
    }
    PSCM_ASSERT(num->is_float());
    auto n = num->to_float();
    int m = int(n);
    return new Number(m);
  }
  PSCM_THROW_EXCEPTION("Wrong type argument in position 1: " + x.to_string());
}

Cell is_symbol(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return Cell(arg.is_sym());
}

Cell symbol_to_string(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_sym());
  auto name = arg.to_symbol()->name();
  return new String(std::string(name));
}

Cell string_to_symbol(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return new Symbol(arg.to_str()->str());
}

Cell is_string(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return Cell(arg.is_str());
}

Cell make_string(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto k = car(args);
  PSCM_ASSERT(k.is_num());
  auto num = k.to_number();
  PSCM_ASSERT(num->is_int());
  auto sz = num->to_int();
  char ch = '\0';
  if (!cdr(args).is_nil()) {
    auto ch_tmp = cadr(args);
    PSCM_ASSERT(ch_tmp.is_char());
    auto ch_tmp2 = ch_tmp.to_char();
    int n = ch_tmp2->to_int();
    ch = n;
  }
  return new String(sz, ch);
}

Cell proc_string(Cell args) {
  std::string s;
  for_each(
      [&s](Cell expr, auto) {
        PSCM_ASSERT(expr.is_char());
        auto ch = expr.to_char();
        auto n = ch->to_int();
        s.push_back(n);
      },
      args);
  return new String(std::move(s));
}

Cell string_length(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_str());
  auto s = arg.to_str();
  return new Number(int64_t(s->length()));
}

Cell string_ref(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  auto k = cadr(args);
  PSCM_ASSERT(arg.is_str());
  PSCM_ASSERT(k.is_num());
  auto s = arg.to_str()->str();
  auto num = k.to_number();
  auto idx = num->to_int();
  return Char::from(s[idx]);
}

Cell string_set(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  auto k = cadr(args);
  auto ch = caddr(args);
  PSCM_ASSERT(arg.is_str());
  PSCM_ASSERT(k.is_num());
  PSCM_ASSERT(ch.is_char());
  auto s = arg.to_str();
  auto num = k.to_number();
  auto idx = num->to_int();
  s->set(idx, ch.to_char()->to_int());
  return Cell::none();
}

Cell is_string_equal(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto lhs = car(args);
  auto rhs = cadr(args);
  PSCM_ASSERT(lhs.is_str());
  PSCM_ASSERT(rhs.is_str());
  return Cell((*lhs.to_str()) == (*rhs.to_str()));
}

Cell is_string_less(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto lhs = car(args);
  auto rhs = cadr(args);
  PSCM_ASSERT(lhs.is_str());
  PSCM_ASSERT(rhs.is_str());
  return Cell((*lhs.to_str()) < (*rhs.to_str()));
}

Cell is_string_greater(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto lhs = car(args);
  auto rhs = cadr(args);
  PSCM_ASSERT(lhs.is_str());
  PSCM_ASSERT(rhs.is_str());
  return Cell((*lhs.to_str()) > (*rhs.to_str()));
}

Cell is_string_less_or_equal(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto lhs = car(args);
  auto rhs = cadr(args);
  PSCM_ASSERT(lhs.is_str());
  PSCM_ASSERT(rhs.is_str());
  return Cell((*lhs.to_str()) <= (*rhs.to_str()));
}

Cell is_string_greater_or_equal(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto lhs = car(args);
  auto rhs = cadr(args);
  PSCM_ASSERT(lhs.is_str());
  PSCM_ASSERT(rhs.is_str());
  return Cell((*lhs.to_str()) >= (*rhs.to_str()));
}

Cell is_string_equal_case_insensitive(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto lhs = car(args);
  auto rhs = cadr(args);
  PSCM_ASSERT(lhs.is_str());
  PSCM_ASSERT(rhs.is_str());
  return Cell((*lhs.to_str()).to_downcase() == (*rhs.to_str()).to_downcase());
}

Cell is_string_less_case_insensitive(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto lhs = car(args);
  auto rhs = cadr(args);
  PSCM_ASSERT(lhs.is_str());
  PSCM_ASSERT(rhs.is_str());
  return Cell((*lhs.to_str()).to_downcase() < (*rhs.to_str()).to_downcase());
}

Cell is_string_greater_case_insensitive(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto lhs = car(args);
  auto rhs = cadr(args);
  PSCM_ASSERT(lhs.is_str());
  PSCM_ASSERT(rhs.is_str());
  return Cell((*lhs.to_str()).to_downcase() > (*rhs.to_str()).to_downcase());
}

Cell is_string_less_or_equal_case_insensitive(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto lhs = car(args);
  auto rhs = cadr(args);
  PSCM_ASSERT(lhs.is_str());
  PSCM_ASSERT(rhs.is_str());
  return Cell((*lhs.to_str()).to_downcase() <= (*rhs.to_str()).to_downcase());
}

Cell is_string_greater_or_equal_case_insensitive(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto lhs = car(args);
  auto rhs = cadr(args);
  PSCM_ASSERT(lhs.is_str());
  PSCM_ASSERT(rhs.is_str());
  return Cell((*lhs.to_str()).to_downcase() >= (*rhs.to_str()).to_downcase());
}

Cell proc_substring(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  auto start = cadr(args);
  auto end = caddr(args);
  PSCM_ASSERT(arg.is_str());
  PSCM_ASSERT(start.is_num());
  PSCM_ASSERT(end.is_num());
  auto s = arg.to_str();
  auto num_start = start.to_number();
  auto num_end = end.to_number();
  PSCM_ASSERT(num_start->is_int());
  PSCM_ASSERT(num_end->is_int());
  return new String(s->substring(num_start->to_int(), num_end->to_int()));
}

Cell string_append(Cell args) {
  std::string s;
  for_each(
      [&s](Cell expr, auto) {
        PSCM_ASSERT(expr.is_str());
        s += expr.to_str()->str();
      },
      args);
  return new String(std::move(s));
}

Cell string_to_list(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_str());
  auto s = arg.to_str();
  auto p = cons(nil, nil);
  Cell ret = p;
  for (auto ch : s->str()) {
    auto new_pair = cons(Char::from(ch), nil);
    p->second = new_pair;
    p = new_pair;
  }
  return cdr(ret);
}

Cell list_to_string(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  std::string s;
  for_each(
      [&s](Cell expr, auto) {
        PSCM_ASSERT(expr.is_char());
        auto ch = expr.to_char();
        s.push_back(ch->to_int());
      },
      args);
  return new String(std::move(s));
}

Cell string_copy(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_str());
  auto s = arg.to_str();
  return new String(std::string(s->str()));
}

Cell string_fill(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  auto ch = cadr(args);
  PSCM_ASSERT(arg.is_str());
  PSCM_ASSERT(arg.is_char());
  auto s = arg.to_str();
  s->fill(ch.to_char()->to_int());
  return arg;
}

Cell is_number(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return Cell(arg.is_num());
}

Cell is_complex(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_num());
  auto num = arg.to_number();
  return Cell(num->is_int() || num->is_complex());
}

Cell is_real(Cell args) {
  if (is_integer(args).to_bool()) {
    return Cell::bool_true();
  }
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_num());
  auto num = arg.to_number();
  return Cell(num->is_int() || num->is_float());
}

Cell is_integer(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_num());
  auto num = arg.to_number();
  if (num->is_int()) {
    return Cell::bool_true();
  }
  if (num->is_float()) {
    auto n = num->to_float();
    return Cell(std::int64_t(n) == n);
  }
  if (num->is_rational()) {
    auto r = num->to_rational();
    return Cell(std::int64_t(r.to_float()) == r.to_int());
  }
  if (num->is_complex()) {
    auto c = num->to_complex();
    auto imag_part = c.imag_part();
    return Cell(std::int64_t(imag_part) == imag_part);
  }
  return Cell::bool_false();
}

Cell is_rational(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_num());
  auto num = arg.to_number();
  return Cell(num->is_rational());
}

Cell string_to_number(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_str());
  auto s = arg.to_str();
  Parser parser(std::string(s->str()));
  try {
    auto num = parser.parse();
    if (num.is_num()) {
      return num;
    }
  }
  catch (...) {
    // do nothing
  }
  return Cell::bool_false();
}

Cell number_to_string(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_num());
  auto s = arg.to_string();
  return new String(s);
}

Cell is_char(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return Cell(arg.is_char());
}

Cell is_char_equal(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg1.is_char());
  PSCM_ASSERT(arg2.is_char());
  auto ch1 = arg1.to_char();
  auto ch2 = arg2.to_char();
  return Cell(*ch1 == *ch2);
}

Cell is_char_less(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg1.is_char());
  PSCM_ASSERT(arg2.is_char());
  auto ch1 = arg1.to_char();
  auto ch2 = arg2.to_char();
  return Cell(*ch1 < *ch2);
}

Cell is_char_greater(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg1.is_char());
  PSCM_ASSERT(arg2.is_char());
  auto ch1 = arg1.to_char();
  auto ch2 = arg2.to_char();
  return Cell(*ch1 > *ch2);
}

Cell is_char_less_or_equal(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg1.is_char());
  PSCM_ASSERT(arg2.is_char());
  auto ch1 = arg1.to_char();
  auto ch2 = arg2.to_char();
  return Cell(*ch1 <= *ch2);
}

Cell is_char_greater_or_equal(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg1.is_char());
  PSCM_ASSERT(arg2.is_char());
  auto ch1 = arg1.to_char();
  auto ch2 = arg2.to_char();
  return Cell(*ch1 >= *ch2);
}

Cell is_char_equal_case_insensitive(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg1.is_char());
  PSCM_ASSERT(arg2.is_char());
  auto ch1 = arg1.to_char();
  auto ch2 = arg2.to_char();
  return Cell(ch1->to_downcase() == ch2->to_downcase());
}

Cell is_char_less_case_insensitive(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg1.is_char());
  PSCM_ASSERT(arg2.is_char());
  auto ch1 = arg1.to_char();
  auto ch2 = arg2.to_char();
  return Cell(ch1->to_downcase() < ch2->to_downcase());
}

Cell is_char_greater_case_insensitive(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg1.is_char());
  PSCM_ASSERT(arg2.is_char());
  auto ch1 = arg1.to_char();
  auto ch2 = arg2.to_char();
  return Cell(ch1->to_downcase() > ch2->to_downcase());
}

Cell is_char_less_or_equal_case_insensitive(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg1.is_char());
  PSCM_ASSERT(arg2.is_char());
  auto ch1 = arg1.to_char();
  auto ch2 = arg2.to_char();
  return Cell(ch1->to_downcase() <= ch2->to_downcase());
}

Cell is_char_greater_or_equal_case_insensitive(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg1.is_char());
  PSCM_ASSERT(arg2.is_char());
  auto ch1 = arg1.to_char();
  auto ch2 = arg2.to_char();
  return Cell(ch1->to_downcase() >= ch2->to_downcase());
}

Cell is_char_alphabetic(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_char());
  auto ch = arg.to_char();
  return Cell(ch->is_alphabetic());
}

Cell is_char_numeric(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_char());
  auto ch = arg.to_char();
  return Cell(ch->is_numeric());
}

Cell is_char_whitespace(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_char());
  auto ch = arg.to_char();
  return Cell(ch->is_whitespace());
}

Cell is_char_upper_case(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_char());
  auto ch = arg.to_char();
  return Cell(ch->to_upcase() == *ch);
}

Cell is_char_lower_case(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_char());
  auto ch = arg.to_char();
  return Cell(ch->to_downcase() == *ch);
}

Cell char_to_integer(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_char());
  auto ch = arg.to_char();
  return new Number(ch->to_int());
}

Cell integer_to_char(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_num());
  auto num = arg.to_number();
  PSCM_ASSERT(num->is_int());
  auto n = num->to_int();
  return Char::from(n);
}

Cell char_upcase(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_char());
  auto ch = arg.to_char();
  return new Char(ch->to_upcase());
}

Cell char_downcase(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_char());
  auto ch = arg.to_char();
  return new Char(ch->to_downcase());
}

Cell reverse_argl(Cell argl) {
  auto p = cons(nil, nil);
  auto args = argl;
  while (!args.is_nil()) {
    p->second = cons(car(args), p->second);
    args = cdr(args);
  }
  return p->second;
}

Cell construct_apply_argl(Cell argl) {
  auto args = cons(nil, nil);
  auto p = args;
  while (true) {
    auto a = car(argl);
    auto b = cdr(argl);
    if (b.is_nil()) {
      p->second = a;
      break;
    }
    else {
      auto new_pair = cons(a, nil);
      p->second = new_pair;
      p = new_pair;
    }
    argl = cdr(argl);
  }
  return args->second;
}

Evaluator::Evaluator(Scheme& scm)
    : scm_(scm) {
}

Cell Evaluator::eval(Cell expr, SymbolTable *env) {
  reg_ = Register{ .expr = expr, .env = env, .cont = Label::DONE };
  pos_ = Label::EVAL;
  run();
  SPDLOG_INFO("eval ret: {}", reg_.val);
  return reg_.val;
}

void Evaluator::run() {
  while (true) {
    if (step_ > 10000) {
      PSCM_THROW_EXCEPTION("evaluator terminate due to reach max step: " + std::to_string(step_));
    }
    step_++;
    switch (pos_) {
    case Label::DONE: {
      if (!stack_.empty()) {
        PSCM_THROW_EXCEPTION("stack is not empty. need debug: " + stack_.to_string());
      }
      return;
    }
    case Label::EVAL: {
      PRINT_STEP();
      SPDLOG_INFO("eval expr: {}", reg_.expr);
      if (reg_.expr.is_self_evaluated()) {
        reg_.val = reg_.expr;
        GOTO(reg_.cont);
      }
      if (reg_.expr.is_sym()) {
        reg_.val = reg_.env->get(reg_.expr.to_symbol());
        auto sym = reg_.expr.to_symbol();
        GOTO(reg_.cont);
      }
      if (reg_.expr.is_pair()) {
        GOTO(Label::APPLY);
      }
      SPDLOG_ERROR("unsupported expr: {}", reg_.expr);
      PSCM_THROW_EXCEPTION("unsupported expr");
    }
    case Label::APPLY: {
      PRINT_STEP();
      SPDLOG_INFO("apply: {}", reg_.expr);
      // restore after_apply
      PSCM_PUSH_STACK(env);
      PSCM_PUSH_STACK(unev);
      PSCM_PUSH_STACK(proc);
      PSCM_PUSH_STACK(argl);
      PSCM_PUSH_STACK(expr);
      PSCM_PUSH_STACK(cont);
      reg_.cont = Label::AFTER_APPLY;
      PSCM_PUSH_STACK(cont);
      reg_.unev = cdr(reg_.expr);
      reg_.expr = car(reg_.expr);
      reg_.cont = Label::AFTER_EVAL_OP;
      GOTO(Label::EVAL);
    }
    case Label::APPLY_MACRO: {
      PRINT_STEP();
      PSCM_ASSERT(reg_.proc.is_macro());
      auto f = reg_.proc.to_macro();
      PSCM_ASSERT(f->is_func());
      reg_.cont = Label::AFTER_APPLY_MACRO;
      reg_.expr = f->call(reg_.unev);
      GOTO(Label::EVAL);
    }
    case Label::APPLY_FUNC: {
      PRINT_STEP();
      SPDLOG_INFO("apply {} with args: {}", reg_.proc, reg_.argl);
      PSCM_ASSERT(reg_.proc.is_func());
      auto f = reg_.proc.to_func();
      auto args = reg_.argl;
      SPDLOG_INFO("func args: {}", args);
      reg_.val = f->call(args);
      SPDLOG_INFO("func ret: {}", reg_.val);
      PSCM_POP_STACK(argl);
      PSCM_POP_STACK(cont);
      GOTO(reg_.cont);
    }
    case Label::APPLY_PROC: {
      PRINT_STEP();
      SPDLOG_INFO("apply {} with args: {}", reg_.proc, reg_.argl);
      PSCM_ASSERT(reg_.proc.is_proc());
      auto proc = reg_.proc.to_proc();
      PSCM_ASSERT(proc);
      auto args = reg_.argl;
      PSCM_POP_STACK(argl);
      bool ok = proc->check_args(args);
      if (!ok) {
        PSCM_THROW_EXCEPTION("Wrong number of arguments to " + Cell(proc).to_string());
      }
      auto proc_env = proc->create_proc_env(args);

      reg_.env = proc_env;
      if (proc->body_.is_nil()) {
        if (proc->name_ == &callcc) {
          // special handle for call/cc
          PSCM_ASSERT(args.is_pair());
          auto p2 = car(args);
          reg_.expr = cons(p2, cons(new Continuation(reg_, stack_, reg_type_stack_), nil));
          reg_.cont = Label::AFTER_APPLY_PROC;
          GOTO(Label::EVAL);
        }
        else if (proc->name_ == &call_with_values) {
          // special handle for call-with-values
          PSCM_ASSERT(args.is_pair());
          SPDLOG_INFO("call-with-values args: {}", args);
          reg_.expr = cons(car(args), nil);
          reg_.unev = cdr(args);
          reg_.cont = Label::AFTER_EVAL_CALL_WITH_VALUES_PRODUCER;
          GOTO(Label::EVAL);
        }
        else if (proc->name_ == &values) {
          // special handle for values
          PSCM_ASSERT(args.is_pair());
          SPDLOG_INFO("values args: {}", args);
          reg_.val = args;
          GOTO(Label::AFTER_APPLY_PROC);
        }
        else {
          reg_.val = nil;
          GOTO(Label::AFTER_APPLY_PROC);
        }
      }
      else {
        //        reg_.expr = proc->body_;
        reg_.expr = car(proc->body_);
        reg_.unev = cdr(proc->body_);
        reg_.cont = Label::AFTER_EVAL_FIRST_EXPR;
        GOTO(Label::EVAL);
      }
    }
    case Label::APPLY_CONT: {
      PRINT_STEP();
      SPDLOG_INFO("apply {} with args: {}", reg_.proc, reg_.argl);
      PSCM_ASSERT(reg_.proc.is_cont());
      auto cont = reg_.proc.to_cont();
      auto args = reg_.argl;
      if (args.is_nil()) {
        PSCM_THROW_EXCEPTION("Invalid arguments of Continuation: " + reg_.proc.to_string());
      }
      auto val = car(args);
      SPDLOG_INFO("val: {}", val);
      if (!cdr(args).is_nil()) {
        PSCM_THROW_EXCEPTION("Invalid arguments of Continuation: " + reg_.proc.to_string());
      }
      reg_ = cont->reg_;
      stack_ = cont->stack_;
      reg_type_stack_ = cont->reg_type_stack_;
      reg_.val = val;
      reg_.env = new SymbolTable(reg_.env);
      PSCM_POP_STACK(cont);
      GOTO(reg_.cont);
    }
    case Label::EVAL_ARGS: {
      PRINT_STEP();
      PSCM_PUSH_STACK(argl);
      PSCM_PUSH_STACK(cont);
      reg_.argl = nil;
      if (reg_.unev.is_nil()) {
        GOTO(Label::AFTER_EVAL_ARGS);
      }
      else {
        reg_.expr = car(reg_.unev);
        reg_.unev = cdr(reg_.unev);
        reg_.cont = Label::AFTER_EVAL_FIRST_ARG;
        GOTO(Label::EVAL);
      }
    }
    case Label::AFTER_EVAL_FIRST_ARG: {
      PRINT_STEP();
      reg_.argl = cons(reg_.val, reg_.argl);
      if (reg_.unev.is_nil()) {
        GOTO(Label::AFTER_EVAL_ARGS);
      }
      else {
        PSCM_PUSH_STACK(argl);
        SPDLOG_INFO("push argl: {}", reg_.argl);
        reg_.expr = car(reg_.unev);
        reg_.unev = cdr(reg_.unev);
        reg_.cont = Label::AFTER_EVAL_OTHER_ARG;
        GOTO(Label::EVAL);
      }
    }
    case Label::AFTER_EVAL_OTHER_ARG: {
      PRINT_STEP();
      PSCM_POP_STACK(argl);
      SPDLOG_INFO("pop argl: {}", reg_.argl);
      reg_.argl = cons(reg_.val, reg_.argl);
      SPDLOG_INFO("argl: {}", reg_.argl);
      if (reg_.unev.is_nil()) {
        GOTO(Label::AFTER_EVAL_ARGS);
      }
      else {
        PSCM_PUSH_STACK(argl);
        SPDLOG_INFO("push argl: {}", reg_.argl);
        reg_.expr = car(reg_.unev);
        reg_.unev = cdr(reg_.unev);
        reg_.cont = Label::AFTER_EVAL_OTHER_ARG;
        GOTO(Label::EVAL);
      }
    }
    case Label::AFTER_EVAL_ARGS: {
      PRINT_STEP();
      reg_.argl = reverse_argl(reg_.argl);
      PSCM_POP_STACK(cont);
      GOTO(reg_.cont);
    }
    case Label::AFTER_EVAL_OP: {
      PRINT_STEP();
      PSCM_POP_STACK(cont);
      reg_.proc = reg_.val;
      PSCM_ASSERT(reg_.proc.is_func() || reg_.proc.is_proc() || reg_.proc.is_macro() || reg_.proc.is_cont());
      if (reg_.proc.is_func()) {
        PSCM_PUSH_STACK(cont);
        reg_.cont = Label::APPLY_FUNC;
        GOTO(Label::EVAL_ARGS);
      }
      else if (reg_.proc.is_proc()) {
        PSCM_PUSH_STACK(cont);
        reg_.cont = Label::APPLY_PROC;
        GOTO(Label::EVAL_ARGS);
      }
      else if (reg_.proc.is_macro()) {
        PSCM_PUSH_STACK(cont);
        //        PSCM_PUSH_STACK(argl);
        reg_.cont = Label::AFTER_APPLY_MACRO;
        auto macro_pos = reg_.proc.to_macro()->pos();
        GOTO(macro_pos);
      }
      else if (reg_.proc.is_cont()) {
        reg_.cont = Label::APPLY_CONT;
        GOTO(Label::EVAL_ARGS);
      }
      PSCM_THROW_EXCEPTION("unsupported " + reg_.proc.to_string());
    }
    case Label::AFTER_APPLY: {
      PRINT_STEP();
      PSCM_POP_STACK(cont);
      PSCM_POP_STACK(expr);
      PSCM_POP_STACK(argl);
      PSCM_POP_STACK(proc);
      PSCM_POP_STACK(unev);
      PSCM_POP_STACK(env);
      GOTO(reg_.cont);
    }
    case Label::AFTER_APPLY_FUNC: {
      PRINT_STEP();
      PSCM_POP_STACK(argl);
      SPDLOG_INFO("pop argl: {}", reg_.argl);
      pos_ = Label::AFTER_APPLY;
      break;
    }
    case Label::AFTER_APPLY_PROC: {
      PRINT_STEP();
      PSCM_POP_STACK(cont);
      GOTO(reg_.cont);
    }
    case Label::AFTER_APPLY_MACRO: {
      PRINT_STEP();
      PSCM_POP_STACK(cont);
      GOTO(reg_.cont);
    }
    case Label::APPLY_APPLY: {
      PRINT_STEP();
      auto proc = car(reg_.unev);
      auto args = cdr(reg_.unev);
      SPDLOG_INFO("apply {}: {}", proc, args);
      PSCM_ASSERT(proc.is_sym());
      PSCM_ASSERT(args.is_sym());
      proc = reg_.env->get(proc.to_symbol());
      args = reg_.env->get(args.to_symbol());
      args = construct_apply_argl(args);
      reg_.cont = Label::AFTER_APPLY_MACRO;
      PSCM_PUSH_STACK(cont);
      if (proc.is_func()) {
        PSCM_PUSH_STACK(argl);
        reg_.proc = proc;
        reg_.argl = args;
        GOTO(Label::APPLY_FUNC);
      }
      else if (proc.is_proc()) {
        PSCM_PUSH_STACK(argl);
        reg_.proc = proc;
        reg_.argl = args;
        GOTO(Label::APPLY_PROC);
      }
      else if (proc.is_cont()) {
        PSCM_PUSH_STACK(argl);
        reg_.proc = proc;
        reg_.argl = list(args);
        GOTO(Label::APPLY_CONT);
      }
      else {
        PSCM_THROW_EXCEPTION("not supported now");
      }
    }
    case Label::APPLY_DEFINE: {
      PRINT_STEP();
      auto var_name = car(reg_.unev);
      auto val = cdr(reg_.unev);
      if (var_name.is_sym()) {
        reg_.val = var_name;
        SPDLOG_INFO("val: {}", val);
        reg_.expr = car(val);
      }
      else {
        auto proc_name = car(var_name);
        auto args = cdr(var_name);
        auto expr = cons(lambda, cons(args, val));
        reg_.val = proc_name;
        reg_.expr = expr;
      }
      PSCM_PUSH_STACK(val);
      reg_.cont = Label::AFTER_EVAL_DEFINE_ARG;
      GOTO(Label::EVAL);
    }
    case Label::APPLY_COND: {
      PRINT_STEP();
      auto clause = car(reg_.unev);
      auto test = car(clause);
      if (test.is_sym()) {
        PSCM_ASSERT(test.to_symbol());
        auto sym = test.to_symbol();
        if (*sym == cond_else) {
          reg_.expr = cadr(clause);
          reg_.cont = Label::AFTER_APPLY_MACRO;
          GOTO(Label::EVAL);
        }
      }
      reg_.expr = test;
      reg_.cont = Label::AFTER_EVAL_COND_TEST;
      GOTO(Label::EVAL);
    }
    case Label::APPLY_IF: {
      PRINT_STEP();
      reg_.expr = car(reg_.unev);
      reg_.cont = Label::AFTER_EVAL_IF_PRED;
      GOTO(Label::EVAL);
    }
    case Label::APPLY_AND: {
      PRINT_STEP();
      if (reg_.unev.is_nil()) {
        reg_.val = Cell::bool_true();
        GOTO(Label::AFTER_APPLY_MACRO);
      }
      auto expr = car(reg_.unev);
      reg_.expr = expr;
      reg_.cont = Label::AFTER_EVAL_AND_EXPR;
      GOTO(Label::EVAL);
    }
    case Label::APPLY_OR: {
      PRINT_STEP();
      if (reg_.unev.is_nil()) {
        reg_.val = Cell::bool_false();
        GOTO(Label::AFTER_APPLY_MACRO);
      }
      auto expr = car(reg_.unev);
      reg_.expr = expr;
      reg_.cont = Label::AFTER_EVAL_OR_EXPR;
      GOTO(Label::EVAL);
    }
    case Label::APPLY_SET: {
      PRINT_STEP();
      auto var_name = car(reg_.unev);
      auto val = cadr(reg_.unev);
      PSCM_ASSERT(var_name.is_sym());
      reg_.val = var_name;
      PSCM_PUSH_STACK(val);
      reg_.expr = val;
      reg_.cont = Label::AFTER_EVAL_SET_ARG;
      GOTO(Label::EVAL);
    }
    case Label::APPLY_DELAY: {
      PRINT_STEP();
      auto expr = car(reg_.unev);
      auto proc = new Procedure(nullptr, nil, list(expr), reg_.env);
      Cell p = new Promise(proc);
      reg_.val = p;
      GOTO(Label::AFTER_APPLY_MACRO);
    }
    case Label::APPLY_BEGIN: {
      PRINT_STEP();
      if (reg_.unev.is_nil()) {
        reg_.val = Cell::none();
        GOTO(Label::AFTER_APPLY_MACRO);
      }
      else {
        reg_.cont = Label::AFTER_APPLY_MACRO;
        PSCM_PUSH_STACK(cont);
        reg_.cont = Label::AFTER_EVAL_FIRST_EXPR;
        reg_.expr = car(reg_.unev);
        reg_.unev = cdr(reg_.unev);
        GOTO(Label::EVAL);
      }
    }
    case Label::AFTER_EVAL_DEFINE_ARG: {
      PRINT_STEP();
      auto val = reg_.val;
      PSCM_POP_STACK(val);
      auto var_name = reg_.val;
      PSCM_ASSERT(var_name.is_sym());
      auto sym = var_name.to_symbol();
      reg_.env->insert(sym, val);
      if (val.is_proc()) {
        auto proc = val.to_proc();
        PSCM_ASSERT(proc);
        proc->name_ = sym;
      }
      reg_.val = Cell{};
      PSCM_POP_STACK(cont);
      GOTO(reg_.cont);
    }
    case Label::AFTER_EVAL_SET_ARG: {
      PRINT_STEP();
      auto val = reg_.val;
      PSCM_POP_STACK(val);
      auto var_name = reg_.val;
      PSCM_ASSERT(var_name.is_sym());
      auto sym = var_name.to_symbol();
      SPDLOG_INFO("AFTER set! {} -> {}", var_name, val);
      reg_.env->set(sym, val);
      reg_.val = Cell{};
      PSCM_POP_STACK(cont);
      GOTO(reg_.cont);
    }
    case Label::APPLY_LAMBDA: {
      PRINT_STEP();
      auto args = car(reg_.unev);
      auto body = cdr(reg_.unev);
      auto proc = new Procedure(nullptr, args, body, reg_.env);
      reg_.val = proc;
      PSCM_POP_STACK(cont);
      GOTO(reg_.cont);
    }
    case Label::APPLY_QUOTE: {
      PRINT_STEP();
      reg_.val = car(reg_.unev);
      PSCM_POP_STACK(cont);
      GOTO(reg_.cont);
    }
    case Label::APPLY_QUASIQUOTE: {
      PRINT_STEP();
      reg_.cont = Label::AFTER_APPLY_MACRO;
      reg_.expr = scm_quasiquote(scm_, reg_.env, reg_.unev);
      GOTO(Label::EVAL);
    }
    case Label::APPLY_FOR_EACH: {
      PRINT_STEP();
      auto proc = car(reg_.unev);
      auto lists = cdr(reg_.unev);
      proc = reg_.env->get(proc.to_symbol());
      lists = reg_.env->get(lists.to_symbol());
      SPDLOG_INFO("proc: {}", proc);
      SPDLOG_INFO("lists: {}", lists);
      PSCM_ASSERT(proc.is_proc() || proc.is_func());
      PSCM_ASSERT(lists.is_pair());
      auto len = list_length(lists);
      if (len == 0) {
        PSCM_THROW_EXCEPTION("ERROR args of proc: " + proc.to_string());
      }

      if (len == 1) {
        auto list1 = car(lists);
        if (list1.is_nil()) {
          reg_.val = Cell::none();
          PSCM_POP_STACK(cont);
          GOTO(reg_.cont);
        }
        reg_.expr = list(proc, car(list1));
        reg_.unev = list(cdr(list1));
      }
      else if (len == 2) {
        auto list1 = car(lists);
        auto list2 = cadr(lists);
        if (list1.is_nil() && list2.is_nil()) {
          reg_.val = Cell::none();
          PSCM_POP_STACK(cont);
          GOTO(reg_.cont);
        }
        reg_.expr = list(proc, car(list1), car(list2));
        reg_.unev = list(cdr(list1), cdr(list2));
      }
      else {
        PSCM_THROW_EXCEPTION("not support now");
      }
      reg_.proc = proc;
      reg_.cont = Label::AFTER_EVAL_FOR_EACH_FIRST_EXPR;
      GOTO(Label::EVAL);
    }
    case Label::APPLY_MAP: {
      PRINT_STEP();
      reg_.argl = nil;

      auto proc = car(reg_.unev);
      auto lists = cdr(reg_.unev);
      reg_.proc = reg_.env->get(proc.to_symbol());
      reg_.unev = reg_.env->get(lists.to_symbol());
      auto pos = eval_map_expr(Label::AFTER_EVAL_MAP_FIRST_EXPR);
      GOTO(pos);
    }
    case Label::APPLY_FORCE: {
      PRINT_STEP();
      auto promise = car(reg_.unev);
      PSCM_ASSERT(promise.is_sym());
      promise = reg_.env->get(promise.to_symbol());
      PSCM_ASSERT(promise.is_promise());
      auto p = promise.to_promise();
      if (p->ready()) {
        reg_.val = p->result();
        GOTO(Label::AFTER_APPLY_MACRO);
      }
      else {
        PSCM_PUSH_STACK(unev);
        auto proc = p->proc();
        reg_.expr = list(Cell(proc));
        reg_.cont = Label::AFTER_EVAL_PROMISE;
        GOTO(Label::EVAL);
      }
    }
    case Label::AFTER_EVAL_PROMISE: {
      PSCM_POP_STACK(unev);
      auto promise = car(reg_.unev);
      PSCM_ASSERT(promise.is_sym());
      promise = reg_.env->get(promise.to_symbol());
      PSCM_ASSERT(promise.is_promise());
      auto p = promise.to_promise();
      p->set_result(reg_.val);
      GOTO(Label::AFTER_APPLY_MACRO);
    }
    case Label::AFTER_EVAL_FIRST_EXPR: {
      PRINT_STEP();
      if (reg_.unev.is_nil()) {
        PSCM_POP_STACK(cont);
        GOTO(reg_.cont);
      }
      else {
        reg_.expr = car(reg_.unev);
        reg_.unev = cdr(reg_.unev);
        pos_ = Label::EVAL;
        break;
      }
    }
    case Label::AFTER_EVAL_FOR_EACH_FIRST_EXPR: {
      PRINT_STEP();
      auto proc = reg_.proc;
      auto lists = reg_.unev;
      auto len = list_length(reg_.unev);
      if (len == 0) {
        PSCM_THROW_EXCEPTION("ERROR args of proc: " + proc.to_string());
      }

      if (len == 1) {
        auto list1 = car(lists);
        if (list1.is_nil()) {
          reg_.val = Cell::none();
          PSCM_POP_STACK(cont);
          GOTO(reg_.cont);
        }
        reg_.expr = list(proc, car(list1));
        reg_.unev = list(cdr(list1));
      }
      else if (len == 2) {
        auto list1 = car(lists);
        auto list2 = cadr(lists);
        if (list1.is_nil() && list2.is_nil()) {
          reg_.val = Cell::none();
          PSCM_POP_STACK(cont);
          GOTO(reg_.cont);
        }
        reg_.expr = list(proc, car(list1), car(list2));
        reg_.unev = list(cdr(list1), cdr(list2));
      }
      else {
        PSCM_THROW_EXCEPTION("not support now");
      }
      reg_.cont = Label::AFTER_EVAL_FOR_EACH_FIRST_EXPR;
      GOTO(Label::EVAL);
    }

    case Label::AFTER_EVAL_MAP_FIRST_EXPR: {
      PRINT_STEP();
      reg_.argl = cons(reg_.val, reg_.argl);
      auto pos = eval_map_expr(Label::AFTER_EVAL_MAP_OTHER_EXPR);
      if (pos == Label::EVAL) {
        PSCM_PUSH_STACK(argl);
      }

      GOTO(pos);
    }
    case Label::AFTER_EVAL_MAP_OTHER_EXPR: {
      PRINT_STEP();
      PSCM_POP_STACK(argl);
      reg_.argl = cons(reg_.val, reg_.argl);
      auto pos = eval_map_expr(Label::AFTER_EVAL_MAP_OTHER_EXPR);
      if (pos == Label::EVAL) {
        PSCM_PUSH_STACK(argl);
      }
      else {
        reg_.val = reverse_argl(reg_.val);
      }

      GOTO(pos);
    }
    case Label::AFTER_EVAL_COND_TEST: {
      PRINT_STEP();
      if (reg_.val.is_bool() && !reg_.val.to_bool()) {
        reg_.unev = cdr(reg_.unev);
        if (reg_.unev.is_nil()) {
          reg_.val = Cell::none();
          GOTO(Label::AFTER_APPLY_MACRO);
        }
        auto clause = car(reg_.unev);
        auto test = car(clause);
        if (test.is_sym()) {
          PSCM_ASSERT(test.to_symbol());
          auto sym = test.to_symbol();
          if (*sym == cond_else) {
            reg_.expr = cadr(clause);
            reg_.cont = Label::AFTER_APPLY_MACRO;
            SPDLOG_INFO("cond expr: {}", reg_.expr);
            GOTO(Label::EVAL);
          }
        }
        reg_.expr = car(clause);
        reg_.cont = Label::AFTER_EVAL_COND_TEST;
        GOTO(Label::EVAL);
      }
      auto clause = car(reg_.unev);
      auto tmp = cdr(clause);
      if (tmp.is_nil()) {
        PSCM_THROW_EXCEPTION("Invalid cond expr: " + clause.to_string());
      }
      auto arrow = car(tmp);
      if (arrow.is_sym() && *arrow.to_symbol() == "=>"_sym) {
        auto recipient = cadr(tmp);
        reg_.expr = cons(recipient, list(list(quote, reg_.val)));
        reg_.cont = Label::AFTER_APPLY_MACRO;
        GOTO(Label::EVAL);
      }
      else {
        auto expr = cadr(clause);
        reg_.expr = expr;
        SPDLOG_INFO("cond expr: {}", reg_.expr);
        reg_.cont = Label::AFTER_APPLY_MACRO;
        GOTO(Label::EVAL);
      }
    }
    case Label::AFTER_EVAL_IF_PRED: {
      PRINT_STEP();
      PSCM_ASSERT(reg_.val.is_bool());
      bool pred = reg_.val.to_bool();
      if (pred) {
        auto consequent = cadr(reg_.unev);
        reg_.expr = consequent;
      }
      else {
        auto alternate = cddr(reg_.unev);
        if (alternate.is_nil()) {
          GOTO(Label::AFTER_APPLY_MACRO);
        }
        else {
          reg_.expr = car(alternate);
        }
      }
      reg_.cont = Label::AFTER_APPLY_MACRO;
      GOTO(Label::EVAL);
    }
    case Label::AFTER_EVAL_AND_EXPR: {
      PRINT_STEP();
      if (reg_.val.is_bool() && !reg_.val.to_bool()) {
        reg_.val = Cell::bool_false();
        GOTO(Label::AFTER_APPLY_MACRO);
      }
      else {
        reg_.unev = cdr(reg_.unev);
        if (reg_.unev.is_nil()) {
          GOTO(Label::AFTER_APPLY_MACRO);
        }
        auto expr = car(reg_.unev);
        reg_.expr = expr;
        reg_.cont = Label::AFTER_EVAL_AND_EXPR;
        GOTO(Label::EVAL);
      }
    }
    case Label::AFTER_EVAL_OR_EXPR: {
      PRINT_STEP();
      if (reg_.val.is_bool() && !reg_.val.to_bool()) {
        reg_.unev = cdr(reg_.unev);
        if (reg_.unev.is_nil()) {
          reg_.val = Cell::bool_false();
          GOTO(Label::AFTER_APPLY_MACRO);
        }
        auto expr = car(reg_.unev);
        reg_.expr = expr;
        reg_.cont = Label::AFTER_EVAL_OR_EXPR;
        GOTO(Label::EVAL);
      }
      GOTO(Label::AFTER_APPLY_MACRO);
    }
    case Label::AFTER_EVAL_CALL_WITH_VALUES_PRODUCER: {
      PRINT_STEP();
      auto consumer = car(reg_.unev);
      SPDLOG_INFO("consumer: {}", consumer);
      SPDLOG_INFO("consumer args: {}", reg_.val);
      // FIXME
      // hack
      if (reg_.val.is_num()) {
        reg_.val = cons(reg_.val, nil);
      }
      auto expr = list(new Symbol("apply"), consumer, list(quote, reg_.val));
      SPDLOG_INFO("expr: {}", expr);
      reg_.expr = expr;
      reg_.cont = Label::AFTER_APPLY_PROC;
      GOTO(Label::EVAL);
    }
    default: {
      SPDLOG_ERROR("Unsupported pos: {}", pos_);
      PSCM_THROW_EXCEPTION("Unsupported pos: " + to_string(pos_));
    }
    }
  }
}

Label Evaluator::eval_map_expr(Label default_pos) {
  auto proc = reg_.proc;
  auto lists = reg_.unev;
  SPDLOG_INFO("proc: {}", proc);
  SPDLOG_INFO("lists: {}", lists);
  PSCM_ASSERT(proc.is_proc() || proc.is_func());
  PSCM_ASSERT(lists.is_pair());
  auto len = list_length(lists);
  if (len == 0) {
    PSCM_THROW_EXCEPTION("ERROR args of proc: " + proc.to_string());
  }

  if (len == 1) {
    auto list1 = car(lists);
    if (list1.is_nil()) {
      reg_.val = reg_.argl;
      PSCM_POP_STACK(cont);
      return reg_.cont;
    }
    reg_.expr = list(proc, list(quote, car(list1)));
    reg_.unev = list(cdr(list1));
  }
  else if (len == 2) {
    auto list1 = car(lists);
    auto list2 = cadr(lists);
    if (list1.is_nil() && list2.is_nil()) {
      reg_.val = reg_.argl;
      PSCM_POP_STACK(cont);
      return reg_.cont;
    }
    reg_.expr = list(proc, list(quote, car(list1)), list(quote, car(list2)));
    reg_.unev = list(cdr(list1), cdr(list2));
  }
  else {
    PSCM_THROW_EXCEPTION("not support now");
  }
  reg_.proc = proc;
  reg_.cont = default_pos;
  return Label::EVAL;
}

std::ostream& operator<<(std::ostream& out, const Evaluator::Register& reg) {
  out << "expr: " << reg.expr;
  out << ", ";
  out << "env: " << reg.env;
  out << ", ";
  out << "proc: " << reg.proc;
  out << ", ";
  out << "argl: " << reg.argl;
  out << ", ";
  out << "cont: " << reg.cont;
  out << ", ";
  out << "val: " << reg.val;
  out << ", ";
  out << "unev: " << reg.unev;
  return out;
}

std::ostream& operator<<(std::ostream& out, const Evaluator::RegisterType& reg) {
  switch (reg) {
  case Evaluator::reg_expr:
    return out << "expr";
    break;
  case Evaluator::reg_env:
    return out << "env";
    break;
  case Evaluator::reg_proc:
    return out << "proc";
    break;
  case Evaluator::reg_argl:
    return out << "argl";
    break;
  case Evaluator::reg_cont:
    return out << "cont";
    break;
  case Evaluator::reg_val:
    return out << "val";
    break;
  case Evaluator::reg_unev:
    return out << "unev";
    break;
  }
  return out;
}

std::string Evaluator::Register::to_string() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

bool Evaluator::Stack::empty() const {
  if (!expr.empty()) {
    return false;
  }
  if (!env.empty()) {
    return false;
  }
  if (!proc.empty()) {
    return false;
  }
  if (!argl.empty()) {
    return false;
  }
  if (!cont.empty()) {
    return false;
  }
  if (!val.empty()) {
    return false;
  }
  if (!unev.empty()) {
    return false;
  }
  return true;
}

std::string Evaluator::Stack::to_string() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

template <typename T>
void print_list(std::ostream& out, const std::vector<T>& l) {
  if (l.empty()) {
    out << "[]";
  }
  else {
    out << "[";
    for (int i = 0; i < l.size() - 1; ++i) {
      out << l.at(i);
      out << ", ";
    }
    out << l.back();
    out << "]";
  }
}

std::ostream& operator<<(std::ostream& out, const Evaluator::Stack& stack) {
  out << "expr: ";
  print_list(out, stack.expr);
  out << ", ";
  out << "env: ";
  print_list(out, stack.env);
  out << ", ";
  out << "proc: ";
  print_list(out, stack.proc);
  out << ", ";
  out << "argl: ";
  print_list(out, stack.argl);
  out << ", ";
  out << "cont: ";
  print_list(out, stack.cont);
  out << ", ";
  out << "val: ";
  print_list(out, stack.val);
  out << ", ";
  out << "unev: ";
  print_list(out, stack.unev);
  return out;
}
} // namespace pscm
