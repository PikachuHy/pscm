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
#include "pscm/Procedure.h"
#include "pscm/Scheme.h"
#include "pscm/Str.h"
#include "pscm/Symbol.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
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

Cell add(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto ret = new Number(0);
  auto arg = car(args);
  while (!arg.is_nil()) {
    auto num = arg.to_number();
    PSCM_ASSERT(num);
    ret->inplace_add(*num);
    args = cdr(args);
    if (args.is_nil()) {
      break;
    }
    arg = car(args);
  }
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

Cell builtin_not(Cell args) {
  PSCM_ASSERT(args.is_pair());
  if (args.is_nil()) {
    PSCM_THROW_EXCEPTION("wrong-number-of-args: " + args.to_string());
  }
  auto arg = car(args);
  PSCM_ASSERT(arg.is_bool());
  return arg.to_bool() ? Cell::bool_false() : Cell::bool_true();
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

Cell newline(Cell args) {
  PSCM_ASSERT(args.is_nil());
  std::cout << std::endl;
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
  PSCM_ASSERT(args.is_pair());
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

Cell assv(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto obj = car(args);
  auto alist = cadr(args);
  while (!alist.is_nil()) {
    auto item = car(alist);
    auto key = car(item);
    if (key == obj) {
      return item;
    }
    alist = cdr(alist);
  }
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

Cell make_vector(Cell args) {
  Cell::Vec v;
  while (!args.is_nil()) {
    v.push_back(car(args));
    args = cdr(args);
  }
  return Cell(new Cell::Vec(v));
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

Cell reverse_argl(Cell argl) {
  auto p = cons(nil, nil);
  auto args = argl;
  while (!args.is_nil()) {
    p->second = cons(car(args), p->second);
    args = cdr(args);
  }
  return p->second;
}

Evaluator::Evaluator() {
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
    if (step_ > 5000) {
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
      auto proc = car(reg_.unev);
      auto args = cadr(reg_.unev);
      PSCM_ASSERT(proc.is_sym());
      PSCM_ASSERT(args.is_sym());
      proc = reg_.env->get(proc.to_symbol());
      args = reg_.env->get(args.to_symbol());
      SPDLOG_INFO("apply proc: {}", proc);
      SPDLOG_INFO("apply args: {}", args);
      reg_.expr = cons(proc, cons(cons(quote, args), nil));
      reg_.cont = Label::AFTER_APPLY_MACRO;
      GOTO(Label::EVAL);
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
    case Label::APPLY_LET: {
      PRINT_STEP();
      reg_.cont = Label::AFTER_APPLY_MACRO;
      reg_.expr = expand_let(reg_.unev);
      GOTO(Label::EVAL);
    }
    case Label::APPLY_LET_STAR: {
      PRINT_STEP();
      reg_.cont = Label::AFTER_APPLY_MACRO;
      reg_.expr = expand_let_star(reg_.unev);
      GOTO(Label::EVAL);
    }
    case Label::APPLY_LETREC: {
      PRINT_STEP();
      reg_.cont = Label::AFTER_APPLY_MACRO;
      reg_.expr = expand_letrec(reg_.unev);
      GOTO(Label::EVAL);
    }
    case Label::APPLY_CASE: {
      PRINT_STEP();
      reg_.cont = Label::AFTER_APPLY_MACRO;
      reg_.expr = expand_case(reg_.unev);
      GOTO(Label::EVAL);
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
    case Label::APPLY_FOR_EACH: {
      PRINT_STEP();
      auto proc = car(reg_.unev);
      auto list1 = cadr(reg_.unev);
      proc = reg_.env->get(proc.to_symbol());
      list1 = reg_.env->get(list1.to_symbol());
      SPDLOG_INFO("proc: {}", proc);
      PSCM_ASSERT(proc.is_proc());
      PSCM_ASSERT(list1.is_pair());
      if (list1.is_nil()) {
        reg_.val = Cell::none();
        PSCM_POP_STACK(cont);
        GOTO(reg_.cont);
      }
      else {
        reg_.expr = cons(proc, cons(car(list1), nil));
        reg_.proc = proc;
        reg_.unev = cdr(list1);
        reg_.cont = Label::AFTER_EVAL_FOR_EACH_FIRST_EXPR;
        GOTO(Label::EVAL);
      }
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
      if (reg_.unev.is_nil()) {
        reg_.val = Cell::none();
        PSCM_POP_STACK(cont);
        GOTO(reg_.cont);
      }
      else {
        reg_.expr = cons(reg_.proc, cons(car(reg_.unev), nil));
        reg_.unev = cdr(reg_.unev);
        reg_.cont = Label::AFTER_EVAL_FOR_EACH_FIRST_EXPR;
        GOTO(Label::EVAL);
      }
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
      // hack
      if (reg_.val.is_num()) {
        reg_.val = cons(reg_.val, nil);
      }
      auto expr = cons(consumer, reg_.val);
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