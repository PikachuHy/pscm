#include "Evaluator.h"
#include "Mangler.h"
#include "Procedure.h"
#include "SymbolTable.h"
#include "Value.h"

#include <pscm/common_def.h>

namespace pscm::core {
PSCM_INLINE_LOG_DECLARE("pscm.core.Evaluator");

template <typename Op>
struct BinaryOperation {
  template <typename LeftType, typename RightType>
  static typename Op::ReturnType *eval(LeftType *lhs, RightType *rhs) {
    return Op()(lhs, rhs);
  }
};

template <typename A, typename B>
struct GetReturnType;

template <>
struct GetReturnType<IntegerValue, IntegerValue> {
  using value = IntegerValue;
};

template <typename LeftType, typename RightType>
class AddOp {
public:
  using ReturnType = typename GetReturnType<LeftType, RightType>::value;

  ReturnType *operator()(LeftType *lhs, RightType *rhs) {
    auto ret = lhs->value() + rhs->value();
    return new ReturnType(ret);
  }
};

template <typename LeftType, typename RightType>
class MinusOp {
public:
  using ReturnType = typename GetReturnType<LeftType, RightType>::value;

  ReturnType *operator()(LeftType *lhs, RightType *rhs) {
    auto ret = lhs->value() - rhs->value();
    return new ReturnType(ret);
  }
};

class EvaluatorImpl {
public:
  EvaluatorImpl()
      : sym_table_(new SymbolTable()) {
    auto sym_car = new SymbolValue("car");
    sym_table_->put(sym_car, new Procedure(sym_car, {}, {}, nullptr));
  }

  ~EvaluatorImpl() {
    delete sym_table_;
  }

  AST *eval(Value *expr) {
    if (auto p = dynamic_cast<ListValue *>(expr); p) {
      return eval(p);
    }
    //    if (auto p = dynamic_cast<DottedListValue *>(expr); p) {
    //      return eval(p);
    //    }
    if (auto p = dynamic_cast<SymbolValue *>(expr); p) {
      return eval(p);
    }
    //    if (auto p = dynamic_cast<BooleanValue *>(expr); p) {
    //      return p;
    //    }
    if (auto p = dynamic_cast<IntegerValue *>(expr); p) {
      return p;
    }
    //    if (auto p = dynamic_cast<StringValue *>(expr); p) {
    //      return p;
    //    }
    if (expr) {
      PSCM_THROW_EXCEPTION("Unsupported type: " + expr->to_string());
    }
    PSCM_UNIMPLEMENTED();
  };

  AST *eval(ListValue *expr) {
    PSCM_ASSERT(expr);
    auto value_list = expr->value();
    PSCM_ASSERT(!value_list.empty());
    if (auto p = dynamic_cast<SymbolValue *>(value_list[0]); p) {
      if (auto value = sym_table_->lookup(p); value) {
        if (auto f = dynamic_cast<Procedure *>(value); f) {
          std::vector<ExprAST *> args;
          std::vector<const Type *> arg_types;
          for (int i = 1; i < value_list.size(); ++i) {
            auto arg = eval(value_list[i]);
            if (auto func_arg = dynamic_cast<ExprAST *>(arg); func_arg) {
              args.push_back(func_arg);
              auto type = func_arg->type();
              arg_types.push_back(type);
            }
            else {
              PSCM_UNIMPLEMENTED();
            }
          }
          auto call = new CallExprAST(f->name()->to_string(), args, arg_types);
          return call;
        }
      }
      /*
            (cond ((> 3 2) 100)
                  ((< 3 2) 200))
       */
      if (p->to_string() == "cond") {
        return create_cond(value_list);
      }
      if (p->to_string() == "map") {
        return create_map(value_list);
      }
      if (p->to_string() == "quote") {
        auto value = value_list[1];
        if (auto list = dynamic_cast<ListValue *>(value); list) {
          std::vector<ExprAST *> array_value_list;
          for (int i = 0; i < list->value().size(); ++i) {
            auto array_value = eval(list->value()[i]);
            if (auto array_value_expr_ast = dynamic_cast<ExprAST *>(array_value); array_value_expr_ast) {
              array_value_list.push_back(array_value_expr_ast);
            }
            else {
              PSCM_UNIMPLEMENTED();
            }
          }
          return new ArrayExprAST(std::move(array_value_list));
        }
      }

      std::vector<ExprAST *> operands;
      operands.reserve(value_list.size() - 1);
      for (int i = 1; i < value_list.size(); ++i) {
        auto arg = eval(value_list[i]);
        if (auto ast = dynamic_cast<ExprAST *>(arg); ast) {
          operands.push_back(ast);
        }
        else {
          PSCM_UNIMPLEMENTED();
        }
      }
      auto sym = p->to_string();
      if (sym == ">" || sym == "<" || sym == "=") {
        if (operands.size() < 2) {
          PSCM_THROW_EXCEPTION("Invalid arguments: " + expr->to_string() + ", require at least 2");
        }
        auto lhs = operands[0];
        auto rhs = operands[1];
        return new BinaryExprAST(p, lhs, rhs);
      }
      if (sym == "+" || sym == "-") {
        int start_index = 0;
        ExprAST *ret = IntegerValue::zero();
        if (operands.empty()) {
          return ret;
        }
        if (operands.size() == 1) {
          if (sym == "+") {
            if (auto ret_int = dynamic_cast<ExprAST *>(operands[0]); ret_int) {
              return ret_int;
            }
            else {
              PSCM_UNIMPLEMENTED();
            }
          }
        }
        else {
          ret = operands[0];
          start_index = 1;
        }
        if (sym == "-") {
          if (auto ret_int = dynamic_cast<ExprAST *>(operands[0]); ret_int) {
            if (operands.size() == 1) {
              ret = new BinaryExprAST(p, ret, ret_int);
              return ret;
            }
            else {
              ret = ret_int;
            }
          }
          else {
            PSCM_UNIMPLEMENTED();
          }
          start_index = 1;
        }
        for (int i = start_index; i < operands.size(); ++i) {
          auto arg = operands[i];
          ret = new BinaryExprAST(p, ret, arg);
        }
        return ret;
      }
    }
    return nullptr;
  }

  Value *eval(DottedListValue *expr) {
    return nullptr;
  }

  ExprAST *eval(SymbolValue *expr) {
    return sym_table_->lookup(expr);
  }

  IfExprAST *create_cond(const std::vector<Value *>& value_list) {
    std::vector<ExprAST *> list;
    std::vector<std::pair<ExprAST *, ExprAST *>> cond_stmt_list;
    ExprAST *else_stmt = nullptr;
    for (int i = 1; i < value_list.size(); ++i) {
      if (auto cond_stmt = dynamic_cast<ListValue *>(value_list[i]); cond_stmt) {
        PSCM_ASSERT(!cond_stmt->value().empty());
        auto cond = eval(cond_stmt->value()[0]);
        auto then_value = eval(cond_stmt->value()[1]);
        if (!cond) {
          if (auto maybe_else = dynamic_cast<SymbolValue *>(cond_stmt->value()[0]); maybe_else) {
            if (maybe_else->to_string() == "else") {
              auto else_expr_ast = dynamic_cast<ExprAST *>(then_value);
              PSCM_ASSERT(else_expr_ast);
              else_stmt = else_expr_ast;
            }
            else {
              PSCM_THROW_EXCEPTION("Invalid symbol: " + maybe_else->to_string());
            }
          }
        }
        else if (auto cond_expr_ast = dynamic_cast<ExprAST *>(cond); cond_expr_ast) {
          if (auto else_sym = dynamic_cast<SymbolValue *>(cond); else_sym) {
            PSCM_UNIMPLEMENTED();
          }
          else {
            auto then_value_expr_ast = dynamic_cast<ExprAST *>(then_value);
            PSCM_ASSERT(then_value_expr_ast);
            cond_stmt_list.emplace_back(cond_expr_ast, then_value_expr_ast);
          }
        }
        else {
          PSCM_THROW_EXCEPTION("Invalid cond statement: " + cond_stmt->to_string());
        }
      }
      else {
        PSCM_UNIMPLEMENTED();
      }
    }
    IfExprAST *if_expr_ast = nullptr;
    for (auto& [cond, then_value] : cond_stmt_list) {
      if (if_expr_ast) {
        if_expr_ast->add_else_if(cond, then_value);
      }
      else {
        if_expr_ast = new IfExprAST(cond, then_value, else_stmt);
      }
    }
    return if_expr_ast;
  }

  // (map abs '(4 -5 6))
  MapExprAST *create_map(const std::vector<Value *>& value_list) {
    PSCM_ASSERT(value_list.size() == 3);
    auto f = eval(value_list[1]);
    auto args = eval(value_list[2]);
    auto proc = dynamic_cast<Procedure *>(f);
    auto proc_name = proc->name()->to_string();
    if (auto expr_ast = dynamic_cast<ExprAST *>(args); expr_ast) {
      return new MapExprAST(proc_name, expr_ast);
    }
    PSCM_UNIMPLEMENTED();
  }

  SymbolTable *sym_table_;
};

Evaluator::Evaluator()
    : impl_(new EvaluatorImpl()) {
}

Evaluator::~Evaluator() {
  delete impl_;
}

AST *Evaluator::eval(Value *expr) {
  return impl_->eval(expr);
}

void Evaluator::add_proc(SymbolValue *sym, ExprAST *value) {
  impl_->sym_table_->put(sym, value);
}

void Evaluator::push_symbol_table() {
  auto table = new SymbolTable(impl_->sym_table_);
  impl_->sym_table_ = table;
}

void Evaluator::pop_symbol_table() {
  auto parent = impl_->sym_table_->parent();
  PSCM_ASSERT(parent);
  delete impl_->sym_table_;
  impl_->sym_table_ = parent;
}

void Evaluator::add_sym(SymbolValue *sym, ExprAST *value) {
  impl_->sym_table_->put(sym, value);
}

} // namespace pscm::core