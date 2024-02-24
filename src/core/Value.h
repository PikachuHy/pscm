#pragma once
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

namespace pscm::core {

class Array {
public:
  int64_t size;
  int64_t *data;
};
class BooleanType;
class IntegerType;
class ArrayType;

class Type {
public:
  [[nodiscard]] static BooleanType *get_boolean_type();
  [[nodiscard]] static IntegerType *get_integer_type();
  [[nodiscard]] static ArrayType *get_integer_array_type();
  [[nodiscard]] virtual std::string to_string() const = 0;
};

class BooleanType : public Type {
public:
  [[nodiscard]] std::string to_string() const override;
};

class IntegerType : public Type {
  [[nodiscard]] std::string to_string() const override;
};

class ArrayType : public Type {
public:
  explicit ArrayType(Type *element_type)
      : element_type_(element_type) {
  }

  [[nodiscard]] const Type *element_type() const {
    return element_type_;
  }

  [[nodiscard]] std::string to_string() const override;

private:
  Type *element_type_;
};

class PrototypeAST;
class FunctionAST;
class Procedure;
class Evaluator;

struct CodegenContext {
  llvm::LLVMContext& llvm_ctx;
  llvm::Module& llvm_module;
  llvm::IRBuilder<>& builder;
  Evaluator& evaluator;
  std::unordered_map<std::string, PrototypeAST *> func_proto_map;
  std::unordered_map<std::string, llvm::Value *> named_values_map;
  std::unordered_map<std::string, Procedure *> proc_map;

  [[nodiscard]] llvm::Function *get_function(const std::string& name);
  [[nodiscard]] llvm::FunctionCallee get_malloc();
  [[nodiscard]] llvm::StructType *get_array();
};
class ExprAST;

class AST {
public:
  [[nodiscard]] virtual llvm::Value *codegen(CodegenContext& ctx) = 0;
  [[nodiscard]] std::pair<llvm::Function *, const Type *>
  instance_function(CodegenContext& ctx, const std::string& callee, const std::vector<const Type *>& arg_type_list,
                    const Type *return_type = nullptr);
};

class ExprAST : public AST {
public:
  [[nodiscard]] virtual const Type *type() const = 0;
};

class Value {
public:
  [[nodiscard]] virtual std::string to_string() const = 0;

  virtual ~Value() = default;
};

class BooleanValue : public Value {
public:
};

class TrueValue final
    : public BooleanValue
    , public ExprAST {
private:
  TrueValue() = default;

public:
  static const TrueValue *instance() {
    static TrueValue value;
    return &value;
  }

  [[nodiscard]] std::string to_string() const override {
    return "#t";
  }

  [[nodiscard]] llvm::Value *codegen(CodegenContext& ctx) override;

  const Type *type() const override;
};

class FalseValue final
    : public BooleanValue
    , public ExprAST {
private:
  FalseValue() = default;

public:
  static const FalseValue *instance() {
    static FalseValue value;
    return &value;
  }

  [[nodiscard]] std::string to_string() const override {
    return "#f";
  }

  [[nodiscard]] llvm::Value *codegen(CodegenContext& ctx) override;

  const Type *type() const override;
};

class SymbolValue final : public Value {
public:
  explicit SymbolValue(std::string value)
      : value_(std::move(value)) {
  }

  [[nodiscard]] std::string to_string() const override {
    return value_;
  }

private:
  std::string value_;
};

class VariableExprAST : public ExprAST {
public:
  VariableExprAST(SymbolValue *sym, const Type *type)
      : value_(sym)
      , type_(type) {
  }

  [[nodiscard]] llvm::Value *codegen(CodegenContext& ctx) override;

  std::string name() const {
    return value_->to_string();
  }

  const Type *type() const override;

private:
  SymbolValue *value_;
  const Type *type_;
};

class StringValue final : public Value {
public:
  explicit StringValue(std::string value)
      : value_(std::move(value)) {
  }

  [[nodiscard]] std::string to_string() const override {
    std::vector<char> s;
    s.reserve(value_.size() + 2);
    s.push_back('"');
    for (auto ch : value_) {
      if (ch == '"') {
        s.push_back('\\');
      }
      s.push_back(ch);
    }
    s.push_back('"');
    return { s.begin(), s.end() };
  }

private:
  std::string value_;
};

class NumberValue : public Value {};

class IntegerValue final
    : public NumberValue
    , public ExprAST {
public:
  explicit IntegerValue(int64_t value)
      : value_(value) {
  }

  static IntegerValue *zero() {
    static IntegerValue zero(0);
    return &zero;
  }

  [[nodiscard]] std::string to_string() const override {
    return std::to_string(value_);
  }

  [[nodiscard]] int64_t value() const {
    return value_;
  }

  [[nodiscard]] llvm::Value *codegen(CodegenContext& ctx) override;

  const Type *type() const override;

private:
  int64_t value_;
};

class ListValue : public Value {
public:
  explicit ListValue(std::vector<Value *> value_list)
      : value_(std::move(value_list)) {
  }

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] const std::vector<Value *> value() const {
    return value_;
  }

private:
  std::vector<Value *> value_;
};

class DottedListValue : public Value {
public:
  DottedListValue(std::vector<Value *> value_list, Value *value)
      : value1_(std::move(value_list))
      , value2_(value) {
  }

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] const std::vector<Value *>& value1() const {
    return value1_;
  }

  [[nodiscard]] const Value *value2() const {
    return value2_;
  }

private:
  std::vector<Value *> value1_;
  Value *value2_;
};

class ArrayExprAST
    : public Value
    , public ExprAST {
public:
  explicit ArrayExprAST(std::vector<ExprAST *> value)
      : value_(std::move(value)) {
  }

  [[nodiscard]] llvm::Value *codegen(CodegenContext& ctx) override;

  [[nodiscard]] const std::vector<ExprAST *>& value() const {
    return value_;
  }

  [[nodiscard]] std::size_t size() const {
    return value_.size();
  }

  [[nodiscard]] std::string to_string() const override;
  const Type *type() const override;

private:
  std::vector<ExprAST *> value_;
};

class BinaryExprAST : public ExprAST {
public:
  BinaryExprAST(SymbolValue *op, ExprAST *lhs, ExprAST *rhs)
      : op_(op)
      , lhs_(lhs)
      , rhs_(rhs) {
  }

  llvm::Value *codegen(CodegenContext& ctx) override;
  [[nodiscard]] const Type *type() const override;

private:
  SymbolValue *op_;
  ExprAST *lhs_;
  ExprAST *rhs_;
};

class CallExprAST : public ExprAST {
public:
  CallExprAST(std::string callee, std::vector<ExprAST *> args, std::vector<const Type *> types)
      : callee_(std::move(callee))
      , args_(std::move(args))
      , types_(std::move(types))
      , return_type_(nullptr) {
  }

  [[nodiscard]] llvm::Value *codegen(CodegenContext& ctx) override;
  [[nodiscard]] const Type *type() const override;

private:
  std::string callee_;
  std::vector<ExprAST *> args_;
  std::vector<const Type *> types_;
  const Type *return_type_;
};

class IfExprAST : public ExprAST {
public:
  IfExprAST(ExprAST *cond, ExprAST *then_stmt, ExprAST *else_stmt)
      : cond_(cond)
      , then_stmt_(then_stmt)
      , else_stmt_(else_stmt) {
  }

  void add_else_if(ExprAST *cond, ExprAST *then_stmt) {
    else_if_.emplace_back(cond, then_stmt);
  }

  llvm::Value *codegen(CodegenContext& ctx) override;
  const Type *type() const override;

private:
  ExprAST *cond_;
  ExprAST *then_stmt_;
  std::vector<std::pair<ExprAST *, ExprAST *>> else_if_;
  ExprAST *else_stmt_;
};

class MapExprAST : public ExprAST {
public:
  MapExprAST(std::string callee, ExprAST *args)
      : callee_(std::move(callee))
      , args_(args) {
  }

  [[nodiscard]] llvm::Value *codegen(CodegenContext& ctx) override;
  [[nodiscard]] const Type *type() const override;

private:
  std::string callee_;
  ExprAST *args_;
};

class PrototypeAST {
public:
  PrototypeAST(std::string name, std::vector<std::string> args, std::vector<const Type *> arg_type_list,
               const Type *return_type = nullptr)
      : name_(std::move(name))
      , args_(std::move(args))
      , arg_type_list_(std::move(arg_type_list))
      , return_type_(return_type) {
  }

  [[nodiscard]] llvm::Function *codegen(CodegenContext& ctx);

  [[nodiscard]] const std::string& name() const {
    return name_;
  }

private:
  std::string name_;
  std::vector<std::string> args_;
  std::vector<const Type *> arg_type_list_;
  const Type *return_type_;
};

class FunctionAST : public AST {
public:
  FunctionAST(PrototypeAST *proto, AST *body)
      : proto_(proto)
      , body_(body) {
  }

  [[nodiscard]] llvm::Function *codegen(CodegenContext& ctx);

private:
  PrototypeAST *proto_;
  AST *body_;
};
} // namespace pscm::core
