#include "Value.h"
#include "Evaluator.h"
#include "Mangler.h"
#include "Procedure.h"
#include <pscm/common_def.h>
#include <sstream>

namespace pscm::core {

llvm::Function *CodegenContext::get_function(const std::string& name) {
  if (auto f = llvm_module.getFunction(name); f) {
    return f;
  }
  auto it = func_proto_map.find(name);
  if (it != func_proto_map.end()) {
    return it->second->codegen(*this);
  }
  return nullptr;
}

llvm::FunctionCallee CodegenContext::get_malloc() {
  auto malloc_func =
      llvm_module.getOrInsertFunction("malloc", llvm::FunctionType::get(llvm::Type::getInt8PtrTy(llvm_ctx),
                                                                        { llvm::Type::getInt64Ty(llvm_ctx) }, false));
  return malloc_func;
}

llvm::StructType *CodegenContext::get_array() {
  auto array_struct = llvm::StructType::getTypeByName(llvm_ctx, "Array");
  if (array_struct) {
    return array_struct;
  }
  array_struct = llvm::StructType::create(llvm_ctx, "Array");
  array_struct->setBody({ llvm::Type::getInt64Ty(llvm_ctx), llvm::Type::getInt64PtrTy(llvm_ctx) });
  return array_struct;
}

std::string ListValue::to_string() const {
  if (value_.empty()) {
    return "()";
  }
  std::stringstream ss;
  ss << "(";
  for (int i = 0; i < value_.size() - 1; ++i) {
    ss << value_[i]->to_string();
    ss << " ";
  }
  ss << value_.back()->to_string();
  ss << ")";
  return ss.str();
}

std::string DottedListValue::to_string() const {
  PSCM_INLINE_LOG_DECLARE("pscm.core.DottedListValue");
  PSCM_ASSERT(!value1_.empty());
  std::stringstream ss;
  ss << "(";
  for (int i = 0; i < value1_.size() - 1; ++i) {
    ss << value1_[i]->to_string();
    ss << " ";
  }
  ss << value1_.back()->to_string();

  ss << " ";
  ss << ".";
  ss << " ";
  ss << value2_->to_string();
  ss << ")";
  return ss.str();
}

llvm::Value *IntegerValue::codegen(CodegenContext& ctx) {
  return llvm::ConstantInt::get(ctx.llvm_ctx, llvm::APInt(64, value_));
}

const Type *IntegerValue::type() const {
  return Type::get_integer_type();
}

llvm::Value *TrueValue::codegen(CodegenContext& ctx) {
  return llvm::ConstantInt::getTrue(ctx.llvm_ctx);
}

const Type *TrueValue::type() const {
  return Type::get_boolean_type();
}

llvm::Value *FalseValue::codegen(CodegenContext& ctx) {
  return llvm::ConstantInt::getFalse(ctx.llvm_ctx);
}

const Type *FalseValue::type() const {
  return Type::get_boolean_type();
}

llvm::Value *BinaryExprAST::codegen(CodegenContext& ctx) {
  PSCM_INLINE_LOG_DECLARE("pscm.core.BinaryExprAST");
  PSCM_ASSERT(lhs_);
  PSCM_ASSERT(rhs_);
  auto lhs = lhs_->codegen(ctx);
  auto rhs = rhs_->codegen(ctx);
  if (!lhs || !rhs) {
    return nullptr;
  }
  if (op_->to_string() == "+") {
    return ctx.builder.CreateAdd(lhs, rhs, "add_tmp");
  }
  else if (op_->to_string() == "-") {
    return ctx.builder.CreateSub(lhs, rhs, "sub_tmp");
  }
  else if (op_->to_string() == "<") {
    auto cmptmp = ctx.builder.CreateICmpSLT(lhs, rhs, "lt_cmp_tmp");
    return cmptmp;
  }
  else if (op_->to_string() == ">") {
    auto cmptmp = ctx.builder.CreateICmpSGT(lhs, rhs, "gt_cmp_tmp");
    return cmptmp;
  }
  else if (op_->to_string() == "=") {
    return ctx.builder.CreateICmpEQ(lhs, rhs, "eq_cmp_tmp");
  }
  else {
    PSCM_UNIMPLEMENTED();
  }
}

const Type *BinaryExprAST::type() const {
  PSCM_INLINE_LOG_DECLARE("pscm.core.BinaryExprAST");
  auto op = op_->to_string();
  if (op == "<" || op == ">" || op == "=") {
    return Type::get_boolean_type();
  }
  if (op == "+" || op == "-") {
    PSCM_ASSERT(lhs_->type() == rhs_->type());
    return lhs_->type();
  }
  PSCM_UNIMPLEMENTED();
}

llvm::Type *convert_pscm_type_to_llvm_type(CodegenContext& ctx, const Type *type) {
  PSCM_INLINE_LOG_DECLARE("pscm.core.convert_pscm_type_to_llvm_type");
  if (auto array_type = dynamic_cast<const ArrayType *>(type); array_type) {
    if (auto element_type = dynamic_cast<const IntegerType *>(array_type->element_type()); element_type) {
      return llvm::PointerType::get(ctx.get_array(), 0);
    }
    else {
      PSCM_UNIMPLEMENTED();
    }
  }
  else if (auto integer_type = dynamic_cast<const IntegerType *>(type); integer_type) {
    return llvm::Type::getInt64Ty(ctx.llvm_ctx);
  }
  else {
    PSCM_UNIMPLEMENTED();
  }
}

llvm::Function *PrototypeAST::codegen(CodegenContext& ctx) {
  std::vector<llvm::Type *> func_args;
  func_args.reserve(args_.size());
  for (auto arg_type : arg_type_list_) {
    func_args.push_back(convert_pscm_type_to_llvm_type(ctx, arg_type));
  }
  llvm::Type *func_return_type;
  if (return_type_) {
    func_return_type = convert_pscm_type_to_llvm_type(ctx, return_type_);
  }
  else if (!arg_type_list_.empty()) {
    func_return_type = convert_pscm_type_to_llvm_type(ctx, arg_type_list_.front());
  }
  else {
    func_return_type = llvm::Type::getInt64Ty(ctx.llvm_ctx);
  }

  llvm::FunctionType *func_type = llvm::FunctionType::get(func_return_type, func_args, false);
  llvm::Function *func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, name_, ctx.llvm_module);
  unsigned idx = 0;
  for (auto& arg : func->args()) {
    arg.setName(args_[idx]);
    idx++;
  }
  return func;
}

llvm::Function *FunctionAST::codegen(CodegenContext& ctx) {
  ctx.func_proto_map[proto_->name()] = proto_;
  auto func = ctx.get_function(proto_->name());
  if (!func) {
    return nullptr;
  }
  llvm::BasicBlock *bb = llvm::BasicBlock::Create(ctx.llvm_ctx, "entry", func);
  ctx.builder.SetInsertPoint(bb);
  ctx.named_values_map.clear();
  for (auto& arg : func->args()) {
    ctx.named_values_map[std::string(arg.getName())] = &arg;
  }
  if (auto ret = body_->codegen(ctx)) {
    ctx.builder.CreateRet(ret);
    llvm::verifyFunction(*func);
    return func;
  }
  func->eraseFromParent();
  return nullptr;
}

llvm::Value *VariableExprAST::codegen(CodegenContext& ctx) {
  PSCM_INLINE_LOG_DECLARE("pscm.core.VariableExprAST");
  auto value = ctx.named_values_map[this->value_->to_string()];
  if (value) {
    return value;
  }
  PSCM_THROW_EXCEPTION("Unknown symbol: " + this->value_->to_string());
}

const Type *VariableExprAST::type() const {
  return type_;
}

llvm::Value *CallExprAST::codegen(CodegenContext& ctx) {
  PSCM_INLINE_LOG_DECLARE("pscm.core.CallExprAST");
  std::vector<const Type *> arg_type_list;
  for (auto arg : args_) {
    arg_type_list.push_back(arg->type());
  }
  auto mangled_name = Mangler().mangle(callee_, arg_type_list);
  llvm::Function *callee = ctx.llvm_module.getFunction(mangled_name);
  if (!callee) {
    std::tie(callee, return_type_) = this->instance_function(ctx, callee_, arg_type_list, nullptr);
    PSCM_ASSERT(callee);
  }
  if (!return_type_) {
    if (callee->getFunction().getReturnType() == llvm::Type::getInt64Ty(ctx.llvm_ctx)) {
      return_type_ = Type::get_integer_type();
    }
    else {
      PSCM_UNIMPLEMENTED();
    }
  }
  if (callee->arg_size() != args_.size()) {
    PSCM_THROW_EXCEPTION("Incorrect # arguments passed: " + callee_);
  }
  std::vector<llvm::Value *> args;
  args.reserve(args_.size());
  for (auto& arg : args_) {
    args.push_back(arg->codegen(ctx));
  }
  auto calltmp = ctx.builder.CreateCall(callee, args, "calltmp");
  return calltmp;
}

const Type *CallExprAST::type() const {
  PSCM_INLINE_LOG_DECLARE("pscm.core.CallExprAST");
  PSCM_ASSERT(return_type_);
  return return_type_;
}

llvm::Value *IfExprAST::codegen(CodegenContext& ctx) {
  auto cond = cond_->codegen(ctx);
  if (!cond) {
    return nullptr;
  }
  //  llvm::errs() << *cond->getType() << "\n";
  //  cond = ctx.builder.CreateICmpEQ(cond, llvm::ConstantInt::get(ctx.llvm_ctx, llvm::APInt(1, 1)), "if_cond");
  auto func = ctx.builder.GetInsertBlock()->getParent();
  auto then_bb = llvm::BasicBlock::Create(ctx.llvm_ctx, "then", func);
  auto else_bb = llvm::BasicBlock::Create(ctx.llvm_ctx, "else");
  auto merge_bb = llvm::BasicBlock::Create(ctx.llvm_ctx, "if_cont");

  std::vector<llvm::BasicBlock *> else_if_cond_bb_list;
  std::vector<llvm::BasicBlock *> else_if_then_bb_list;
  std::vector<llvm::Value *> else_if_then_stmt_list;
  for (auto& [else_if_cond, then] : else_if_) {
    auto else_if_cond_bb = llvm::BasicBlock::Create(ctx.llvm_ctx, "else_if", func);
    auto else_if_then_bb = llvm::BasicBlock::Create(ctx.llvm_ctx, "else_if_then", func);
    else_if_cond_bb_list.push_back(else_if_cond_bb);
    else_if_then_bb_list.push_back(else_if_then_bb);
  }
  if (else_if_.empty()) {
    ctx.builder.CreateCondBr(cond, then_bb, else_bb);
  }
  else {
    ctx.builder.CreateCondBr(cond, then_bb, else_if_cond_bb_list[0]);
  }
  for (int i = 0; i < else_if_.size(); ++i) {
    auto else_if_cond = else_if_[i].first;
    auto else_if_then = else_if_[i].second;
    auto else_if_cond_bb = else_if_cond_bb_list[i];
    auto else_if_then_bb = else_if_then_bb_list[i];
    ctx.builder.SetInsertPoint(else_if_cond_bb);
    auto else_if_cond_stmt = else_if_cond->codegen(ctx);
    if (!else_if_cond_stmt) {
      return nullptr;
    }
    ctx.builder.SetInsertPoint(else_if_then_bb);
    auto else_if_then_stmt = else_if_then->codegen(ctx);
    if (!else_if_then_stmt) {
      return nullptr;
    }
    else_if_then_stmt_list.push_back(else_if_then_stmt);
    ctx.builder.CreateBr(merge_bb);
    ctx.builder.SetInsertPoint(else_if_cond_bb);
    if (i == else_if_.size() - 1) {
      ctx.builder.CreateCondBr(else_if_cond_stmt, else_if_then_bb, else_bb);
    }
    else {
      ctx.builder.CreateCondBr(else_if_cond_stmt, else_if_then_bb, else_if_cond_bb_list[i + 1]);
    }
  }

  ctx.builder.SetInsertPoint(then_bb);
  auto then_stmt = then_stmt_->codegen(ctx);
  if (!then_stmt) {
    return nullptr;
  }
  ctx.builder.CreateBr(merge_bb);
  then_bb = ctx.builder.GetInsertBlock();

  func->insert(func->end(), else_bb);
  ctx.builder.SetInsertPoint(else_bb);

  auto else_stmt = else_stmt_ ? else_stmt_->codegen(ctx) : llvm::ConstantInt::get(ctx.llvm_ctx, llvm::APInt(64, 0));
  if (!else_stmt) {
    return nullptr;
  }
  ctx.builder.CreateBr(merge_bb);

  func->insert(func->end(), merge_bb);
  ctx.builder.SetInsertPoint(merge_bb);
  llvm::PHINode *phi = ctx.builder.CreatePHI(llvm::Type::getInt64Ty(ctx.llvm_ctx), 2 + else_if_.size(), "if_tmp");
  phi->addIncoming(then_stmt, then_bb);
  for (int i = 0; i < else_if_.size(); ++i) {
    phi->addIncoming(else_if_then_stmt_list[i], else_if_then_bb_list[i]);
  }
  phi->addIncoming(else_stmt, else_bb);
  llvm::verifyFunction(*func);
  return phi;
}

const Type *IfExprAST::type() const {
  return then_stmt_->type();
}

llvm::Value *ArrayExprAST::codegen(CodegenContext& ctx) {
  auto malloc_func = ctx.get_malloc();
  auto array_struct = ctx.get_array();
  auto array_ptr = ctx.builder.CreateCall(malloc_func, { ctx.builder.getInt64(sizeof(Array)) }, "array_ptr");
  auto array_size_ptr = ctx.builder.CreateStructGEP(array_struct, array_ptr, 0, "array_size");
  auto array_data_placeholder = ctx.builder.CreateStructGEP(array_struct, array_ptr, 1, "array_data_placeholder");
  ctx.builder.CreateAlignedStore(llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx.llvm_ctx), value_.size()),
                                 array_size_ptr, llvm::MaybeAlign(8));

  auto array_data_ptr =
      ctx.builder.CreateCall(malloc_func, { ctx.builder.getInt64(value_.size() * 8) }, "array_data_ptr");
  for (size_t i = 0; i < value_.size(); ++i) {
    llvm::Value *ptr = ctx.builder.CreateGEP(llvm::Type::getInt64Ty(ctx.llvm_ctx), array_data_ptr,
                                             ctx.builder.getInt64(i), "array_data_" + std::to_string(i));
    ctx.builder.CreateAlignedStore(value_[i]->codegen(ctx), ptr, llvm::MaybeAlign(8));
  }
  ctx.builder.CreateAlignedStore(array_data_ptr, array_data_placeholder, llvm::MaybeAlign(8));
  return array_ptr;
}

std::string ArrayExprAST::to_string() const {
  return "array ast";
}

const Type *ArrayExprAST::type() const {
  return Type::get_integer_array_type();
}

llvm::Value *MapExprAST::codegen(CodegenContext& ctx) {
  PSCM_INLINE_LOG_DECLARE("pscm.core.MapExprAST");
  if (auto array = dynamic_cast<ArrayExprAST *>(args_); array) {
    std::vector<ExprAST *> value_list;
    for (int i = 0; i < array->size(); ++i) {
      auto call = new CallExprAST(callee_, { array->value()[i] }, { Type::get_integer_type() });
      value_list.push_back(call);
    }
    return ArrayExprAST(value_list).codegen(ctx);
  }
  else if (auto sym = dynamic_cast<VariableExprAST *>(args_); sym) {
    auto array_ptr = ctx.named_values_map[sym->name()];
    auto array_struct = ctx.get_array();

    auto malloc_func = ctx.get_malloc();
    auto ret_array_ptr = ctx.builder.CreateCall(malloc_func, { ctx.builder.getInt64(sizeof(Array)) }, "ret_array_ptr");
    auto ret_array_size_ptr = ctx.builder.CreateStructGEP(array_struct, ret_array_ptr, 0, "ret_array_size");
    auto ret_array_data_placeholder =
        ctx.builder.CreateStructGEP(array_struct, ret_array_ptr, 1, "ret_array_data_placeholder");

    auto array_size_ptr = ctx.builder.CreateStructGEP(array_struct, array_ptr, 0, "array_size_ptr");
    auto array_data_placeholder = ctx.builder.CreateStructGEP(array_struct, array_ptr, 1, "array_data_placeholder");
    auto array_data_ptr = ctx.builder.CreateAlignedLoad(llvm::Type::getInt64PtrTy(ctx.llvm_ctx), array_data_placeholder,
                                                        llvm::MaybeAlign(8), "array_data_ptr");
    auto array_size = ctx.builder.CreateAlignedLoad(llvm::Type::getInt64Ty(ctx.llvm_ctx), array_size_ptr,
                                                    llvm::MaybeAlign(8), "array_size");
    ctx.builder.CreateAlignedStore(array_size, ret_array_size_ptr, llvm::MaybeAlign(8));
    auto ret_array_data_memory_size = ctx.builder.CreateMul(array_size, ctx.builder.getInt64(8), "array_mem_size");
    auto ret_array_data_ptr = ctx.builder.CreateCall(malloc_func, { ret_array_data_memory_size }, "ret_array_data_ptr");
    ctx.builder.CreateAlignedStore(ret_array_data_ptr, ret_array_data_placeholder, llvm::MaybeAlign(8));

    auto func = ctx.builder.GetInsertBlock()->getParent();
    auto loop_cond_bb = llvm::BasicBlock::Create(ctx.llvm_ctx, "loop.cond", func);
    auto loop_body_bb = llvm::BasicBlock::Create(ctx.llvm_ctx, "loop.body", func);
    auto loop_inc_bb = llvm::BasicBlock::Create(ctx.llvm_ctx, "loop.inc", func);
    auto loop_end_bb = llvm::BasicBlock::Create(ctx.llvm_ctx, "loop.end", func);

    auto idx_ptr = ctx.builder.CreateAlloca(llvm::Type::getInt64Ty(ctx.llvm_ctx), nullptr, "idx_ptr");
    ctx.builder.CreateAlignedStore(llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx.llvm_ctx), 0), idx_ptr,
                                   llvm::MaybeAlign(8));
    ctx.builder.CreateBr(loop_cond_bb);

    ctx.builder.SetInsertPoint(loop_cond_bb);
    auto idx = ctx.builder.CreateAlignedLoad(llvm::Type::getInt64Ty(ctx.llvm_ctx), idx_ptr, llvm::MaybeAlign(8), "idx");

    auto cmptmp = ctx.builder.CreateICmpULT(idx, array_size, "cmptmp");
    ctx.builder.CreateCondBr(cmptmp, loop_body_bb, loop_end_bb);

    ctx.builder.SetInsertPoint(loop_body_bb);
    auto input_item_ptr =
        ctx.builder.CreateGEP(llvm::Type::getInt64PtrTy(ctx.llvm_ctx), array_data_ptr, idx, "input_item_ptr");
    auto input_item = ctx.builder.CreateAlignedLoad(llvm::Type::getInt64Ty(ctx.llvm_ctx), input_item_ptr,
                                                    llvm::MaybeAlign(8), "input_item");
    auto ret_output_item_ptr =
        ctx.builder.CreateGEP(llvm::Type::getInt64PtrTy(ctx.llvm_ctx), ret_array_data_ptr, idx, "output_item_ptr");
    auto callee = ctx.get_function(callee_);
    const Type *return_type = nullptr;
    if (!callee) {
      auto map_func_input_type = sym->type();
      if (auto array_type = dynamic_cast<const ArrayType *>(map_func_input_type); array_type) {
        std::tie(callee, return_type) = this->instance_function(ctx, callee_, { array_type->element_type() });
      }
      PSCM_ASSERT(callee);
    }
    std::vector<llvm::Value *> func_args;
    func_args.push_back(input_item);
    auto calltmp = ctx.builder.CreateCall(callee, func_args, "output_item");
    ctx.builder.CreateAlignedStore(calltmp, ret_output_item_ptr, llvm::MaybeAlign(8));

    ctx.builder.CreateBr(loop_inc_bb);

    ctx.builder.SetInsertPoint(loop_inc_bb);
    auto new_idx =
        ctx.builder.CreateAdd(idx, llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx.llvm_ctx), 1), "new_idx");
    ctx.builder.CreateAlignedStore(new_idx, idx_ptr, llvm::MaybeAlign(8));

    ctx.builder.CreateBr(loop_cond_bb);
    ctx.builder.SetInsertPoint(loop_end_bb);

    return ret_array_ptr;
  }
  else {
    PSCM_UNIMPLEMENTED();
  }
}

const Type *MapExprAST::type() const {
  return Type::get_integer_array_type();
}

std::pair<llvm::Function *, const Type *> AST::instance_function(CodegenContext& ctx, const std::string& callee_,
                                                                 const std::vector<const Type *>& args_,
                                                                 const Type *return_type) {
  PSCM_INLINE_LOG_DECLARE("pscm.core.instance_function");
  auto mangled_name = Mangler().mangle(callee_, args_);
  auto t = ctx.builder.GetInsertBlock();

  // instance func callee
  auto it = ctx.proc_map.find(callee_);
  if (it == ctx.proc_map.end()) {
    PSCM_THROW_EXCEPTION("Unknown symbol: " + callee_);
  }
  auto proc = it->second;

  std::vector<std::string> args;
  args.reserve(proc->args().size());
  for (auto& arg : proc->args()) {
    args.push_back(arg->to_string());
  }
  std::vector<AST *> value_to_codegen;
  value_to_codegen.reserve(proc->body().size());
  ctx.evaluator.push_symbol_table();
  for (int i = 0; i < args_.size(); ++i) {
    auto sym = proc->args()[i];
    auto arg = args_[i];
    ctx.evaluator.add_sym(sym, new VariableExprAST(sym, arg));
  }
  for (auto stmt : proc->body()) {
    auto value = ctx.evaluator.eval(stmt);
    value_to_codegen.push_back(value);
    if (auto expr_ast = dynamic_cast<ExprAST *>(value); expr_ast) {
      return_type = expr_ast->type();
    }
    else {
      PSCM_UNIMPLEMENTED();
    }
  }
  ctx.evaluator.pop_symbol_table();
  auto proto = new PrototypeAST(mangled_name, args, args_, return_type);
  auto func = new FunctionAST(proto, value_to_codegen[0]);
  func->codegen(ctx);
  auto callee = ctx.llvm_module.getFunction(mangled_name);
  PSCM_ASSERT(callee);
  ctx.builder.SetInsertPoint(t);
  return { callee, return_type };
}

BooleanType *Type::get_boolean_type() {
  static BooleanType type;
  return &type;
}

IntegerType *Type::get_integer_type() {
  static IntegerType type;
  return &type;
}

ArrayType *Type::get_integer_array_type() {
  static ArrayType type(get_integer_type());
  return &type;
}

std::string BooleanType::to_string() const {
  return "boolean";
}

std::string IntegerType::to_string() const {
  return "integer";
}

std::string ArrayType::to_string() const {
  std::stringstream ss;
  ss << "array";
  ss << "[";
  ss << element_type_->to_string();
  ss << "]";
  return ss.str();
}
} // namespace pscm::core