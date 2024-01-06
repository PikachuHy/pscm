#include "pscm/codegen/mlir/Passes.h"
using namespace mlir;
#include "pscm/codegen/mlir/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "Dialect.cpp.inc"
using namespace pscm;

static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser& parser, mlir::OperationState& result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, 2) || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc, result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

static void printBinaryOp(mlir::OpAsmPrinter& printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(), [=](Type type) {
        return type == resultType;
      })) {
    printer << resultType;
    return;
  }

  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

void ConstantOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, int value) {
}

mlir::LogicalResult ConstantOp::verify() {
  return mlir::success();
}

void AddOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(builder.getI64Type());
  state.addOperands({ lhs, rhs });
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result) {
  return parseBinaryOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter& p) {
  printBinaryOp(p, *this);
}

void FuncOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result) {
  auto buildFuncType = [](mlir::Builder& builder, llvm::ArrayRef<mlir::Type> argTypes,
                          llvm::ArrayRef<mlir::Type> results, mlir::function_interface_impl::VariadicFlag,
                          std::string&) {
    return builder.getFunctionType(argTypes, results);
  };

  return mlir::function_interface_impl::parseFunctionOp(parser, result, false, getFunctionTypeAttrName(result.name),
                                                        buildFuncType, getArgAttrsAttrName(result.name),
                                                        getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter& p) {
  mlir::function_interface_impl::printFunctionOp(p, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
                                                 getResAttrsAttrName());
}

mlir::Region *FuncOp::getCallableRegion() {
  return &getBody();
}

llvm::ArrayRef<mlir::Type> FuncOp::getCallableResults() {
  return getFunctionType().getResults();
}

ArrayAttr FuncOp::getCallableArgAttrs() {
  return getArgAttrs().value_or(nullptr);
}

ArrayAttr FuncOp::getCallableResAttrs() {
  return getResAttrs().value_or(nullptr);
}

mlir::LogicalResult ReturnOp::verify() {
  auto function = cast<FuncOp>((*this)->getParentOp());

  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  const auto& results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values (" << getNumOperands()
                         << ") as the enclosing function (" << results.size() << ")";

  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  if (inputType == resultType || llvm::isa<mlir::UnrankedTensorType>(inputType) ||
      llvm::isa<mlir::UnrankedTensorType>(resultType))
    return mlir::success();

  return emitError() << "type of return operand (" << inputType << ") doesn't match function result type ("
                     << resultType << ")";
}

void PSCMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "Ops.cpp.inc"