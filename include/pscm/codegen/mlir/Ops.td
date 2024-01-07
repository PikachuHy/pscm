#ifndef PSCM_OPS
#define PSCM_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PSCM_Dialect : Dialect {
    let name = "pscm";
    let cppNamespace = "::pscm";
    // let hasConstantMaterializer = 1;
    // let useDefaultTypePrinterParser = 1;
}

class PSCM_Op<string mnemonic, list<Trait> traits = []> :
    Op<PSCM_Dialect, mnemonic, traits>;

def PSCM_Type : AnyTypeOf<[I64]>;

def ConstantOp : PSCM_Op<"constant",
    [Pure]> {
    let summary = "constant";
    let description = [{
        Constant operation
    }];
    let arguments = (ins I64Attr:$value);
    let results = (outs I64);

    // let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "int":$value)>
    ];

    let hasVerifier = 1;

}

def AddOp : PSCM_Op<"add",
    [Pure]> {
    let summary = "addition operation";
    let description = [{
        The "add" operation performs addidtion
    }];

    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I64);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
    
}

def FuncOp : PSCM_Op<"func", [DeclareOpInterfaceMethods<CallableOpInterface>, FunctionOpInterface, IsolatedFromAbove]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let regions = (region AnyRegion:$body);
    let builders = [OpBuilder<(ins
        "StringRef":$name, "FunctionType":$type,
        CArg<"llvm::ArrayRef<NamedAttribute>", "{}">:$attrs
    )>];
    let extraClassDeclaration = [{
        llvm::ArrayRef<mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }
        llvm::ArrayRef<mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
    }];
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

def ReturnOp : PSCM_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
    let arguments = (ins Variadic<PSCM_Type>:$input);
    let assemblyFormat = "($input^ `:` type($input))? attr-dict";
    let builders = [
        OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
    ];
    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];
    let hasVerifier = 1;
}


def PrintOp : PSCM_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  let arguments = (ins I64:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}
#endif
