#ifndef PSCM_OPS
#define PSCM_OPS

def PSCM_Dialect : Dialect {
    let name = "pscm";
    let cppNamespace = "::mlir::toy"
    let hasConstantMaterializer = 1;
    let useDefaultTypePrinterParser = 1;
}

class PSCM_Op<string mnemonic, list<Trait> traits = []> :
    Op<PSCM_Dialect, mnemonic, traits>;

def PSCM_Type : AnyTypeOf<[F64Tensor]>;

def ConstantOp : PSCM_Op<"constant",
    [ConstantLike, Pure]> {
    let summary = "constant";
    let description = [{
        Constant operation
    }]
    let arguments = (ins F64ElemetsAttr:$value);
    let results = (outs F64Tensor)

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "DenseElementAttr":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>,
        OpBuilder<(ins "double":$value)>
    ];

    let hasVerifier = 1;

    let hasFolder = 1;
}

def AddOp : PSCM_Op<"add",
    [Pure]> {
    let summary = "addition operation";
    let description = [{
        The "add" operation performs addidtion
    }];

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ]
    
}
#endif