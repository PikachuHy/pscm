# pscm core

pscm core 是重新写的一套PikachuHy's Scheme实现。
从 core 这个名字也可以看出，我希望它足够的小，但有足够的有用。

## Why

之前做的实现，在支持 Continuation 和 Unicode 后，变得比较难演进。
所以我想再写一个更小的实现，不再拘泥 R5RS 标准。
我对它的定位更多是一个前缀表达式的语言。

## 设计目标

- 学习 LLVM
- 将 Scheme 编译到 LLVM IR，然后用 LLVM JIT运行。(也可以考虑直接出一个可执行文件)
- 强类型

::: warning
pscm 依然处于非常简陋的状态
:::

当前的版本

以下特性已实现编译到 LLVM IR，并使用 LLVM JIT运行。

- 基于 LLVM 17 开发
- 支持整型 `integer`，和整型数组 `array<integer>`
- 支持的操作： `map`, `+`, `-`, `>`, `<`
- 支持简单的函数定义和调用。函数在调用时才做类型检查和代码生成。只定义，不调用不会有任何的效果。

## 整体设计

不再使用之前基于 `Cell` 的实现，现在使用大量的指针和类继承。
基本类型是 `Value`，然后会把各种 `Value` 编译到 `AST`。
这里相当于前端的语法是前缀表达式，后面的流程和传统的 `C/C++` 差不多。

Codegen 时，把各种 `AST` 生成 LLVM IR，接着就可以使用 LLVM JIT执行LLVM IR。

```
Cell -> Value -> AST -> LLVM IR
```

目前依旧复用了老的pscm的parser，暂时没有重新写一套，所以刚拿到的依旧是 `Cell`。

