# pscm cc

![logo](/logo.png)

pscm cc 是 PikachuHy's Scheme 的 C++ 实现版本，代码规模约 7000+ 行。该版本参考 Guile 1.8，基于 `setjmp/longjmp` 实现了 continuation 支持。

::: warning
pscm 依然处于非常简陋的状态
:::

## 设计目标

利用有限的 C++ 特性实现一个精简版本的 Guile 1.8，保留驱动 TeXmacs 所需的必要特性。

> 📋 **详细功能列表和开发规划**：请参考 [pscm_cc 功能现状与开发规划](pscm_cc_roadmap.md)

## 核心架构

### 类型系统

- **统一类型**：所有值都是 `struct SCM`（内部为 `void*`）
- **类型转换**：通过 `cast<Type>(scm)` 转换为具体类型，通过 `wrap(type)` 包装为 `SCM`
- **支持类型**：16 种数据类型（`NIL`, `LIST`, `NUM`, `FLOAT`, `RATIO`, `CHAR`, `STR`, `SYM`, `BOOL`, `PROC`, `FUNC`, `CONT`, `MACRO`, `HASH_TABLE`, `VECTOR`）
- **源位置跟踪**：每个 AST 节点携带可选的源位置信息（文件名、行号、列号），用于错误报告

### 数据结构

- **列表**：使用 `SCM_List` 链表实现，支持 dotted pair
- **环境**：链表结构，通过 `parent` 指针实现词法作用域链
- **哈希表**：链式哈希表，支持 `eq?`、`eqv?`、`equal?` 三种比较方式
- **向量**：固定长度数组，支持随机访问

### 求值器

- **尾递归优化**：使用 `goto` 减少栈深度
- **模块化设计**：每个特殊形式独立文件，统一接口在 `eval.h` 声明
- **支持的特殊形式**：`define`, `lambda`, `if`, `cond`, `case`, `and`, `or`, `begin`, `let`/`let*`/`letrec`, `do`, `for-each`, `map`, `quote`, `quasiquote`, `apply`, `call/cc`, `call-with-values`, `dynamic-wind` 等

### Continuation 实现

- **机制**：基于 `setjmp/longjmp`，通过栈复制保存和恢复执行上下文
- **动态 wind**：支持 `dynamic-wind`，在 continuation 跳转时执行 before/after thunk
- **打印格式**：`#<continuation@地址>`

### 解析器

- **实现**：从零实现的递归下降解析器
- **语法支持**：完整的 Scheme 语法（数字、符号、字符串、布尔值、列表、引号、准引用、点对、注释）
- **特殊处理**：支持 `1+` 和 `1-` 作为符号（避免被解析为数字和运算符）
- **错误报告**：包含文件名、行号、列号的清晰错误信息

### C/C++ 函数注册

兼容 Guile 1.8 接口，支持三种注册方式：

- **固定参数**：`scm_define_function(name, req, opt, rst, func_ptr)`
- **泛型函数**：`scm_define_generic_function(name, func_ptr, init_val)`（如 `+`、`*`）
- **可变参数**：`scm_define_vararg_function(name, func_ptr)`（如 `list`、`apply`）

### 内置函数

支持丰富的内置函数，包括：

- **类型检查**：`procedure?`, `boolean?`, `null?`, `pair?`, `char?`, `number?` 等
- **列表操作**：`car`, `cdr`, `cons`, `list`, `append`, `set-car!`, `set-cdr!` 等
- **数字运算**：`+`, `-`, `*`, `/`, `expt`, `abs` 等（支持整数、浮点数、分数混合运算）
- **字符串操作**：`string-length`, `make-string`, `string-ref`, `string-set!`, `display`, `write` 等
- **向量操作**：`make-vector`, `vector-length`, `vector-ref`, `vector-set!` 等
- **哈希表**：完整的哈希表操作集（创建、设置、获取、删除、遍历）
- **其他**：`gensym`, `not`, `eval`, `equal?`, `eq?`, `eqv?` 等

## 代码组织

### 模块划分

- **核心**：`pscm.h`（类型定义）、`eval.h`（求值接口）、`eval.cc`（主求值器）
- **特殊形式**：`do.cc`, `cond.cc`, `case.cc`, `map.cc`, `apply.cc`, `quasiquote.cc`, `macro.cc`, `let.cc`, `for_each.cc`, `values.cc`
- **内置函数**：`predicate.cc`, `number.cc`, `list.cc`, `string.cc`, `char.cc`, `eq.cc`, `alist.cc`, `hash_table.cc`, `vector.cc`
- **基础设施**：`parse.cc`（解析器）、`print.cc`（打印）、`continuation.cc`（continuation）、`environment.cc`（环境）、`source_location.cc`（源位置）

### 错误处理

- 统一使用 `eval_error` 函数
- 错误信息包含完整源位置
- 宏展开时自动传播源位置
- **调用栈追踪**：自动追踪表达式求值路径，错误时显示完整调用栈（最多 20 层）
- **增强的错误报告**：包含表达式类型、值、源位置和完整求值上下文

## 已知限制

1. **内存管理**：未实现垃圾回收（GC），所有分配的内存不会释放，存在内存泄漏风险
2. **错误处理**：多数情况下直接 `exit(1)`，缺少优雅的错误恢复机制
3. **性能**：环境查找使用线性搜索（O(n)），符号比较使用 `memcmp`
4. **缺失功能**：端口系统、模块系统、Guile API 兼容层等

## 下一步计划

### 高优先级
- 实现垃圾回收机制
- 端口系统（文件 I/O 支持）
- Guile API 兼容层（C API 接口）

### 中优先级
- 模块系统（代码组织）
- 错误处理机制（异常捕获，当前已有调用栈追踪）
- 更多 Scheme 标准特性（`delay`/`force` 等）
