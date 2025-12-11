# pscm cc

pscm cc 是 PikachuHy's Scheme 的 C++ 实现版本，代码规模约 6400 行。该版本参考 Guile 1.8，基于 `setjmp/longjmp` 实现了 continuation 支持。

::: warning
pscm 依然处于非常简陋的状态
:::

## 设计目标

利用有限的 C++ 特性实现一个精简版本的 Guile 1.8，保留驱动 TeXmacs 所需的必要特性。

## 核心特性

### 类型系统

- 统一类型：所有值都是 `struct SCM`（内部为 `void*`）
- 类型转换：通过 `cast<Type>(scm)` 转换为具体类型，通过 `wrap(type)` 包装为 `SCM`
- 支持类型：`NONE`, `NIL`, `LIST`, `PROC`, `CONT`, `FUNC`, `NUM`, `FLOAT`, `CHAR`, `BOOL`, `SYM`, `STR`, `MACRO`, `HASH_TABLE`, `RATIO`, `VECTOR`
- 源位置：每个 AST 节点携带可选的源位置信息（文件名、行号、列号），用于错误报告

### 数据结构

- **列表**：使用 `SCM_List` 链表实现，支持 dotted pair
- **环境**：链表结构，通过 `parent` 指针实现词法作用域链
- **哈希表**：链式哈希表，支持 `eq?`、`eqv?`、`equal?` 三种比较方式
- **向量**：固定长度数组，支持随机访问

### 求值器

- **尾递归优化**：使用 `goto` 减少栈深度
- **特殊形式**：
  - 定义：`define`, `define-macro`, `lambda`, `set!`
  - 控制流：`if`, `cond`（支持 `else` 和 `=>`）, `begin`
  - 作用域：`let`/`let*`/`letrec`（通过宏展开实现）
  - 循环：`do`, `for-each`, `map`
  - 引用：`quote`, `quasiquote`
  - 函数应用：`apply`
  - Continuation：`call/cc`/`call-with-current-continuation`
  - 多值：`call-with-values`, `values`
  - 动态控制：`dynamic-wind`
- **代码组织**：每个特殊形式独立文件，统一接口在 `eval.h` 声明

### Continuation 实现

- **机制**：基于 `setjmp/longjmp`，通过栈复制保存和恢复执行上下文
- **动态 wind**：支持 `dynamic-wind`，在 continuation 跳转时执行 before/after thunk
- **打印格式**：`#<continuation@地址>`

### 解析器

- **实现**：从零实现的递归下降解析器
- **语法支持**：数字、符号、字符串、布尔值、列表、引号、准引用、点对、注释
- **特殊处理**：支持 `1+` 和 `1-` 作为符号（避免被解析为数字和运算符）
- **错误报告**：包含文件名、行号、列号的清晰错误信息
- **源位置**：解析时自动记录每个 AST 节点的源位置

### C/C++ 函数注册

兼容 Guile 1.8 接口，支持三种注册方式：

- **固定参数**：`scm_define_function(name, req, opt, rst, func_ptr)`
- **泛型函数**：`scm_define_generic_function(name, func_ptr, init_val)`（如 `+`、`*`）
- **可变参数**：`scm_define_vararg_function(name, func_ptr)`（如 `list`、`apply`）

### 内置函数

#### 类型检查
`procedure?`, `boolean?`, `null?`, `pair?`, `char?`, `number?`

#### 列表操作
`car`, `cdr`, `cadr`, `cddr`, `caddr`, `cons`, `list`, `append`, `list-head`, `list-tail`, `last-pair`, `set-car!`, `set-cdr!`

#### 数字运算
- **算术**：`+`（泛型，可变参数）、`-`（可变参数，支持一元取反）、`*`（泛型，可变参数，支持分数）、`/`（可变参数，返回分数或浮点数）、`expt`（幂运算）、`abs`
- **比较**：`=`, `<`, `>`, `<=`, `>=`, `negative?`
- **类型提升**：整数、浮点数、分数混合运算自动提升
- **分数**：除法自动返回分数（如 `(/ 1 3)` → `1/3`），使用 GCD 自动简化

#### 字符操作
`char?`, `char->integer`, `integer->char`

#### 字符串操作
`string-length`, `make-string`（可变参数，支持填充字符）, `string-ref`, `string-set!`, `display`, `newline`（支持可选端口）

#### 相等性判断
`eq?`, `eqv?`, `equal?`

#### 关联列表
`assv`, `assoc`, `acons`, `assoc-ref`, `assoc-set!`, `assq-set!`, `assoc-remove!`

#### 哈希表
- **创建**：`make-hash-table`（支持可选容量）
- **设置**：`hash-set!`, `hashq-set!`, `hashv-set!`（分别使用 `equal?`、`eq?`、`eqv?`）
- **获取**：`hash-ref`, `hashq-ref`, `hashv-ref`
- **句柄**：`hash-get-handle`, `hashq-get-handle`, `hashv-get-handle`（返回 `(key . value)`）
- **创建句柄**：`hash-create-handle!`, `hashq-create-handle!`, `hashv-create-handle!`
- **删除**：`hash-remove!`, `hashq-remove!`, `hashv-remove!`
- **遍历**：`hash-fold`

#### 其他
`gensym`, `not`, `eval`

## 代码组织

### 模块划分

- **核心**：`pscm.h`（类型定义）、`eval.h`（求值接口）、`eval.cc`（主求值器）
- **特殊形式**：`do.cc`, `cond.cc`, `map.cc`, `apply.cc`, `quasiquote.cc`, `macro.cc`, `let.cc`, `for_each.cc`, `values.cc`
- **内置函数**：`predicate.cc`, `number.cc`, `list.cc`, `string.cc`, `char.cc`, `eq.cc`, `alist.cc`, `hash_table.cc`
- **基础设施**：`parse.cc`（解析器）、`print.cc`（打印）、`continuation.cc`（continuation）、`environment.cc`（环境）、`source_location.cc`（源位置）

### 错误处理

- 统一使用 `eval_error` 函数
- 错误信息包含完整源位置
- 宏展开时自动传播源位置

## 已知限制

1. **内存管理**：未实现垃圾回收（GC），所有分配的内存不会释放，存在内存泄漏风险
2. **错误处理**：多数情况下直接 `exit(1)`，缺少优雅的错误恢复机制
3. **性能**：环境查找使用线性搜索（O(n)），符号比较使用 `memcmp`

## TODO

### 高优先级
- 实现垃圾回收机制
- 集成哈希表到环境查找（将查找复杂度降至 O(1)）

### 中优先级
- 改进错误处理机制（支持异常捕获）
- 支持更多 Scheme 标准特性（`case`, `and`, `or` 等）

### 低优先级
- 优化解析器性能
- 性能分析和优化
- Guile 1.8 API 兼容层（为 TeXmacs 集成做准备）
