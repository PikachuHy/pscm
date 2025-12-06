# pscm cc

pscm cc 是重新写的一套PikachuHy's Scheme的实现。
这个版本最大的特性是学习类似Guile 1.8的方式，基于`longjmp/setjmp`实现了continuation。
代码规模大概在2000行。

cc后缀表示使用C++编写。实际上，这版本实现是使用尽可能少的C++特性来实现，目前用到了模板、重载、inline函数和lambda表达式等。

另外一个特殊点是，开发过程中基于llvm lit工具封装了pscm-lit做测试，可以直接写scm文件，然后检查求值结果是不是正确，非常方便。

parser部分复用了老的parser，后面会重写。

::: warning
pscm 依然处于非常简陋的状态
:::

## 设计目标

- 利用有限的C++特性实现一个精简版本的Guile 1.8，只保留驱动TeXmacs时的必要的特性。

## 整体设计

### 类型系统

- 所有的类型都是`struct SCM`(内部是`void*`)。通过`cast<Type>(scm)`可以将`struct SCM`类型转换为具体的`Type`类型；通过`wrap(type)`可以将具体的`Type`类型转换为`struct SCM`(数字类型暂不支持)。
- 支持的数据类型：`NONE`, `NIL`, `LIST`, `PROC`, `CONT`, `FUNC`, `NUM`, `BOOL`, `SYM`, `STR`

### 数据结构

- List直接用链表实现(`SCM_List`)，不再是pair套pair，操作时更加简便。
- 环境(Environment)使用链表结构，支持词法作用域(通过parent指针实现作用域链)。

### 求值器

- eval部分，用了goto，在部分情况下可以减小栈的深度，做到类似尾递归的效果。
- 支持的特殊形式：`define`, `lambda`, `let`/`let*`/`letrec`(通过宏展开实现), `set!`, `quote`, `if`, `cond`(支持`else`和`=>`语法), `do`, `call/cc`/`call-with-current-continuation`, `for-each`

### Continuation实现

- 基于`setjmp/longjmp`实现continuation，通过栈复制的方式保存和恢复执行上下文。

### C/C++函数注册

- C/C++代码注册到Scheme中采用类似Guile 1.8的接口，方便后续在TeXmacs中使用。
- 支持固定参数数量的函数(1个或2个参数)和泛型函数(如`+`运算符)

### 内置函数

当前实现的内置函数包括：
- 类型检查：`procedure?`, `boolean?`, `null?`, `pair?`
- 列表操作：`car`, `cdr`, `cadr`
- 数字运算：`+`, `-`, `*`, `=`, `<`, `>`, `<=`, `>=`, `negative?`
- 相等性判断：`eq?`, `eqv?`
- 关联列表：`assv`

## 已知限制

1. **内存管理**：当前没有实现垃圾回收(GC)机制，所有分配的内存都不会被释放，可能导致内存泄漏。
2. **错误处理**：错误处理机制较为简陋，多数情况下直接`exit(1)`，缺少优雅的错误恢复机制。
3. **性能**：环境查找使用线性搜索，符号比较使用`strcmp`，在大规模代码中可能成为性能瓶颈。

## TODO

- 后续会使用AI来重写代码，目前实现的部分完全是手搓的。
- parser重写
- 实现垃圾回收机制
- 优化环境查找性能(考虑使用哈希表)
- 改进错误处理机制
