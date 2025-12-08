# pscm cc

pscm cc 是重新写的一套PikachuHy's Scheme的实现。
这个版本最大的特性是学习类似Guile 1.8的方式，基于`longjmp/setjmp`实现了continuation。
代码规模约4800行。

cc后缀表示使用C++编写。实际上，这版本实现是使用尽可能少的C++特性来实现，目前用到了模板、重载、inline函数和lambda表达式等。

另外一个特殊点是，开发过程中基于llvm lit工具封装了pscm-lit做测试，可以直接写scm文件，然后检查求值结果是不是正确，非常方便。

~~parser部分复用了老的parser，后面会重写。~~
parser部分已经完全重写(by AI🤣)，实现了一个从零开始的递归下降解析器，不再依赖旧的解析器实现。

::: warning
pscm 依然处于非常简陋的状态
:::

## 设计目标

- 利用有限的C++特性实现一个精简版本的Guile 1.8，只保留驱动TeXmacs时的必要的特性。

## 整体设计

### 类型系统

- 所有的类型都是`struct SCM`(内部是`void*`)。通过`cast<Type>(scm)`可以将`struct SCM`类型转换为具体的`Type`类型；通过`wrap(type)`可以将具体的`Type`类型转换为`struct SCM`(数字类型暂不支持)。
- 支持的数据类型：`NONE`, `NIL`, `LIST`, `PROC`, `CONT`, `FUNC`, `NUM`, `FLOAT`, `CHAR`, `BOOL`, `SYM`, `STR`, `MACRO`, `HASH_TABLE`
- 每个AST节点都携带可选的源位置信息（文件名、行号、列号），用于错误报告和调试
- 支持整数和浮点数混合运算，自动进行类型提升

### 数据结构

- List直接用链表实现(`SCM_List`)，不再是pair套pair，操作时更加简便。
- 环境(Environment)使用链表结构，支持词法作用域(通过parent指针实现作用域链)。
- 哈希表(`SCM_HashTable`)使用链式哈希表实现，支持三种比较方式（`eq?`、`eqv?`、`equal?`），采用C风格数据结构，避免使用C++标准库容器。

### 求值器

- eval部分，用了goto，在部分情况下可以减小栈的深度，做到类似尾递归的效果。
- 支持的特殊形式：
  - **定义和绑定**：`define`, `define-macro`, `lambda`, `set!`
  - **控制流**：`if`, `cond`(支持`else`和`=>`语法)
  - **作用域**：`let`/`let*`/`letrec`(通过宏展开实现)
  - **循环**：`do`, `for-each`, `map`
  - **引用**：`quote`, `quasiquote`
  - **函数应用**：`apply`
  - **Continuation**：`call/cc`/`call-with-current-continuation`
- 代码结构高度模块化：
  - 每个特殊形式都有独立的处理函数（`eval_define`, `eval_lambda`, `eval_if`, `eval_cond`, `eval_do`, `eval_map`, `eval_apply`等）
  - 特殊形式已拆分到独立文件：`do.cc`（do特殊形式）、`cond.cc`（cond特殊形式）、`map.cc`（map特殊形式）、`apply.cc`（apply特殊形式）、`quasiquote.cc`（准引用）、`macro.cc`（宏展开）、`let.cc`（let系列宏展开）、`for_each.cc`（for-each特殊形式）
  - 公共接口统一在`eval.h`中声明
- 统一的错误处理机制：
  - 使用`eval_error`函数提供清晰的错误信息
  - 错误信息包含源位置（文件名、行号、列号）
  - 宏展开时自动传播源位置信息，便于定位错误

### Continuation实现

- 基于`setjmp/longjmp`实现continuation，通过栈复制的方式保存和恢复执行上下文。

### 解析器

- 完全从零实现的递归下降解析器，不依赖任何外部解析库。
- 支持完整的Scheme语法：数字、符号、字符串、布尔值、列表、引号、准引用、点对、注释等。
- 提供清晰的错误报告，包含文件名、行号和列号信息。
- 支持文件解析和字符串解析两种模式。
- 解析时自动记录每个AST节点的源位置信息，用于后续错误报告。

### C/C++函数注册

- C/C++代码注册到Scheme中采用类似Guile 1.8的接口，方便后续在TeXmacs中使用。
- 支持三种函数注册方式：
  - 固定参数数量的函数：`scm_define_function`（指定必需参数数量）
  - 泛型函数：`scm_define_generic_function`（支持可变参数，如`+`、`*`运算符）
  - 可变参数函数：`scm_define_vararg_function`（如`list`、`append`、`-`、`make-string`、`apply`）

### 内置函数

当前实现的内置函数包括：
- **类型检查**：`procedure?`, `boolean?`, `null?`, `pair?`, `char?`
- **列表操作**：`car`, `cdr`, `cadr`, `cddr`, `caddr`, `cons`, `list`, `append`
- **数字运算**：
  - 算术运算：`+`（泛型，支持可变参数）、`-`（可变参数，支持一元取反和多元减法）、`*`（泛型，支持可变参数）、`abs`
  - 比较运算：`=`, `<`, `>`, `<=`, `>=`, `negative?`
  - 支持整数和浮点数混合运算，自动类型提升
- **字符操作**：`char?`, `char->integer`, `integer->char`
- **字符串操作**：`string-length`, `make-string`（可变参数，支持可选填充字符）, `string-ref`, `string-set!`, `display`
- **相等性判断**：`eq?`, `eqv?`, `equal?`
- **关联列表**：`assv`, `assoc`, `acons`, `assoc-ref`, `assoc-set!`, `assq-set!`, `assoc-remove!`
- **哈希表操作**：
  - 创建：`make-hash-table`（可变参数，支持可选容量参数）
  - 设置值：`hash-set!`, `hashq-set!`, `hashv-set!`（分别使用`equal?`、`eq?`、`eqv?`比较）
  - 获取值：`hash-ref`, `hashq-ref`, `hashv-ref`
  - 获取句柄：`hash-get-handle`, `hashq-get-handle`, `hashv-get-handle`（返回`(key . value)`对）
  - 创建句柄：`hash-create-handle!`, `hashq-create-handle!`, `hashv-create-handle!`
  - 删除：`hash-remove!`, `hashq-remove!`, `hashv-remove!`
  - 遍历：`hash-fold`（支持对哈希表所有条目进行折叠操作）
- **其他**：`gensym`, `not`, `eval`

## 代码优化

### 已完成优化

1. **代码模块化**：
   - 所有特殊形式处理函数都已提取为独立函数（`eval_define`, `eval_lambda`, `eval_if`, `eval_cond`, `eval_do`, `eval_for_each`, `eval_map`, `eval_apply`等）
   - 特殊形式已拆分到独立文件：
     - `do.cc`：do特殊形式
     - `cond.cc`：cond特殊形式（支持`else`和`=>`语法）
     - `map.cc`：map特殊形式
     - `apply.cc`：apply特殊形式
     - `quasiquote.cc`：准引用处理
     - `macro.cc`：宏展开和定义
     - `let.cc`：let/let*/letrec宏展开
   - 内置函数按功能分类到独立文件：
     - `predicate.cc`：类型检查谓词
     - `number.cc`：数字运算（支持整数和浮点数）
     - `list.cc`：列表操作
     - `string.cc`：字符串操作
     - `char.cc`：字符操作
     - `eq.cc`：相等性判断
     - `alist.cc`：关联列表操作
     - `environment.cc`：环境管理
     - `function.cc`：函数应用
     - `procedure.cc`：过程应用
     - `continuation.cc`：continuation实现
     - `source_location.cc`：源位置跟踪
     - `hash_table.cc`：哈希表实现（支持eq?/eqv?/equal?三种比较方式）
   - 公共接口统一在`eval.h`中声明，遵循"先include pscm.h再include其他头文件"的规范
   - 提取了公共辅助函数（`lookup_symbol`, `apply_procedure`, `count_list_length`, `update_do_variables`等）
   - 统一使用`make_list_dummy()`辅助函数创建dummy list，减少代码重复
   - 哈希表实现中提取了通用辅助函数（`validate_and_get_bucket_idx`, `update_entry_value`, `insert_entry_to_bucket`等），消除代码重复

2. **错误处理改进**：
   - 统一使用`eval_error`函数进行错误报告
   - 错误信息包含完整的源位置信息（文件名、行号、列号）
   - 宏展开时自动传播源位置信息，确保错误能追溯到原始代码位置
   - 使用`report_arg_mismatch`统一处理参数不匹配错误
   - 改进的错误消息格式，便于调试

3. **代码质量提升**：
   - 使用静态局部变量实现单例模式，减少全局状态
   - 统一符号单例的创建方式，使用宏减少代码重复
   - 移除C++标准库依赖，完全使用C风格函数（`printf`/`fprintf`替代`std::cout`）
   - 改进REPL输入处理，支持更大的输入缓冲区
   - 统一的include顺序规范，提高代码一致性
   - 哈希表实现采用C风格数据结构，避免使用`std::map`、`std::set`、`std::unordered_map`等C++标准库容器
   - 优化符号比较，使用`memcmp`替代`strncmp`提升性能

4. **解析器重写**：
   - 完全从零实现，不依赖旧的解析器
   - 支持完整的Scheme语法特性（包括准引用）
   - 清晰的错误报告机制
   - 自动记录每个AST节点的源位置信息

5. **新特性支持**：
   - 实现了`define-macro`，支持用户定义宏
   - 实现了`map`函数，支持多列表参数
   - 实现了`quasiquote`，支持准引用语法
   - 实现了`apply`函数，支持动态函数应用
   - 实现了浮点数类型（`FLOAT`）和字符类型（`CHAR`）
   - 实现了整数和浮点数混合运算，自动类型提升
   - 实现了完整的字符串操作函数集
   - 实现了字符和整数之间的转换
   - 实现了关联列表的完整操作集（增删改查）
   - 实现了`equal?`函数，支持深度相等性比较
   - 实现了`eval`函数，支持动态求值
   - 实现了源位置跟踪系统，提升错误报告质量
   - 实现了完整的哈希表功能集，支持三种比较方式（`eq?`、`eqv?`、`equal?`），兼容Guile 1.8的哈希表API
   - 实现了`for-each`特殊形式，支持多列表参数，与`map`类似但用于副作用操作
   - 实现了`display`函数，用于输出Scheme对象

## 已知限制

1. **内存管理**：当前没有实现垃圾回收(GC)机制，所有分配的内存都不会被释放，可能导致内存泄漏。这是最高优先级的改进项。

2. **错误处理**：错误处理机制虽然已改进，包含详细的源位置信息，但多数情况下仍直接`exit(1)`，缺少优雅的错误恢复机制（如异常捕获）。

3. **性能**：
   - 环境查找使用线性搜索（O(n)复杂度），符号比较使用`memcmp`，在大规模代码中可能成为性能瓶颈
   - 哈希表已实现，可用于优化环境查找性能（待集成）

4. **代码组织**：代码已高度模块化，特殊形式和内置函数都已拆分到独立文件，提高了可维护性。

## TODO

### 高优先级
- **实现垃圾回收机制**：这是当前最紧迫的任务，解决内存泄漏问题
- **集成哈希表到环境查找**：哈希表已实现，需要将其集成到环境查找中，将查找复杂度从O(n)降至O(1)

### 中优先级
- **改进错误处理机制**：支持异常处理和错误恢复（如`catch`/`throw`）
- **代码进一步模块化**：继续拆分`eval.cc`，提高可维护性
- **支持更多Scheme标准特性**：`case`, `and`, `or`, `begin`等

### 低优先级
- **优化解析器性能**：支持更大的文件
- **性能分析和优化**：识别热点代码，针对性优化
- **Guile 1.8 API兼容层**：为TeXmacs集成做准备
