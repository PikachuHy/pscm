# pscm_cc Eval 实现分析

## 概述

本文档分析 pscm_cc 中 eval 的实现方式，对比 Guile 1.8 的实现，并讨论其优劣以及改进方向。

## pscm_cc 的实现

### 核心架构

pscm_cc 的 eval 实现位于 `src/c/eval.cc`，采用**直接求值**的方式，没有使用代码转换或记忆化（memoization）机制。

### 主要函数

#### 1. `eval_with_env` - 核心求值函数

```c
SCM *eval_with_env(SCM_Environment *env, SCM *ast) {
  // 1. 推入求值栈（用于错误追踪）
  push_eval_stack(ast);
  
  // 2. 保存当前上下文
  SCM *old_context = g_current_eval_context;
  g_current_eval_context = ast;
  
entry:
  // 3. 根据表达式类型分发处理
  if (!is_pair(ast)) {
    // 原子值：符号、数字、字符串等
    if (is_sym(ast)) {
      // 符号：查找环境中的绑定
      SCM_Symbol *sym = cast<SCM_Symbol>(ast);
      if (sym->data && sym->data[0] == ':') {
        // 关键字（以:开头）自求值
        return ast;
      }
      SCM *result = lookup_symbol(env, sym);
      return result;
    }
    return ast;  // 其他原子值自求值
  }
  
  // 4. 列表：可能是特殊形式或过程调用
  SCM_List *l = cast<SCM_List>(ast);
  
  if (is_sym(l->data)) {
    // 检查是否是宏调用
    SCM *val = scm_env_exist(env, sym);
    if (val && is_macro(val)) {
      // 宏展开
      SCM *expanded = expand_macro_call(env, macro, l->next, ast);
      ast = expand_macros(env, expanded);
      goto entry;  // 尾递归优化
    }
    
    // 处理特殊形式
    if (is_sym_val(l->data, "if")) {
      // if 处理
    }
    else if (is_sym_val(l->data, "lambda")) {
      // lambda 处理
    }
    // ... 其他特殊形式
    else {
      // 变量引用：查找并构建调用表达式
      auto new_list = make_list(val);
      new_list->next = l->next;
      ast = wrap(new_list);
      goto entry;  // 尾递归优化
    }
  }
  else if (is_cont(l->data)) {
    // continuation 调用
    scm_dynthrow(l->data, cont_arg);
  }
  else if (is_proc(l->data)) {
    // 过程调用
    auto proc = cast<SCM_Procedure>(l->data);
    SCM *result = apply_procedure(env, proc, l->next);
    return result;
  }
  // ...
}
```

**关键特性：**
- 使用 `goto entry` 实现尾递归优化，避免栈溢出
- 支持宏展开（在特殊形式检查之前）
- 统一的错误处理和上下文追踪
- 直接求值，没有代码转换

#### 2. 错误处理和调试支持

```c
// 求值栈追踪
struct EvalStackFrame {
  char *source_location;  // 源位置信息
  char *expr_str;         // 表达式字符串（未使用）
  EvalStackFrame *next;
};

static void push_eval_stack(SCM *expr) {
  // 只追踪有源位置的表达式，避免保存临时栈对象
  const char *loc = get_source_location_str(expr);
  if (!loc) return;  // 跳过临时表达式
  
  // 复制源位置字符串
  char *loc_copy = new char[strlen(loc) + 1];
  strcpy(loc_copy, loc);
  
  // 创建栈帧
  EvalStackFrame *frame = new EvalStackFrame();
  frame->source_location = loc_copy;
  frame->next = g_eval_stack;
  g_eval_stack = frame;
}

void print_eval_stack() {
  // 打印求值调用栈，用于错误报告
}
```

**特点：**
- 只追踪有源位置的表达式，避免保存临时对象
- 在错误时打印完整的调用栈
- 提供详细的类型错误信息

#### 3. 特殊形式处理

每个特殊形式都有独立的处理函数：

```c
// if 特殊形式
static SCM *eval_if(SCM_Environment *env, SCM_List *l, SCM **ast) {
  auto pred = eval_with_env(env, l->next->data);
  if (is_truthy(pred)) {
    *ast = l->next->next->data;
    return nullptr;  // 信号：继续求值
  }
  if (l->next->next->next) {
    *ast = l->next->next->next->data;
    return nullptr;
  }
  return scm_none();
}

// lambda 特殊形式
SCM *eval_lambda(SCM_Environment *env, SCM_List *l) {
  SCM *param_spec = l->next->data;
  // 处理单符号参数（rest parameter）
  if (is_sym(param_spec)) {
    // 转换为 dotted pair 形式
  }
  auto proc = make_proc(nullptr, proc_sig, l->next->next, env);
  return wrap(proc);
}
```

**特点：**
- 模块化设计，每个特殊形式独立处理
- 使用返回 `nullptr` 的信号机制来继续求值（尾递归优化）
- 支持 Scheme 的各种语法特性

### 优势

1. **实现简单直观**：直接求值，易于理解和维护
2. **错误信息完善**：提供详细的调用栈和源位置信息
3. **尾递归优化**：使用 `goto` 避免栈溢出
4. **宏支持**：在特殊形式检查之前处理宏展开
5. **模块化**：每个特殊形式独立实现，便于扩展

### 劣势

1. **性能开销**：每次求值都要重新解析和分发，没有记忆化
2. **符号查找效率低**：每次都要遍历环境链查找符号
3. **缺少代码优化**：没有将表达式转换为更高效的形式
4. **调试支持有限**：虽然有调用栈，但缺少更高级的调试功能
5. **内存管理**：使用 `new/delete` 手动管理，可能泄漏

## Guile 1.8 的实现

### 核心架构

Guile 1.8 使用**记忆化（Memoization）**机制，在第一次执行时将表达式转换为更高效的形式。

### 主要机制

#### 1. Memoization（记忆化）

Guile 在第一次求值表达式时，会将其转换为更高效的形式：

- **变量引用** → **iloc（间接位置）**或**variable 对象**
- **特殊形式** → **isym（内部符号）**
- **宏调用** → **展开后的代码**

```c
// Guile 的 lookup_symbol 在 memoization 阶段
static SCM lookup_symbol (const SCM symbol, const SCM env) {
  // 在环境中查找符号
  // 如果是局部变量，返回 iloc
  // 如果是全局变量，返回 variable 对象
  // 如果未找到，返回 SCM_UNDEFINED
}

// iloc 结构：<frame_number, binding_number, last?>
// 用于快速访问局部变量，避免重复查找
```

#### 2. iloc（间接位置）

iloc 是 memoized 的局部变量引用，包含三个值：
- `frame_nr`：环境帧的相对编号
- `binding_nr`：绑定在帧中的编号
- `last?`：是否是帧中最后一个绑定（用于 dotted pair）

```c
#define SCM_MAKE_ILOC(frame_nr, binding_nr, last_p) \
  SCM_PACK ( \
    ((frame_nr) << 8) \
    + ((binding_nr) << 20) \
    + ((last_p) ? SCM_ICDR : 0) \
    + scm_tc8_iloc )

// 查找 iloc 对应的值
SCM *scm_ilookup (SCM iloc, SCM env) {
  // 根据 frame_nr 和 binding_nr 快速定位变量
}
```

**优势：**
- 避免重复的环境查找
- 直接访问变量位置，性能高
- 支持嵌套环境的高效访问

#### 3. isym（内部符号）

特殊形式在 memoization 阶段被转换为 isym：

```c
// isym 表
static const char *const isymnames[] = {
  "#@and",
  "#@begin",
  "#@case",
  "#@cond",
  "#@do",
  "#@if",
  "#@lambda",
  "#@let",
  // ...
};

// 在 ceval 中根据 isym 分发
if (SCM_ISYMP (SCM_CAR (x))) {
  switch (ISYMNUM (SCM_CAR (x))) {
    case (ISYMNUM (SCM_IM_IF)):
      // 处理 if
      break;
    case (ISYMNUM (SCM_IM_LAMBDA)):
      // 处理 lambda
      break;
    // ...
  }
}
```

**优势：**
- 快速分发，避免字符串比较
- 代码更紧凑
- 支持代码优化

#### 4. ceval - 核心求值器

```c
static SCM CEVAL (SCM x, SCM env) {
loop:
  // 1. 检查是否是 isym（memoized 特殊形式）
  if (SCM_ISYMP (SCM_CAR (x))) {
    switch (ISYMNUM (SCM_CAR (x))) {
      case (ISYMNUM (SCM_IM_IF)):
        // 处理 if
        goto carloop;  // 尾递归
      case (ISYMNUM (SCM_IM_LAMBDA)):
        RETURN (scm_closure (...));
      // ...
    }
  }
  
  // 2. 检查是否是 iloc（memoized 变量）
  if (SCM_ILOCP (SCM_CAR (x))) {
    RETURN (*scm_ilookup (SCM_CAR (x), env));
  }
  
  // 3. 检查是否是 variable（全局变量）
  if (SCM_VARIABLEP (SCM_CAR (x))) {
    RETURN (SCM_VARIABLE_REF (SCM_CAR (x)));
  }
  
  // 4. 其他情况：符号查找或过程调用
  // ...
}
```

**关键特性：**
- 使用 `goto` 实现尾递归优化
- 根据 memoized 形式快速分发
- 支持调试版本（deval）和普通版本（ceval）

#### 5. 线程安全

Guile 处理了并发 memoization 的竞争条件：

```c
static SCM *scm_lookupcar1 (SCM vloc, SCM genv, int check) {
  // 查找符号并 memoize
  // 如果检测到竞争条件（另一个线程已经 memoized），返回 NULL
  // 调用者需要重新处理整个表达式
  if (!scm_is_eq (SCM_CAR (vloc), var)) {
    // 竞争条件：其他线程已经改变了这个 cell
    goto race;
  }
  // ...
race:
  // 如果已经是 variable 或 iloc，直接返回
  // 如果是 isym（特殊形式已被 memoized），返回 NULL
  return NULL;
}
```

### Guile 1.8 的优势

1. **性能优化**：memoization 避免重复查找和解析
2. **高效变量访问**：iloc 直接访问变量位置
3. **快速分发**：isym 避免字符串比较
4. **线程安全**：处理并发 memoization
5. **代码优化**：在 memoization 阶段可以进行优化
6. **调试支持**：deval 提供完整的调试框架

### Guile 1.8 的劣势

1. **实现复杂**：memoization 机制增加了复杂度
2. **内存开销**：需要存储 memoized 形式
3. **代码可读性**：isym 和 iloc 降低了代码可读性
4. **调试困难**：memoized 代码更难调试

## 对比总结

| 特性 | pscm_cc | Guile 1.8 |
|------|---------|-----------|
| **求值方式** | 直接求值 | Memoization + 求值 |
| **变量查找** | 每次遍历环境链 | iloc 直接访问 |
| **特殊形式分发** | 字符串比较 | isym 快速分发 |
| **代码转换** | 无 | 有（memoization） |
| **性能** | 较低 | 较高 |
| **实现复杂度** | 简单 | 复杂 |
| **可读性** | 高 | 低（memoized 代码） |
| **调试支持** | 基础 | 完善（deval） |
| **线程安全** | 单线程 | 多线程支持 |
| **内存管理** | 手动 | GC 管理 |

## pscm_cc 的改进方向

### 1. 短期改进（高优先级）

#### 1.1 实现变量缓存机制

**问题**：每次求值都要重新查找符号，效率低。

**改进**：在第一次查找后，将符号替换为直接引用或索引。

```c
// 在 eval_with_env 中
if (is_sym(ast)) {
  SCM_Symbol *sym = cast<SCM_Symbol>(ast);
  // 检查是否已经缓存
  if (sym->cached_location) {
    return *sym->cached_location;
  }
  // 查找并缓存
  SCM *result = lookup_symbol(env, sym);
  sym->cached_location = result;  // 缓存位置
  return result;
}
```

**注意**：需要考虑环境变化时清除缓存。

#### 1.2 优化特殊形式分发

**问题**：使用字符串比较分发特殊形式，效率低。

**改进**：使用哈希表或枚举类型。

```c
// 定义特殊形式枚举
enum SpecialForm {
  SF_IF, SF_LAMBDA, SF_LET, SF_LETSTAR, SF_LETREC,
  SF_QUOTE, SF_QUASIQUOTE, SF_SET, SF_DEFINE,
  // ...
};

// 在初始化时建立符号到枚举的映射
static std::unordered_map<std::string, SpecialForm> special_forms;

// 在 eval_with_env 中
if (is_sym(l->data)) {
  auto it = special_forms.find(sym->data);
  if (it != special_forms.end()) {
    switch (it->second) {
      case SF_IF: return eval_if(...);
      case SF_LAMBDA: return eval_lambda(...);
      // ...
    }
  }
}
```

#### 1.3 改进错误处理

**问题**：错误处理可以更完善。

**改进**：
- 使用异常机制（如果支持 C++ 异常）
- 提供更详细的错误信息
- 支持错误恢复机制

```c
// 定义异常类型
class EvalError : public std::exception {
  std::string message;
  SCM *context;
  EvalStackFrame *stack;
public:
  const char *what() const noexcept override {
    return message.c_str();
  }
  void print_stack() const {
    // 打印调用栈
  }
};

// 在 eval_with_env 中
try {
  // 求值逻辑
} catch (const EvalError &e) {
  e.print_stack();
  throw;
}
```

#### 1.4 内存管理改进

**问题**：使用 `new/delete` 手动管理，可能泄漏。

**改进**：
- 使用智能指针
- 或集成 GC
- 或使用对象池

```c
// 使用智能指针
std::unique_ptr<EvalStackFrame> frame(new EvalStackFrame());

// 或使用对象池
class EvalStackFramePool {
  std::vector<std::unique_ptr<EvalStackFrame>> pool;
public:
  EvalStackFrame *acquire() {
    if (pool.empty()) {
      return new EvalStackFrame();
    }
    auto frame = std::move(pool.back());
    pool.pop_back();
    return frame.release();
  }
  void release(EvalStackFrame *frame) {
    pool.push_back(std::unique_ptr<EvalStackFrame>(frame));
  }
};
```

### 2. 中期改进（中优先级）

#### 2.1 实现简单的 Memoization

**问题**：每次求值都要重新解析。

**改进**：实现简单的 memoization，至少缓存变量查找结果。

```c
struct MemoizedExpr {
  SCM *original;      // 原始表达式
  SCM *memoized;       // memoized 形式
  bool is_memoized;    // 是否已 memoized
};

// 在第一次求值时 memoize
SCM *memoize_expr(SCM *expr, SCM_Environment *env) {
  if (is_sym(expr)) {
    // 查找并替换为直接引用
    SCM *val = lookup_symbol(env, cast<SCM_Symbol>(expr));
    // 创建 memoized 形式（可以是 iloc 或 variable）
    return create_memoized_var(expr, val);
  }
  // 其他情况...
}
```

#### 2.2 实现 iloc 机制

**问题**：局部变量查找效率低。

**改进**：实现类似 Guile 的 iloc 机制。

```c
// iloc 结构
struct SCM_Iloc {
  uint16_t frame_nr;    // 环境帧编号
  uint16_t binding_nr;  // 绑定编号
  bool is_last;         // 是否是最后一个绑定
};

// 在 lambda 求值时创建 iloc
SCM *create_iloc(uint16_t frame, uint16_t binding, bool last) {
  // 创建 iloc 对象
}

// 查找 iloc
SCM *ilookup(SCM_Iloc *iloc, SCM_Environment *env) {
  // 根据 frame_nr 和 binding_nr 快速定位
  SCM_Environment *frame = env;
  for (int i = 0; i < iloc->frame_nr; i++) {
    frame = frame->parent;
  }
  // 根据 binding_nr 定位变量
  // ...
}
```

#### 2.3 优化尾递归

**问题**：虽然使用了 `goto`，但可以进一步优化。

**改进**：
- 识别更多尾递归情况
- 使用 trampoline 模式
- 优化循环结构

```c
// Trampoline 模式
SCM *trampoline_eval(SCM *expr, SCM_Environment *env) {
  while (true) {
    SCM *result = eval_with_env(expr, env);
    if (result->type == SCM::TAIL_CALL) {
      // 尾调用：继续循环
      expr = result->tail_expr;
      env = result->tail_env;
      continue;
    }
    return result;
  }
}
```

#### 2.4 改进宏系统

**问题**：宏展开可以更高效。

**改进**：
- 缓存宏展开结果
- 优化宏展开算法
- 支持更多宏特性

### 3. 长期改进（低优先级）

#### 3.1 实现完整的 Memoization

**问题**：缺少代码转换和优化。

**改进**：实现类似 Guile 的完整 memoization 机制。

```c
// Memoization 阶段
SCM *memoize(SCM *expr, SCM_Environment *env) {
  if (is_sym(expr)) {
    // 转换为 iloc 或 variable
  }
  else if (is_pair(expr)) {
    // 检查是否是特殊形式
    if (is_special_form(expr)) {
      // 转换为 isym
      return create_isym(expr);
    }
    // 递归 memoize 子表达式
    memoize(car(expr), env);
    memoize(cdr(expr), env);
  }
  return expr;
}
```

#### 3.2 实现代码生成

**问题**：解释执行效率有限。

**改进**：实现 JIT 编译或字节码生成。

```c
// 字节码生成
enum Bytecode {
  BC_LOAD_CONST,    // 加载常量
  BC_LOAD_VAR,      // 加载变量
  BC_CALL,          // 调用过程
  BC_IF,            // 条件跳转
  // ...
};

// 生成字节码
std::vector<Bytecode> compile(SCM *expr) {
  // 将表达式编译为字节码
}

// 字节码解释器
SCM *interpret(std::vector<Bytecode> &code) {
  // 解释执行字节码
}
```

#### 3.3 多线程支持

**问题**：当前是单线程实现。

**改进**：添加线程安全机制。

```c
// 线程安全的环境查找
SCM *thread_safe_lookup(SCM_Symbol *sym, SCM_Environment *env) {
  std::lock_guard<std::mutex> lock(env_mutex);
  return lookup_symbol(env, sym);
}

// 线程安全的 memoization
SCM *thread_safe_memoize(SCM *expr, SCM_Environment *env) {
  // 检查是否已被其他线程 memoized
  // 处理竞争条件
}
```

#### 3.4 性能分析工具

**问题**：缺少性能分析工具。

**改进**：添加性能分析功能。

```c
// 性能计数器
struct EvalStats {
  size_t eval_count;
  size_t symbol_lookup_count;
  size_t special_form_count;
  std::map<std::string, size_t> special_form_stats;
};

// 在 eval_with_env 中收集统计信息
void collect_stats(const char *form_name) {
  stats.eval_count++;
  if (form_name) {
    stats.special_form_stats[form_name]++;
  }
}
```

## 结论

pscm_cc 的 eval 实现是一个**简单而直接**的实现，适合学习和理解 Scheme 求值器的基本原理。与 Guile 1.8 相比，它牺牲了性能来换取实现的简单性。

对于 pscm_cc 的目标（驱动 TeXmacs），当前的实现应该是足够的。但如果要进一步提升性能，建议优先实现：

1. **变量缓存机制**（避免重复查找）
2. **特殊形式哈希表分发**（提高分发效率）
3. **简单的 memoization**（至少缓存变量查找）

这些改进将显著提高 eval 的性能，同时保持代码的简洁性。更复杂的优化（如完整的 memoization、JIT 编译）可以根据实际需求逐步实现。
