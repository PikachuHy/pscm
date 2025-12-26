# catch/throw 异常处理机制实现

本文档描述 pscm_cc 中 `catch` 和 `throw` 异常处理机制的实现，参考 Guile 1.8 的实现。

**状态**：✅ 已完成

## 一、设计目标

1. **异常捕获**：实现 `catch tag thunk handler`，允许捕获特定标签的异常
2. **异常抛出**：实现 `throw key args ...`，允许抛出异常
3. **错误恢复**：将当前的 `exit(1)` 错误处理改为可恢复的异常机制
4. **错误对象**：定义错误类型系统，支持错误消息格式化

## 二、核心设计

### 2.1 基于 setjmp/longjmp 的实现

参考 Guile 1.8，使用 `setjmp`/`longjmp` 实现异常处理：

```c
struct jmp_buf_and_retval {
  jmp_buf buf;           // 跳转缓冲区
  SCM *throw_tag;        // 抛出的标签
  SCM *throw_args;       // 抛出的参数列表
};

// catch 的核心实现
SCM *scm_c_catch(SCM *tag, 
                 SCM *(*body)(void *), void *body_data,
                 SCM *(*handler)(void *, SCM *, SCM *), void *handler_data) {
  struct jmp_buf_and_retval jbr;
  SCM *jmpbuf;
  SCM *answer;
  
  jmpbuf = make_jmpbuf();
  answer = scm_none();
  
  // 将 catch 信息添加到动态环境（类似 dynamic-wind）
  // 使用 g_wind_chain 存储 catch 信息
  // ...
  
  if (setjmp(jbr.buf)) {
    // 异常被捕获
    SCM *throw_tag = jbr.throw_tag;
    SCM *throw_args = jbr.throw_args;
    answer = handler(handler_data, throw_tag, throw_args);
  } else {
    // 正常执行 body
    answer = body(body_data);
  }
  
  return answer;
}
```

### 2.2 Catch 栈管理

使用全局 catch 栈管理活跃的 catch 信息：

```c
// catch 信息结构
struct catch_info {
  SCM *tag;              // 捕获的标签（符号或 #t）
  struct jmp_buf_and_retval *jbr;  // 跳转缓冲区和返回值结构
  SCM *(*handler)(void *, SCM *, SCM *);  // 处理函数
  void *handler_data;    // 处理函数数据
};

// 全局 catch 栈
static struct catch_info *g_catch_stack[100];
static int g_catch_stack_top = 0;
```

**实现说明**：
- 使用固定大小的数组（100）存储 catch 信息，避免动态分配
- 进入 `catch` 时，将 catch 信息压入栈
- 退出 `catch` 时（正常或异常），从栈中弹出
- `throw` 时从栈顶向下搜索匹配的 catch

### 2.3 throw 实现

`throw` 函数查找 catch 栈中的匹配标签，匹配后执行 `longjmp`：

```c
SCM *scm_throw(SCM *key, SCM *args) {
  // 1. 从栈顶向下遍历 g_catch_stack，查找匹配的 catch
  for (int i = g_catch_stack_top - 1; i >= 0; i--) {
    struct catch_info *info = g_catch_stack[i];
    
    // 2. 检查 catch 是否活跃且标签匹配
    if (info && info->jbr && info->jbr->active && tags_match(info->tag, key)) {
      // 3. 设置异常信息并跳转
      info->jbr->throw_tag = key;
      info->jbr->throw_args = args;
      longjmp(info->jbr->buf, 1);
    }
  }
  
  // 4. 如果没有找到匹配的 catch，调用未捕获异常处理器
  scm_uncaught_throw(key, args);
  return nullptr;  // 永远不会到达
}
```

**标签匹配规则**（`tags_match` 函数）：
- 如果 `tag` 是 `#t`，匹配所有异常
- 如果 `tag` 和 `key` 都是符号，比较符号名字（`strcmp`）
- 否则使用指针相等性比较

### 2.4 错误处理改进

将 `eval_error` 从 `exit(1)` 改为抛出异常：

```c
// 定义标准错误标签
SCM *g_error_key = nullptr;  // 初始化为符号 "error"

[[noreturn]] void eval_error(const char *format, ...) {
  va_list args;
  va_start(args, format);
  
  // 格式化错误消息
  char *message = format_error_message(format, args);
  va_end(args);
  
  // 构造错误参数列表
  // 格式：(subr message parts rest)
  // 或简化为：(message)
  SCM *error_args = make_list(wrap(make_string(message)));
  
  // 抛出异常而不是 exit(1)
  scm_throw(g_error_key, error_args);
}
```

### 2.5 错误对象系统

定义错误类型和错误对象结构：

```c
// 错误类型枚举
enum ErrorType {
  ERROR_TYPE_ERROR,      // 通用错误
  ERROR_TYPE_TYPE_ERROR, // 类型错误
  ERROR_TYPE_ARG_ERROR,  // 参数错误
  ERROR_TYPE_RANGE_ERROR // 范围错误
};

// 错误对象结构（可选，初期可以简化）
struct SCM_Error {
  SCM base;
  ErrorType type;
  char *message;
  SCM *args;  // 错误参数列表
};
```

初期实现可以简化，直接使用列表存储错误信息。

## 三、API 设计

### 3.1 Scheme 层 API

```scheme
;; catch tag thunk handler
;; 捕获标签为 tag 的异常，执行 thunk，如果抛出异常则调用 handler
(catch 'error 
       (lambda () (throw 'error "something went wrong"))
       (lambda (key . args) 
         (display "Caught error: ")
         (display args)
         (newline)
         'handled))

;; throw key args ...
;; 抛出异常，key 是标签（符号），args 是参数列表
(throw 'error "an error occurred")
```

### 3.2 C 层 API

```c
// 核心 catch 函数
SCM *scm_c_catch(SCM *tag,
                 SCM *(*body)(void *), void *body_data,
                 SCM *(*handler)(void *, SCM *, SCM *), void *handler_data);

// 抛出异常
SCM *scm_throw(SCM *key, SCM *args);

// 未捕获异常处理
[[noreturn]] void scm_uncaught_throw(SCM *key, SCM *args);
```

## 四、实现状态

### ✅ 步骤 1：基础 catch/throw 机制（已完成）

1. ✅ 创建 `throw.h` 和 `throw.cc`
2. ✅ 实现 `jmp_buf_and_retval` 结构
3. ✅ 实现 `scm_c_catch` 核心函数
4. ✅ 实现 `scm_throw` 函数
5. ✅ 使用全局 catch 栈（`g_catch_stack`）管理 catch 信息

**实现细节**：
- 使用全局数组 `g_catch_stack[100]` 存储活跃的 catch 信息
- 使用 `g_catch_stack_top` 跟踪栈顶位置
- `setjmp`/`longjmp` 实现非局部跳转
- `tags_match` 函数支持符号匹配和 `#t` 通配符

### ✅ 步骤 2：Scheme 层接口（已完成）

1. ✅ 实现 `scm_c_catch_scheme`（Scheme 可调用的 catch）
2. ✅ 实现 `scm_c_throw_scheme`（Scheme 可调用的 throw）
3. ✅ 在 `init.cc` 中注册 `catch` 和 `throw` 为可变参数函数

**实现细节**：
- `scm_c_catch_scheme` 接收 `(tag thunk handler)` 三个参数
- `scm_c_throw_scheme` 接收 `(key . args)` 可变参数
- 参数在调用前已由 `eval_with_func` 求值
- 使用 `scm_body_thunk` 和 `scm_handle_by_proc` 作为 C 回调函数

### ✅ 步骤 3：错误处理改进（已完成）

1. ✅ 定义标准错误标签（`g_error_key`，符号 `"error"`）
2. ✅ 修改 `eval_error` 使用 `scm_throw` 而不是 `exit(1)`
3. ✅ 修改 `type_error` 使用 `eval_error` 而不是 `exit(1)`
4. ✅ 实现 `scm_uncaught_throw` 作为未捕获异常的默认处理器

**实现细节**：
- `eval_error` 构造错误消息字符串，包装为 Scheme 字符串对象
- 错误参数格式：`(message)`，其中 `message` 是包含完整错误信息的字符串
- `type_error` 现在也通过 `eval_error` 抛出异常，可以被 `catch` 捕获
- `scm_uncaught_throw` 打印错误信息、调用栈，然后 `exit(1)`

### ⚠️ 步骤 4：错误对象系统（未实现）

1. ⚠️ 定义错误类型枚举（当前使用简单的字符串消息）
2. ⚠️ 实现错误对象结构（当前使用列表存储错误信息）
3. ⚠️ 实现错误消息格式化函数（当前使用 `sprintf` 格式化）

**当前实现**：
- 错误信息以字符串形式存储在 `(message)` 列表中
- 错误消息包含源位置、错误类型、实际类型、求值上下文等信息
- 未来可以扩展为更结构化的错误对象系统

## 五、与现有系统的集成

### 5.1 与 dynamic-wind 的集成

catch 和 dynamic-wind 是独立的机制：
- catch 使用独立的 `g_catch_stack` 管理
- dynamic-wind 使用 `g_wind_chain` 管理
- 两者可以协同工作：当 throw 跨越 dynamic-wind 时，wind guard 会正确执行

### 5.2 与 continuation 的集成

catch 和 continuation 可以协同工作：
- 如果 continuation 跨越 catch，catch 信息会被正确保存和恢复
- 如果 throw 跨越 continuation，会正确查找 catch

### 5.3 与调用栈追踪的集成

保持现有的调用栈追踪功能，在 catch 和 throw 时不影响栈追踪。

## 六、测试用例

### 6.1 基础 catch/throw

```scheme
;; 测试 1: 基本 catch/throw
(catch 'test
       (lambda () (throw 'test "hello"))
       (lambda (key . args) 
         (display "Caught: ")
         (display args)
         (newline)
         'ok))
;; 期望输出: Caught: (hello)
;; 期望返回值: ok

;; 测试 2: catch #t 捕获所有异常
(catch #t
       (lambda () (throw 'any-key "any error"))
       (lambda (key . args) 
         (display "Caught any: ")
         (display key)
         (display " ")
         (display args)
         (newline)
         'ok))

;; 测试 3: 嵌套 catch
(catch 'outer
       (lambda ()
         (catch 'inner
                (lambda () (throw 'inner "inner error"))
                (lambda (key . args) 'inner-handled))
         (throw 'outer "outer error"))
       (lambda (key . args) 'outer-handled))
;; 期望返回值: outer-handled
```

### 6.2 错误处理改进

```scheme
;; 测试 4: eval_error 改为抛出异常
(catch 'error
       (lambda () 
         ;; 触发一个错误（例如类型错误）
         (car 42))
       (lambda (key . args) 
         (display "Caught error: ")
         (display args)
         (newline)
         'error-handled))
;; 期望: 不再直接 exit(1)，而是被 catch 捕获
```

### 6.3 与 dynamic-wind 的交互

```scheme
;; 测试 5: catch 和 dynamic-wind 的交互
(let ((x 0))
  (catch 'test
         (lambda ()
           (dynamic-wind
            (lambda () (set! x (+ x 1)))  ; before
            (lambda () (throw 'test "error"))
            (lambda () (set! x (+ x 10)))))  ; after
         (lambda (key . args) x))
  x)
;; 期望: x 的值反映了 dynamic-wind 的正确执行
```

## 七、注意事项

1. **内存管理**：确保 `jmp_buf` 和相关数据结构不会被 GC 回收（目前 pscm_cc 没有 GC，但需要考虑未来）
2. **栈溢出**：避免在 catch 处理函数中无限递归抛出异常
3. **线程安全**：如果未来支持多线程，需要考虑线程局部存储
4. **性能**：`setjmp`/`longjmp` 有一定的性能开销，但异常处理通常不是性能关键路径

## 八、实现细节

### 8.1 关键实现点

1. **参数求值问题**：
   - 问题：`scm_handle_by_proc` 使用 `apply_procedure` 调用 handler，但 `apply_procedure` 会对参数求值
   - 解决：创建 `apply_procedure_with_values` 函数，直接使用已求值的参数，避免重复求值

2. **Rest 参数绑定**：
   - handler 通常使用 `(lambda (key . args) ...)` 形式
   - 需要正确处理 `penultimate_param`（倒数第二个参数）和 rest 参数的绑定

3. **错误处理统一**：
   - `eval_error` 和 `type_error` 都使用 `scm_throw` 抛出异常
   - 标准错误标签 `g_error_key` 为符号 `"error"`
   - 错误消息格式化为字符串，包装在 `(message)` 列表中

### 8.2 测试覆盖

所有测试用例已通过（`test/base/catch_throw_tests.scm`）：
- ✅ 基本 catch/throw
- ✅ catch #t 捕获所有异常
- ✅ 正常返回值（无异常时）
- ✅ Handler 接收 key 和 args
- ✅ Handler 返回值
- ✅ 错误处理（`eval_error` 和 `type_error` 可被捕获）

## 九、参考实现

- Guile 1.8: `guile/libguile/throw.c` 和 `throw.h`
- 核心函数：`scm_c_catch`, `scm_throw`, `scm_handle_by_proc`
- 实现方式：基于 `setjmp`/`longjmp` 的非局部跳转

