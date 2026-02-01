# 错误处理改进

本文档描述 pscm_cc 中错误处理系统的改进，包括源位置跟踪、堆栈跟踪和错误恢复机制。

**状态**：✅ 已完成

## 一、改进概述

### 1.1 主要改进

1. **源位置上下文**：在环境搜索错误中添加源位置信息
2. **堆栈跟踪改进**：从 `exit` 改为 `abort` 以改善堆栈跟踪可见性
3. **Eval 调用栈**：在崩溃时打印 Scheme eval 调用栈
4. **错误消息增强**：提供更详细的错误上下文信息

## 二、源位置跟踪

### 2.1 环境搜索错误增强

在 `scm_env_search` 函数中，当符号找不到时，会尝试获取并打印源位置信息：

```c
// Try to get source location from current eval context for debugging
const char *loc_str = nullptr;
if (g_current_eval_context) {
  loc_str = get_source_location_str(g_current_eval_context);
}

if (loc_str) {
  SCM_ERROR_SYMTBL("find %s, not found (at %s)\n", sym->data, loc_str);
} else {
  SCM_ERROR_SYMTBL("find %s, not found\n", sym->data);
}
```

### 2.2 错误输出示例

**交互模式**：
```
[symtbl] environment.cc:104 find undefined-symbol, not found (at <unknown>:1:1)
```

**从文件加载**：
```
[symtbl] environment.cc:104 find undefined-symbol, not found (at /tmp/test_symbol.scm:2:1)
```

## 三、堆栈跟踪改进

### 3.1 从 exit 改为 abort

将错误处理中的 `exit(1)` 改为 `abort()`，以便在调试时获得完整的堆栈跟踪：

```c
// 之前
eval_error("error message");
exit(1);

// 现在
eval_error("error message");
abort();  // 触发信号处理器，打印堆栈跟踪
```

### 3.2 信号处理器

设置信号处理器以捕获崩溃并打印堆栈跟踪：

```c
void setup_abort_handler() {
  signal(SIGABRT, print_stacktrace);
  signal(SIGSEGV, print_stacktrace);
  signal(SIGILL, print_stacktrace);
}
```

### 3.3 堆栈跟踪打印

`print_stacktrace` 函数使用 `backtrace` 和 `atos` 打印 C++ 堆栈跟踪：

```c
void print_stacktrace(int signal) {
  void *callstack[128];
  int frames = backtrace(callstack, 128);
  
  // 使用 atos 解析地址
  // ...
  
  // 如果 pscm 正在执行 eval，也打印 eval 调用栈
  print_eval_stack();
  
  exit(1);
}
```

## 四、Eval 调用栈

### 4.1 调用栈跟踪

在 eval 过程中维护调用栈，记录每个表达式的源位置：

```c
struct EvalStackFrame {
  char *source_location;        // 源位置字符串
  char *expr_str;               // 表达式字符串表示
  EvalStackFrame *next;         // 下一个帧
};
```

### 4.2 调用栈打印

在崩溃时，除了 C++ 堆栈跟踪，还会打印 Scheme eval 调用栈：

```
Stack trace:
#0 function_name at file.cc:line
#1 another_function at file.cc:line
...

Evaluation call stack (most recent first):
  #0: /path/to/file.scm:10:5
      (some-expression)
  #1: /path/to/file.scm:5:2
      (another-expression)
  ...
```

## 五、错误消息增强

### 5.1 上下文信息

错误消息现在包含：
- 源位置（文件名、行号、列号）
- 当前求值的表达式
- 调用栈信息

### 5.2 错误格式

```
pscm: uncaught throw to 'error: /path/to/file.scm:10:5: symbol 'undefined-symbol' not found

=== Evaluation Call Stack ===
Evaluation call stack (most recent first):
  #0: /path/to/file.scm:10:5
      undefined-symbol
=== End of Call Stack ===
```

## 六、使用场景

### 6.1 调试未定义符号

当遇到未定义符号时，可以快速定位问题：

```
[symtbl] environment.cc:104 find undefined-symbol, not found (at test.scm:10:5)
```

### 6.2 调试崩溃

当程序崩溃时，可以获得完整的堆栈信息，包括：
- C++ 调用栈（使用 atos 解析）
- Scheme eval 调用栈
- 源位置信息

## 七、参考

- 实现文件：`src/c/abort.cc`、`src/c/environment.cc`、`src/c/eval.cc`
- 相关功能：源位置跟踪、错误处理、堆栈跟踪
