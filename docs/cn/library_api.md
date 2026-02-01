# pscm_cc 库 API 使用指南

pscm_cc 现在可以作为库被其他应用使用。本文档介绍如何使用 pscm_cc 的库接口。

## 概述

pscm_cc 提供了 C API，允许其他应用嵌入 Scheme 解释器。主要接口定义在 `src/c/pscm_api.h` 中。

## 基本使用

### 1. 初始化

在使用任何 pscm_cc 功能之前，必须先初始化：

```c
#include "pscm_api.h"

int main() {
    pscm_init();
    // ... 使用 pscm_cc ...
    return 0;
}
```

### 2. 解析和求值

#### 解析字符串

```c
// 解析 Scheme 代码字符串，返回 AST
SCM *ast = pscm_parse("(+ 1 2)");
```

#### 解析文件

```c
// 解析 Scheme 文件，返回表达式列表
SCM_List *exprs = pscm_parse_file("example.scm");
```

#### 求值 AST

```c
// 求值一个 AST 节点
SCM *result = pscm_eval(ast);
```

#### 直接求值字符串

```c
// 解析并求值字符串（一步完成）
SCM *result = pscm_eval_string("(+ 1 2)");
```

#### 求值文件

```c
// 解析并求值文件中的所有表达式，返回最后一个结果
SCM *result = pscm_eval_file("example.scm");
```

### 3. 环境操作

#### 获取全局环境

```c
SCM_Environment *env = pscm_get_global_env();
```

#### 创建新环境

```c
// 创建新环境，继承全局环境
SCM_Environment *new_env = pscm_create_env(NULL);

// 或指定父环境
SCM_Environment *child_env = pscm_create_env(parent_env);
```

### 4. 调试控制

```c
// 启用/禁用调试输出
pscm_set_debug_enabled(true);
pscm_set_ast_debug_enabled(true);

// 查询当前状态
bool debug = pscm_get_debug_enabled();
bool ast_debug = pscm_get_ast_debug_enabled();
```

### 5. 错误处理

```c
// 设置自定义错误处理器
void my_error_handler(const char *message) {
    fprintf(stderr, "Custom error: %s\n", message);
}

pscm_set_error_handler(my_error_handler);
```

## 完整示例

```c
#include "pscm_api.h"
#include <stdio.h>

int main() {
    // 初始化
    pscm_init();
    
    // 求值简单表达式
    SCM *result = pscm_eval_string("(+ 1 2)");
    if (result) {
        // 打印结果（需要包含 print.h 或使用其他打印函数）
        print_ast(result);
        printf("\n");
    }
    
    // 求值文件
    SCM *file_result = pscm_eval_file("example.scm");
    
    // 清理（当前为 no-op，但保留接口）
    pscm_cleanup();
    
    return 0;
}
```

## API 参考

### 初始化函数

- `void pscm_init(void)` - 初始化 pscm_cc 库
- `void pscm_cleanup(void)` - 清理资源（当前为 no-op）

### 求值函数

- `SCM *pscm_eval(SCM *ast)` - 求值 AST 节点
- `SCM *pscm_eval_string(const char *code)` - 解析并求值字符串
- `SCM *pscm_eval_file(const char *filename)` - 解析并求值文件

### 解析函数

- `SCM *pscm_parse(const char *code)` - 解析字符串为 AST
- `SCM_List *pscm_parse_file(const char *filename)` - 解析文件为表达式列表

### 环境函数

- `SCM_Environment *pscm_get_global_env(void)` - 获取全局环境
- `SCM_Environment *pscm_create_env(SCM_Environment *parent)` - 创建新环境

### 调试函数

- `void pscm_set_debug_enabled(bool enabled)` - 设置调试输出
- `void pscm_set_ast_debug_enabled(bool enabled)` - 设置 AST 调试输出
- `bool pscm_get_debug_enabled(void)` - 获取调试状态
- `bool pscm_get_ast_debug_enabled(void)` - 获取 AST 调试状态

### 错误处理

- `void pscm_set_error_handler(pscm_error_handler_t handler)` - 设置错误处理器

### 字符串和符号转换

- `SCM *scm_from_c_string(const char *data, int len)` - 从 C 字符串创建 Scheme 字符串
- `SCM *scm_from_locale_stringn(const char *str, size_t len)` - 从 C locale 字符串创建 Scheme 字符串
- `char *scm_to_locale_stringn(SCM *str, size_t *lenp)` - 将 Scheme 字符串转换为 C locale 字符串

### 过程调用

- `SCM *apply_procedure(SCM_Environment *env, SCM_Procedure *proc, SCM_List *args)` - 应用 Scheme 过程
- `SCM *apply_procedure_with_values(SCM_Environment *env, SCM_Procedure *proc, SCM_List *args)` - 应用过程（参数已求值）
- `SCM *eval_with_func(SCM_Function *func, SCM_List *l)` - 求值 C/C++ 函数

### 文件加载

- `SCM *scm_c_primitive_load(const char *filename)` - 加载并执行文件（C API，接受 `const char *`）
- `SCM *scm_c_primitive_load_from_scm(SCM *filename)` - 加载并执行文件（Scheme API，接受 `SCM *`）
- `SCM *scm_eval_expression_list(SCM_List *expr_list)` - 求值表达式列表

### 选项系统

- `SCM *scm_c_read_options_interface()` - 获取读取选项接口
- `SCM *scm_c_read_set(SCM *key, SCM *val)` - 设置读取选项
- `SCM *scm_c_read_enable(SCM *key)` - 启用读取选项
- `SCM *scm_c_read_disable(SCM *key)` - 禁用读取选项
- `SCM *scm_c_debug_options_interface()` - 获取调试选项接口
- `SCM *scm_c_debug_set(SCM *key, SCM *val)` - 设置调试选项
- `SCM *scm_c_debug_enable(SCM *key)` - 启用调试选项
- `SCM *scm_c_debug_disable(SCM *key)` - 禁用调试选项

### 端口操作

- `SCM *scm_c_make_soft_port(SCM_List *args)` - 创建软端口

## 注意事项

1. **线程安全**：当前实现不是线程安全的。如果需要在多线程环境中使用，需要添加适当的同步机制。

2. **内存管理**：pscm_cc 当前没有垃圾回收机制，所有分配的内存不会自动释放。

3. **全局状态**：pscm_cc 使用全局变量（`g_env`, `g_wind_chain` 等）。在多实例场景中需要注意状态隔离。

4. **错误处理**：默认情况下，错误会通过 `eval_error` 抛出异常。如果设置了错误处理器，可能会被调用。

## 与命令行工具的关系

`pscm_cc` 命令行工具现在使用这些库 API 实现。`main.cc` 中的代码展示了如何使用这些 API 构建一个完整的 Scheme 解释器。

