# Debug Options 系统实现

本文档描述 pscm_cc 中 `debug-options-interface`、`debug-set!`、`debug-enable` 和 `debug-disable` 的实现，参考 Guile 1.8 的实现。

**状态**：✅ 已完成

## 一、设计目标

1. **调试选项配置**：实现 `debug-options-interface`，显示当前调试选项状态
2. **选项设置**：实现 `debug-set!`，设置调试选项的值
3. **选项启用/禁用**：实现 `debug-enable` 和 `debug-disable`，用于布尔选项的启用和禁用
4. **Guile 兼容性**：与 Guile 1.8 的行为保持一致

## 二、核心设计

### 2.1 选项数据结构

```c
// 调试选项类型
enum scm_debug_option_type {
  SCM_DEBUG_OPTION_BOOL,   // 布尔选项
  SCM_DEBUG_OPTION_INT,    // 整数选项
  SCM_DEBUG_OPTION_SCM     // SCM 值选项
};

// 调试选项结构
struct scm_t_debug_option {
  enum scm_debug_option_type type;  // 选项类型
  const char *name;                 // 选项名称
  union {
    bool bool_val;                  // 布尔值
    int int_val;                    // 整数值
    SCM *scm_val;                   // SCM 值
  } value;                          // 当前值
  const char *doc;                  // 文档字符串
};
```

### 2.2 选项列表

当前支持的调试选项包括：

- `show-file-name`：布尔类型，控制是否显示文件名
- `stack`：整数类型，控制堆栈大小
- `depth`：整数类型，控制深度限制
- `maxdepth`：整数类型，控制最大深度
- `frames`：整数类型，控制帧数
- `indent`：整数类型，控制缩进
- `width`：整数类型，控制宽度
- `procnames`：布尔类型，控制是否显示过程名
- `backtrace`：布尔类型，控制是否显示回溯
- `trace`：布尔类型，控制是否显示跟踪
- `cheap`：布尔类型，控制是否使用廉价模式

### 2.3 宏实现

所有选项操作函数都实现为 Scheme 宏，使用 `define-macro`：

```scheme
(define-macro (debug-set! key val)
  `(scm_c_debug_set! ,(quote key) ,val))

(define-macro (debug-enable key)
  `(scm_c_debug_enable ,(quote key)))

(define-macro (debug-disable key)
  `(scm_c_debug_disable ,(quote key)))
```

## 三、API 说明

### 3.1 debug-options-interface

显示当前所有调试选项的状态：

```scheme
(debug-options-interface)
;; => (show-file-name #t stack 20000 depth 20 maxdepth 1000 frames 3 indent 10 width 79 procnames cheap)
```

返回一个列表，包含所有选项的当前值。

### 3.2 debug-set!

设置调试选项的值：

```scheme
(debug-set! stack 200000)
(debug-set! frames 5)
(debug-set! width 100)
```

### 3.3 debug-enable

启用布尔选项：

```scheme
(debug-enable backtrace)
(debug-enable trace)
```

### 3.4 debug-disable

禁用布尔选项：

```scheme
(debug-disable backtrace)
```

## 四、实现细节

### 4.1 选项查找

通过选项名称在选项数组中查找对应的选项：

```c
static scm_t_debug_option *find_debug_option(const char *name) {
  for (int i = 0; i < NUM_DEBUG_OPTS; i++) {
    if (strcmp(scm_debug_opts[i].name, name) == 0) {
      return &scm_debug_opts[i];
    }
  }
  return nullptr;
}
```

### 4.2 选项值获取

`debug-options-interface` 按照特定顺序构建返回值列表，与 Guile 1.8 保持一致。

### 4.3 选项值修改

- 对于布尔选项：使用 `debug-enable` 和 `debug-disable` 进行启用/禁用
- 对于整数选项：使用 `debug-set!` 直接设置值
- 对于 SCM 选项：使用 `debug-set!` 直接设置值

## 五、测试

测试文件：`test/base/debug_set_tests.scm`

测试覆盖：
- 默认选项值
- 设置整数选项
- 设置 SCM 选项
- 启用/禁用布尔选项
- 多个选项操作
- 值保持

## 六、参考

- Guile 1.8 源码：`libguile/debug.c`
- 测试用例：`test/base/debug_set_tests.scm`
