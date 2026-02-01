# Read Options 系统实现

本文档描述 pscm_cc 中 `read-options-interface`、`read-set!`、`read-enable` 和 `read-disable` 的实现，参考 Guile 1.8 的实现。

**状态**：✅ 已完成

## 一、设计目标

1. **选项配置**：实现 `read-options-interface`，显示当前读取选项状态
2. **选项设置**：实现 `read-set!`，设置读取选项的值
3. **选项启用/禁用**：实现 `read-enable` 和 `read-disable`，用于布尔选项的启用和禁用
4. **Guile 兼容性**：与 Guile 1.8 的行为保持一致

## 二、核心设计

### 2.1 选项数据结构

```c
// 读取选项类型
enum scm_read_option_type {
  SCM_READ_OPTION_BOOL,  // 布尔选项
  SCM_READ_OPTION_SCM    // SCM 值选项
};

// 读取选项结构
struct scm_t_read_option {
  enum scm_read_option_type type;  // 选项类型
  const char *name;                // 选项名称
  SCM *value;                      // 当前值
  const char *doc;                 // 文档字符串
};
```

### 2.2 选项列表

当前支持的读取选项：

- `keywords`：SCM 类型，控制关键字处理方式（默认 `#f`）
- `positions`：布尔类型，控制是否记录位置信息（默认启用）

### 2.3 宏实现

所有选项操作函数都实现为 Scheme 宏，使用 `define-macro`：

```scheme
(define-macro (read-set! key val)
  `(scm_c_read_set! ,(quote key) ,val))

(define-macro (read-enable key)
  `(scm_c_read_enable ,(quote key)))

(define-macro (read-disable key)
  `(scm_c_read_disable ,(quote key)))
```

## 三、API 说明

### 3.1 read-options-interface

显示当前所有读取选项的状态：

```scheme
(read-options-interface)
;; => (keywords #f positions)
```

返回一个列表，包含所有选项的当前值。

### 3.2 read-set!

设置读取选项的值：

```scheme
(read-set! keywords #f)
(read-set! positions #t)
```

### 3.3 read-enable

启用布尔选项：

```scheme
(read-enable positions)
```

### 3.4 read-disable

禁用布尔选项：

```scheme
(read-disable positions)
```

## 四、实现细节

### 4.1 选项查找

通过选项名称在选项数组中查找对应的选项：

```c
static scm_t_read_option *find_read_option(const char *name) {
  for (int i = 0; i < NUM_READ_OPTS; i++) {
    if (strcmp(scm_read_opts[i].name, name) == 0) {
      return &scm_read_opts[i];
    }
  }
  return nullptr;
}
```

### 4.2 选项值获取

`read-options-interface` 按照特定顺序构建返回值列表，与 Guile 1.8 保持一致。

### 4.3 选项值修改

- 对于布尔选项：使用 `read-enable` 和 `read-disable` 进行启用/禁用
- 对于 SCM 选项：使用 `read-set!` 直接设置值

## 五、测试

测试文件：`test/base/read_set_tests.scm`

测试覆盖：
- 默认选项值
- 设置选项值
- 启用/禁用布尔选项
- 多个选项操作
- 值保持

## 六、参考

- Guile 1.8 源码：`libguile/read.c`
- 测试用例：`test/base/read_set_tests.scm`
