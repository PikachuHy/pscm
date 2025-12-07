# pscm_cc 字符支持实现说明

::: tip 状态：已实现
字符支持功能已在 pscm_cc 中完整实现，包括字符类型、解析、打印和相关函数。
:::

## 问题分析

### 当前问题

`char_tests.scm` 测试失败，因为 pscm_cc 的解析器不支持字符字面量语法：
- `#\A` - 字符 A
- `#\.` - 字符点号
- `#\space` - 空格字符
- `#\newline` - 换行符

错误信息：
```
test/base/char_tests.scm:9:16: parse error: unexpected character
```

### 测试用例需求

从 `char_tests.scm` 可以看到需要支持：
1. 字符字面量：`#\A`, `#\.`, `#\a`
2. `char->integer` 函数：将字符转换为整数
3. `integer->char` 函数：将整数转换为字符
4. 字符打印：打印字符本身（不带 `#\` 前缀）

## Guile 1.8 的实现方式

### 1. 字符表示

Guile 1.8 使用 **立即值（Immediate Value）** 技术存储字符：

```c
// chars.h
#define SCM_CHARP(x) (SCM_ITAG8(x) == scm_tc8_char)
#define SCM_CHAR(x) ((unsigned int)SCM_ITAG8_DATA(x))
#define SCM_MAKE_CHAR(x) SCM_MAKE_ITAG8((scm_t_bits) (unsigned char) (x), scm_tc8_char)
```

**特点**：
- 字符值直接编码在 SCM 值中（tagged pointer）
- 使用 `scm_tc8_char` 作为类型码
- 字符值存储在低 8 位（或更多位，取决于实现）
- 不需要额外的内存分配

### 2. 字符类型

字符是一个独立的基础类型，与数字、字符串等并列。

### 3. 字符解析

Guile 的读取器（reader）在解析时识别 `#\` 前缀：
- `#\A` → 字符 'A'
- `#\.` → 字符 '.'
- `#\space` → 字符 ' '（空格）
- `#\newline` → 字符 '\n'（换行符）
- `#\tab` → 字符 '\t'（制表符）

### 4. 字符操作函数

Guile 提供以下字符操作函数：
- `char?` - 判断是否为字符
- `char->integer` - 字符转整数
- `integer->char` - 整数转字符
- `char=?`, `char<?`, `char>?` 等比较函数
- `char-upcase`, `char-downcase` 等转换函数

## pscm_cc 当前架构

### 当前类型系统

```cpp
struct SCM {
  enum Type { 
    NONE, NIL, LIST, PROC, CONT, FUNC, 
    NUM, FLOAT, BOOL, SYM, STR, MACRO 
  } type;
  void *value;
  SCM_SourceLocation *source_loc;
};
```

**问题**：
- 没有 `CHAR` 类型
- 解析器不支持 `#\` 语法
- 没有字符相关的函数

## 架构调整方案

### 方案1：添加 CHAR 类型，直接使用 void* 存储（推荐）

**优点**：
- 与浮点数实现方式一致（直接使用 `void*` 存储）
- 不需要额外的内存分配
- 类型系统清晰

**实现**：

```cpp
struct SCM {
  enum Type { 
    ..., NUM, FLOAT, CHAR, BOOL, SYM, STR, MACRO 
  } type;
  void *value;  // 对于 CHAR，直接存储 char 值（通过 void* 转换）
  SCM_SourceLocation *source_loc;
};

// 辅助函数
inline void* char_to_ptr(char val) {
  return (void*)(uintptr_t)(unsigned char)val;
}

inline char ptr_to_char(void *ptr) {
  return (char)(uintptr_t)ptr;
}

inline SCM *scm_from_char(char val) {
  SCM *scm = new SCM();
  scm->type = SCM::CHAR;
  scm->value = char_to_ptr(val);
  scm->source_loc = nullptr;
  return scm;
}

inline char scm_to_char(SCM *scm) {
  if (is_char(scm)) {
    return ptr_to_char(scm->value);
  }
  return 0;
}

inline bool is_char(SCM *scm) {
  return scm->type == SCM::CHAR;
}
```

### 方案2：使用立即值技术（类似 Guile）

**优点**：
- 与 Guile 实现方式一致
- 内存效率最高

**缺点**：
- 需要修改 SCM 的表示方式
- 实现复杂度较高

### 方案3：使用单独的结构体

**缺点**：
- 需要额外的内存分配
- 与当前架构不一致

## 推荐实现方案：方案1

### 实现步骤

1. **修改类型定义**（`pscm.h`）：
   - 添加 `SCM::CHAR` 类型
   - 添加 `is_char()` 函数
   - 添加 `char_to_ptr()` 和 `ptr_to_char()` 转换函数
   - 添加 `scm_from_char()` 和 `scm_to_char()` 函数

2. **修改解析器**（`parse.cc`）：
   - 添加 `parse_char()` 函数，识别 `#\` 前缀
   - 支持单字符：`#\A`, `#\.`
   - 支持命名字符：`#\space`, `#\newline`, `#\tab`

3. **修改打印函数**（`print.cc`）：
   - 添加字符打印逻辑
   - 打印字符本身（不带 `#\` 前缀）

4. **实现字符函数**（新建 `char.cc`）：
   - `char?` - 判断是否为字符
   - `char->integer` - 字符转整数
   - `integer->char` - 整数转字符
   - `char=?`, `char<?`, `char>?` 等比较函数

5. **修改相等性比较**（`eq.cc`）：
   - 支持字符相等性比较

### 字符解析规则

```cpp
static SCM *parse_char(Parser *p) {
  // 必须匹配 #\
  if (p->pos[0] != '#' || p->pos[1] != '\\') {
    return nullptr;
  }
  
  p->pos += 2;  // 跳过 #\
  p->column += 2;
  
  // 检查命名字符
  if (isalpha((unsigned char)*p->pos)) {
    // 可能是命名字符：space, newline, tab 等
    const char *start = p->pos;
    while (isalpha((unsigned char)*p->pos)) {
      p->pos++;
      p->column++;
    }
    
    int len = p->pos - start;
    if (len == 5 && strncmp(start, "space", 5) == 0) {
      return scm_from_char(' ');
    } else if (len == 7 && strncmp(start, "newline", 7) == 0) {
      return scm_from_char('\n');
    } else if (len == 3 && strncmp(start, "tab", 3) == 0) {
      return scm_from_char('\t');
    }
    // 如果不是命名字符，则作为普通字符处理
    // 回退到 start
    p->pos = start;
    p->column -= len;
  }
  
  // 单字符
  if (*p->pos == '\0') {
    parse_error(p, "unexpected end of input in character literal");
  }
  
  char ch = *p->pos;
  p->pos++;
  p->column++;
  
  return scm_from_char(ch);
}
```

### 字符打印

```cpp
if (is_char(ast)) {
  char ch = ptr_to_char(ast->value);
  printf("%c", ch);
  return;
}
```

### 字符函数实现

```cpp
// char.cc
SCM *scm_c_is_char(SCM *arg) {
  return is_char(arg) ? scm_bool_true() : scm_bool_false();
}

SCM *scm_c_char_to_integer(SCM *arg) {
  if (!is_char(arg)) {
    eval_error("char->integer: expected character");
    return nullptr;
  }
  char ch = scm_to_char(arg);
  SCM *scm = new SCM();
  scm->type = SCM::NUM;
  scm->value = (void*)(int64_t)(unsigned char)ch;
  return scm;
}

SCM *scm_c_integer_to_char(SCM *arg) {
  if (!is_num(arg)) {
    eval_error("integer->char: expected integer");
    return nullptr;
  }
  int64_t val = (int64_t)arg->value;
  if (val < 0 || val > 255) {
    eval_error("integer->char: value out of range");
    return nullptr;
  }
  return scm_from_char((char)val);
}
```

## 实际实现

### 实现状态

✅ **已完全实现**，采用方案1（添加 CHAR 类型，直接使用 void* 存储）

### 实现细节

1. **类型定义**（`pscm.h`）：
   - ✅ 已添加 `SCM::CHAR` 类型
   - ✅ 已实现 `is_char()` 函数
   - ✅ 已实现 `char_to_ptr()` 和 `ptr_to_char()` 转换函数
   - ✅ 已实现 `scm_from_char()` 和 `scm_to_char()` 函数

2. **解析器**（`parse.cc`）：
   - ✅ 已实现 `parse_char()` 函数，支持 `#\` 前缀
   - ✅ 支持单字符：`#\A`, `#\.`
   - ✅ 支持命名字符：`#\space`, `#\newline`, `#\tab`

3. **打印函数**（`print.cc`）：
   - ✅ 已实现字符打印逻辑
   - ✅ 打印字符本身（不带 `#\` 前缀）

4. **字符函数**（`char.cc`）：
   - ✅ `char?` - 判断是否为字符
   - ✅ `char->integer` - 字符转整数
   - ✅ `integer->char` - 整数转字符

5. **相等性比较**（`eq.cc`）：
   - ✅ 已支持字符相等性比较

### 实现代码示例

实际实现与推荐方案完全一致：

```cpp
// pscm.h 中的实现
inline void* char_to_ptr(char val) {
  return (void*)(uintptr_t)(unsigned char)val;
}

inline char ptr_to_char(void *ptr) {
  return (char)(uintptr_t)ptr;
}

inline SCM *scm_from_char(char val) {
  SCM *scm = new SCM();
  scm->type = SCM::CHAR;
  scm->value = char_to_ptr(val);
  scm->source_loc = nullptr;
  return scm;
}

inline char scm_to_char(SCM *scm) {
  if (is_char(scm)) {
    return ptr_to_char(scm->value);
  }
  return 0;
}
```

## 总结

**实际实现采用方案1**：添加 `CHAR` 类型，直接使用 `void*` 存储字符值。

**关键点**：
1. ✅ 类型系统层面添加字符类型
2. ✅ 解析器支持 `#\` 字符字面量语法
3. ✅ 实现字符相关函数（`char?`, `char->integer`, `integer->char`）
4. ✅ 支持字符打印和比较

**与浮点数实现的对比**：
- 浮点数：`double` (8 字节) → `void*` (通过 union)
- 字符：`char` (1 字节) → `void*` (直接转换，更简单)

