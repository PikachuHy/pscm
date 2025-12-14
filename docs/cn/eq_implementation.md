# pscm_cc Eq/Eqv/Equal 实现分析

## 概述

本文档分析 pscm_cc 中 `eq?`、`eqv?`、`equal?` 的实现方式，对比 Guile 1.8 的实现，并讨论其优劣以及改进方向。特别关注代码库中比较函数使用混乱的问题。

## pscm_cc 的实现

### 核心函数

#### 1. `_eq` - 内部比较函数（语义混乱）

```c
bool _eq(SCM *lhs, SCM *rhs) {
  // 特殊处理：允许 NUM <-> FLOAT 比较
  if ((lhs->type == SCM::NUM && rhs->type == SCM::FLOAT) ||
      (lhs->type == SCM::FLOAT && rhs->type == SCM::NUM)) {
    return _number_eq(lhs, rhs);
  }
  
  if (lhs->type != rhs->type) {
    return false;
  }

  switch (lhs->type) {
  case SCM::LIST:
    // 问题：使用深度比较，不符合 R4RS 的 eq? 语义
    // 注释说"为了向后兼容"，但这是错误的
    return _list_eq(lhs, rhs);
  case SCM::NUM:
  case SCM::FLOAT:
    return _number_eq(lhs, rhs);
  case SCM::CHAR:
    return ptr_to_char(lhs->value) == ptr_to_char(rhs->value);
  // ... 其他类型
  }
}
```

**问题：**
- **语义不清晰**：既不是 `eq?`（列表使用深度比较），也不是 `eqv?`（没有先检查指针相等）
- **列表比较错误**：对列表使用深度比较，不符合 R4RS 规范
- **被多处混用**：在 `case.cc`、`alist.cc`、`hash_table.cc` 等地方被当作 `eqv?` 或 `equal?` 使用

#### 2. `scm_c_is_eq` - eq? 实现

```c
SCM *scm_c_is_eq(SCM *lhs, SCM *rhs) {
  // 指针相等
  if (lhs == rhs) {
    return scm_bool_true();
  }
  
  // 特殊处理：允许 NUM <-> FLOAT 比较
  if ((lhs->type == SCM::NUM && rhs->type == SCM::FLOAT) ||
      (lhs->type == SCM::FLOAT && rhs->type == SCM::NUM)) {
    return _number_eq(lhs, rhs) ? scm_bool_true() : scm_bool_false();
  }
  
  switch (lhs->type) {
  case SCM::LIST:
  case SCM::VECTOR:
  case SCM::PROC:
    // 对于复合类型，eq? 使用指针相等
    return scm_bool_false();  // 如果指针不相等，返回 #f
  case SCM::NUM:
  case SCM::FLOAT:
  case SCM::RATIO:
    // 问题：对于数字，使用值比较，不符合 R4RS
    return _number_eq(lhs, rhs) ? scm_bool_true() : scm_bool_false();
  case SCM::CHAR:
    // 问题：对于字符，使用值比较，不符合 R4RS
    return (ch_lhs == ch_rhs) ? scm_bool_true() : scm_bool_false();
  // ...
  }
}
```

**问题：**
- **数字比较错误**：R4RS 规定 `eq?` 对数字的返回值是未指定的，不应该使用值比较
- **字符比较错误**：R4RS 规定 `eq?` 对字符的返回值是未指定的，不应该使用值比较
- **应该只使用指针比较**：除了立即值（`#t`、`#f`、`()`），其他都应该使用指针比较

#### 3. `scm_c_is_eqv` - eqv? 实现

```c
SCM *scm_c_is_eqv(SCM *lhs, SCM *rhs) {
  // 指针相等
  if (lhs == rhs) {
    return scm_bool_true();
  }
  
  // 特殊处理：允许 NUM <-> FLOAT 比较
  if ((lhs->type == SCM::NUM && rhs->type == SCM::FLOAT) ||
      (lhs->type == SCM::FLOAT && rhs->type == SCM::NUM)) {
    return _number_eq(lhs, rhs) ? scm_bool_true() : scm_bool_false();
  }
  
  // 类型不同，返回 #f
  if (lhs->type != rhs->type) {
    return scm_bool_false();
  }
  
  switch (lhs->type) {
  case SCM::LIST:
  case SCM::VECTOR:
    // 问题：对于列表和向量，eqv? 应该使用指针相等（和 eq? 一样）
    return scm_bool_false();
  case SCM::NUM:
  case SCM::FLOAT:
  case SCM::RATIO:
    // 对于数字，使用值比较（正确）
    return _number_eq(lhs, rhs) ? scm_bool_true() : scm_bool_false();
  case SCM::CHAR:
    // 对于字符，使用值比较（正确）
    return (ch_lhs == ch_rhs) ? scm_bool_true() : scm_bool_false();
  // ...
  }
}
```

**问题：**
- **实现基本正确**：对数字和字符使用值比较，对复合类型使用指针比较
- **但和 `scm_c_is_eq` 几乎一样**：缺少对精确/不精确数字的区分

#### 4. `scm_c_is_equal` - equal? 实现

```c
SCM *scm_c_is_equal(SCM *lhs, SCM *rhs) {
  return _equal_recursive(lhs, rhs) ? scm_bool_true() : scm_bool_false();
}

static bool _equal_recursive(SCM *lhs, SCM *rhs) {
  if (lhs == rhs) {
    return true;
  }
  
  if (lhs->type != rhs->type) {
    // 特殊处理：允许 NUM <-> FLOAT 比较
    if ((lhs->type == SCM::NUM && rhs->type == SCM::FLOAT) ||
        (lhs->type == SCM::FLOAT && rhs->type == SCM::NUM)) {
      return _number_eq(lhs, rhs);
    }
    return false;
  }
  
  switch (lhs->type) {
  case SCM::LIST:
    return _list_equal(lhs, rhs);  // 递归比较列表
  case SCM::VECTOR:
    return _vector_equal(lhs, rhs);  // 递归比较向量
  case SCM::PROC:
  case SCM::CONT:
    // 对于复合类型，equal? 使用指针相等（和 eq? 一样）
    return false;
  // ...
  }
}
```

**实现基本正确**：递归比较列表和向量，对过程等使用指针比较。

### 代码库中的混乱使用

#### 1. `case.cc` 中的使用

```c
// case.cc
static bool value_in_datums(SCM *value, SCM_List *datums) {
  for (SCM_List *it = datums; it; it = it->next) {
    if (it->data && _eq(value, it->data)) {  // 使用 _eq
      return true;
    }
  }
  return false;
}
```

**问题：**
- 注释说"使用 eqv? 进行比较"，但实际使用 `_eq`
- `_eq` 对列表使用深度比较，不符合 case 的需求
- 应该使用 `eqv?` 语义（先指针比较，再值比较）

#### 2. `list.cc` 中的重复定义

```c
// list.cc
static bool _eqv(SCM *lhs, SCM *rhs) {
  if (lhs == rhs) return true;
  // ... 简化的 eqv? 实现
  return lhs == rhs;
}

// memv 使用这个 _eqv
SCM *scm_c_memv(SCM *obj, SCM *lst) {
  // ...
  if (current->data && _eqv(obj, current->data)) {  // 使用局部 _eqv
    return wrap(current);
  }
}

// member 使用 _eq（错误！）
SCM *scm_c_member(SCM *obj, SCM *lst) {
  extern bool _eq(SCM *lhs, SCM *rhs);
  if (current->data && _eq(obj, current->data)) {  // 应该使用 equal?
    return wrap(current);
  }
}
```

**问题：**
- `list.cc` 中定义了另一个 `_eqv` 函数，与 `eq.cc` 中的实现重复
- `member` 使用 `_eq`，但应该使用 `equal?` 语义

#### 3. `alist.cc` 中的自定义函数

```c
// alist.cc
static bool is_eq_pointer(SCM *lhs, SCM *rhs) {
  if (lhs == rhs) {
    return true;
  }
  // 对符号使用内容比较（因为符号是 interned 的）
  if (is_sym(lhs) && is_sym(rhs)) {
    return memcmp(sym1->data, sym2->data, sym1->len) == 0;
  }
  // 对数字使用值比较
  if (is_num(lhs)) {
    return lhs->value == rhs->value;
  }
  return lhs == rhs;
}

// assq 使用 is_eq_pointer
SCM *scm_c_assq(SCM *key, SCM *alist) {
  if (pair->data && is_eq_pointer(key, pair->data)) {  // 自定义函数
    return l->data;
  }
}

// assoc 使用 _eq（错误！）
SCM *scm_c_assoc(SCM *key, SCM *alist) {
  if (pair->data && _eq(key, pair->data)) {  // 应该使用 equal?
    return l->data;
  }
}
```

**问题：**
- 定义了 `is_eq_pointer` 函数，语义与 `eq?` 类似但不完全一致
- `assoc` 使用 `_eq`，但应该使用 `equal?` 语义

#### 4. `hash_table.cc` 中的自定义函数

```c
// hash_table.cc
static bool scm_cmp_eq(SCM *lhs, SCM *rhs) {
  // 对符号使用内容比较
  if (is_sym(lhs) && is_sym(rhs)) {
    return memcmp(sym1->data, sym2->data, sym1->len) == 0;
  }
  return lhs == rhs;
}

static bool scm_cmp_eqv(SCM *lhs, SCM *rhs) {
  if (lhs == rhs) return true;
  // ... 简化的 eqv? 实现
  return lhs == rhs;
}

static bool scm_cmp_equal(SCM *lhs, SCM *rhs) {
  extern bool _eq(SCM *lhs, SCM *rhs);
  return _eq(lhs, rhs);  // 使用 _eq，但 _eq 的语义不正确
}
```

**问题：**
- 定义了三个自定义比较函数，与 `eq.cc` 中的实现重复
- `scm_cmp_equal` 使用 `_eq`，但 `_eq` 的语义不正确（对列表使用深度比较，但不对向量递归）

### 优势

1. **功能基本完整**：实现了 `eq?`、`eqv?`、`equal?` 三个函数
2. **代码结构清晰**：函数分离明确
3. **支持多种类型**：支持数字、字符、列表、向量等

### 劣势

1. **语义混乱**：
   - `_eq` 的语义不清晰（既不是 `eq?` 也不是 `eqv?`）
   - `scm_c_is_eq` 对数字和字符使用值比较，不符合 R4RS
2. **代码重复**：
   - `list.cc` 中定义了 `_eqv`
   - `alist.cc` 中定义了 `is_eq_pointer`
   - `hash_table.cc` 中定义了 `scm_cmp_eq`、`scm_cmp_eqv`、`scm_cmp_equal`
3. **使用混乱**：
   - `case.cc` 使用 `_eq` 但应该使用 `eqv?`
   - `member` 使用 `_eq` 但应该使用 `equal?`
   - `assoc` 使用 `_eq` 但应该使用 `equal?`
   - `hash_table.cc` 的 `scm_cmp_equal` 使用 `_eq` 但语义不正确
4. **缺少优化**：
   - 没有针对常见情况的优化
   - 没有使用宏来内联简单比较

## Guile 1.8 的实现

### 核心机制

#### 1. `scm_is_eq` - 底层指针比较宏

```c
// __scm.h 或类似文件
#define scm_is_eq(x, y) ((x) == (y))
```

**特点：**
- 简单的指针比较宏
- 用于内部快速比较
- 不处理数字和字符的特殊情况

#### 2. `scm_eq_p` - eq? 实现

```c
SCM_DEFINE1 (scm_eq_p, "eq?", scm_tc7_rpsubr, (SCM x, SCM y), ...)
{
  return scm_from_bool (scm_is_eq (x, y));
}
```

**特点：**
- **简单直接**：只使用指针比较
- **符合 R4RS**：对数字和字符的返回值是未指定的（由指针比较决定）
- **性能高**：只是一个宏调用

#### 3. `scm_eqv_p` - eqv? 实现

```c
SCM_PRIMITIVE_GENERIC_1 (scm_eqv_p, "eqv?", scm_tc7_rpsubr, (SCM x, SCM y), ...)
{
  // 1. 先检查指针相等
  if (scm_is_eq (x, y))
    return SCM_BOOL_T;
  
  // 2. 检查立即值
  if (SCM_IMP (x))
    return SCM_BOOL_F;
  if (SCM_IMP (y))
    return SCM_BOOL_F;
  
  // 3. 检查类型
  if (SCM_CELL_TYPE (x) != SCM_CELL_TYPE (y))
    {
      // 处理混合类型（如精确/不精确数字）
      // ...
      return SCM_BOOL_F;
    }
  
  // 4. 对数字进行值比较
  if (SCM_NUMP (x))
    {
      if (SCM_BIGP (x)) {
        return scm_from_bool (scm_i_bigcmp (x, y) == 0);
      } else if (SCM_REALP (x)) {
        // 使用特殊的浮点数比较（处理 NaN、+0/-0）
        return scm_from_bool (real_eqv (SCM_REAL_VALUE (x), SCM_REAL_VALUE (y)));
      } else if (SCM_FRACTIONP (x)) {
        return scm_i_fraction_equalp (x, y);
      } else { /* complex */
        return scm_from_bool (real_eqv (SCM_COMPLEX_REAL (x), SCM_COMPLEX_REAL (y)) 
                             && real_eqv (SCM_COMPLEX_IMAG (x), SCM_COMPLEX_IMAG (y)));
      }
    }
  
  // 5. 其他类型使用指针比较（和 eq? 一样）
  return SCM_BOOL_F;
}
```

**关键特性：**
- **双重检查**：先检查指针相等（快速路径），再检查值相等
- **特殊处理**：对浮点数使用 `real_eqv`（处理 NaN、+0/-0）
- **精确/不精确区分**：精确数字和不精确数字不相等
- **符合 R4RS**：对复合类型使用指针比较

#### 4. `scm_equal_p` - equal? 实现

```c
SCM_PRIMITIVE_GENERIC_1 (scm_equal_p, "equal?", scm_tc7_rpsubr, (SCM x, SCM y), ...)
{
tailrecurse:
  // 1. 先检查指针相等
  if (scm_is_eq (x, y))
    return SCM_BOOL_T;
  
  // 2. 检查立即值
  if (SCM_IMP (x))
    return SCM_BOOL_F;
  if (SCM_IMP (y))
    return SCM_BOOL_F;
  
  // 3. 递归比较列表
  if (scm_is_pair (x) && scm_is_pair (y))
    {
      if (scm_is_false (scm_equal_p (SCM_CAR (x), SCM_CAR (y))))
        return SCM_BOOL_F;
      x = SCM_CDR(x);
      y = SCM_CDR(y);
      goto tailrecurse;  // 尾递归优化
    }
  
  // 4. 比较字符串
  if (SCM_TYP7 (x) == scm_tc7_string && SCM_TYP7 (y) == scm_tc7_string)
    return scm_string_equal_p (x, y);
  
  // 5. 比较向量
  if (SCM_TYP7 (x) == scm_tc7_vector)
    return scm_i_vector_equal_p (x, y);
  
  // 6. 比较数字（使用 eqv? 语义）
  if (SCM_NUMP (x))
    {
      // ... 使用 eqv? 的比较逻辑
    }
  
  // 7. 其他类型使用指针比较
  return SCM_BOOL_F;
}
```

**关键特性：**
- **尾递归优化**：使用 `goto tailrecurse` 实现
- **递归比较**：对列表、向量递归比较
- **数字比较**：使用 `eqv?` 语义
- **其他类型**：使用指针比较

### Guile 1.8 的优势

1. **语义正确**：
   - `eq?` 只使用指针比较
   - `eqv?` 对数字和字符使用值比较，对复合类型使用指针比较
   - `equal?` 递归比较列表和向量
2. **性能优化**：
   - 使用宏进行快速比较
   - 尾递归优化
   - 先检查指针相等（快速路径）
3. **代码统一**：
   - 所有比较都使用 `scm_is_eq` 宏
   - 没有重复的实现
4. **特殊处理**：
   - 正确处理浮点数的 NaN、+0/-0
   - 区分精确/不精确数字

### Guile 1.8 的劣势

1. **实现复杂**：
   - 需要处理多种数字类型
   - 需要处理 Smob 的泛型函数
2. **依赖其他组件**：
   - 依赖类型系统
   - 依赖泛型函数系统

## 对比总结

| 特性 | pscm_cc | Guile 1.8 |
|------|---------|-----------|
| **eq? 语义** | ❌ 对数字/字符使用值比较 | ✅ 只使用指针比较 |
| **eqv? 语义** | ⚠️ 基本正确，但缺少特殊处理 | ✅ 正确处理数字/字符 |
| **equal? 语义** | ✅ 基本正确 | ✅ 正确，有尾递归优化 |
| **代码统一性** | ❌ 多处重复定义 | ✅ 统一使用宏 |
| **使用一致性** | ❌ 混用不同函数 | ✅ 统一使用 |
| **性能** | ⚠️ 一般 | ✅ 优化良好 |
| **特殊处理** | ❌ 缺少 NaN、+0/-0 处理 | ✅ 正确处理 |

## 关键差异

### 1. eq? 的实现

**pscm_cc：**
```c
// 对数字使用值比较（错误！）
case SCM::NUM:
  return _number_eq(lhs, rhs) ? scm_bool_true() : scm_bool_false();
```

**Guile 1.8：**
```c
// 只使用指针比较（正确）
return scm_from_bool (scm_is_eq (x, y));
```

### 2. eqv? 的实现

**pscm_cc：**
```c
// 基本正确，但缺少特殊处理
case SCM::NUM:
  return _number_eq(lhs, rhs) ? scm_bool_true() : scm_bool_false();
```

**Guile 1.8：**
```c
// 先检查指针相等
if (scm_is_eq (x, y))
  return SCM_BOOL_T;
// 然后对数字进行特殊处理（NaN、+0/-0）
if (SCM_REALP (x)) {
  return scm_from_bool (real_eqv (SCM_REAL_VALUE (x), SCM_REAL_VALUE (y)));
}
```

### 3. 代码组织

**pscm_cc：**
```c
// 多处重复定义
// eq.cc: _eq, scm_c_is_eq, scm_c_is_eqv, scm_c_is_equal
// list.cc: _eqv (重复)
// alist.cc: is_eq_pointer (重复)
// hash_table.cc: scm_cmp_eq, scm_cmp_eqv, scm_cmp_equal (重复)
```

**Guile 1.8：**
```c
// 统一使用宏
#define scm_is_eq(x, y) ((x) == (y))
// 所有比较都基于这个宏
```

## pscm_cc 的改进方向

### 1. 短期改进（高优先级）

#### 1.1 修复 `scm_c_is_eq` 的语义

**问题**：对数字和字符使用值比较，不符合 R4RS。

**改进**：只使用指针比较。

```c
SCM *scm_c_is_eq(SCM *lhs, SCM *rhs) {
  // 只使用指针比较
  return (lhs == rhs) ? scm_bool_true() : scm_bool_false();
}
```

**注意**：R4RS 规定 `eq?` 对数字和字符的返回值是未指定的，由实现决定。但通常实现只使用指针比较。

#### 1.2 修复 `_eq` 的语义或重命名

**问题**：`_eq` 的语义不清晰，被多处混用。

**改进方案 A：重命名为 `_eqv_internal`**

```c
// 重命名为 _eqv_internal，明确语义
bool _eqv_internal(SCM *lhs, SCM *rhs) {
  // 先检查指针相等
  if (lhs == rhs) {
    return true;
  }
  
  // 类型不同，返回 false
  if (lhs->type != rhs->type) {
    // 特殊处理：允许 NUM <-> FLOAT 比较
    if ((lhs->type == SCM::NUM && rhs->type == SCM::FLOAT) ||
        (lhs->type == SCM::FLOAT && rhs->type == SCM::NUM)) {
      return _number_eq(lhs, rhs);
    }
    return false;
  }
  
  // 对数字和字符使用值比较
  if (is_num(lhs) || is_float(lhs) || is_ratio(lhs)) {
    return _number_eq(lhs, rhs);
  }
  if (is_char(lhs)) {
    return ptr_to_char(lhs->value) == ptr_to_char(rhs->value);
  }
  if (is_bool(lhs)) {
    return is_true(lhs) == is_true(rhs);
  }
  
  // 其他类型使用指针比较
  return false;
}
```

**改进方案 B：删除 `_eq`，统一使用公共函数**

```c
// 删除 _eq，统一使用以下函数：
// - scm_c_is_eq_ptr: 指针比较（用于 eq?）
// - scm_c_is_eqv_internal: 值比较（用于 eqv?）
// - scm_c_is_equal_internal: 深度比较（用于 equal?）
```

#### 1.3 统一比较函数接口

**问题**：多处重复定义比较函数。

**改进**：在 `eq.cc` 中定义统一的内部比较函数。

```c
// eq.cc
// 内部比较函数（返回 bool，用于 C++ 代码）
bool scm_is_eq_ptr(SCM *lhs, SCM *rhs) {
  return lhs == rhs;
}

bool scm_is_eqv_internal(SCM *lhs, SCM *rhs) {
  if (lhs == rhs) return true;
  if (lhs->type != rhs->type) {
    // 特殊处理 NUM <-> FLOAT
    if ((lhs->type == SCM::NUM && rhs->type == SCM::FLOAT) ||
        (lhs->type == SCM::FLOAT && rhs->type == SCM::NUM)) {
      return _number_eq(lhs, rhs);
    }
    return false;
  }
  // 对数字、字符使用值比较
  if (is_num(lhs) || is_float(lhs) || is_ratio(lhs)) {
    return _number_eq(lhs, rhs);
  }
  if (is_char(lhs)) {
    return ptr_to_char(lhs->value) == ptr_to_char(rhs->value);
  }
  if (is_bool(lhs)) {
    return is_true(lhs) == is_true(rhs);
  }
  // 其他类型使用指针比较
  return false;
}

bool scm_is_equal_internal(SCM *lhs, SCM *rhs) {
  return _equal_recursive(lhs, rhs);
}

// Scheme 接口函数（返回 SCM*，用于 Scheme 调用）
SCM *scm_c_is_eq(SCM *lhs, SCM *rhs) {
  return scm_is_eq_ptr(lhs, rhs) ? scm_bool_true() : scm_bool_false();
}

SCM *scm_c_is_eqv(SCM *lhs, SCM *rhs) {
  return scm_is_eqv_internal(lhs, rhs) ? scm_bool_true() : scm_bool_false();
}

SCM *scm_c_is_equal(SCM *lhs, SCM *rhs) {
  return scm_is_equal_internal(lhs, rhs) ? scm_bool_true() : scm_bool_false();
}
```

#### 1.4 修复代码库中的混用

**修复 `case.cc`：**

```c
// case.cc
#include "eq.h"  // 引入统一的比较函数

static bool value_in_datums(SCM *value, SCM_List *datums) {
  for (SCM_List *it = datums; it; it = it->next) {
    if (it->data && scm_is_eqv_internal(value, it->data)) {  // 使用 eqv?
      return true;
    }
  }
  return false;
}
```

**修复 `list.cc`：**

```c
// list.cc
#include "eq.h"  // 引入统一的比较函数

// 删除局部 _eqv 定义

SCM *scm_c_memv(SCM *obj, SCM *lst) {
  // ...
  if (current->data && scm_is_eqv_internal(obj, current->data)) {  // 使用统一函数
    return wrap(current);
  }
}

SCM *scm_c_member(SCM *obj, SCM *lst) {
  // ...
  if (current->data && scm_is_equal_internal(obj, current->data)) {  // 使用 equal?
    return wrap(current);
  }
}
```

**修复 `alist.cc`：**

```c
// alist.cc
#include "eq.h"  // 引入统一的比较函数

// 删除 is_eq_pointer 定义

SCM *scm_c_assq(SCM *key, SCM *alist) {
  // ...
  if (pair->data && scm_is_eq_ptr(key, pair->data)) {  // 使用 eq?
    return l->data;
  }
}

SCM *scm_c_assoc(SCM *key, SCM *alist) {
  // ...
  if (pair->data && scm_is_equal_internal(key, pair->data)) {  // 使用 equal?
    return l->data;
  }
}
```

**修复 `hash_table.cc`：**

```c
// hash_table.cc
#include "eq.h"  // 引入统一的比较函数

// 删除 scm_cmp_eq, scm_cmp_eqv, scm_cmp_equal 定义

// 使用统一函数
static bool scm_cmp_eq(SCM *lhs, SCM *rhs) {
  return scm_is_eq_ptr(lhs, rhs);
}

static bool scm_cmp_eqv(SCM *lhs, SCM *rhs) {
  return scm_is_eqv_internal(lhs, rhs);
}

static bool scm_cmp_equal(SCM *lhs, SCM *rhs) {
  return scm_is_equal_internal(lhs, rhs);
}
```

### 2. 中期改进（中优先级）

#### 2.1 添加头文件 `eq.h`

**改进**：创建统一的头文件，导出内部比较函数。

```c
// eq.h
#ifndef PSCM_EQ_H
#define PSCM_EQ_H

#include "pscm.h"

// 内部比较函数（返回 bool，用于 C++ 代码）
bool scm_is_eq_ptr(SCM *lhs, SCM *rhs);
bool scm_is_eqv_internal(SCM *lhs, SCM *rhs);
bool scm_is_equal_internal(SCM *lhs, SCM *rhs);

// Scheme 接口函数（返回 SCM*，用于 Scheme 调用）
SCM *scm_c_is_eq(SCM *lhs, SCM *rhs);
SCM *scm_c_is_eqv(SCM *lhs, SCM *rhs);
SCM *scm_c_is_equal(SCM *lhs, SCM *rhs);

#endif // PSCM_EQ_H
```

#### 2.2 优化 `scm_c_is_eqv` 的特殊处理

**改进**：添加对 NaN、+0/-0 的特殊处理。

```c
// 浮点数比较（处理 NaN、+0/-0）
static bool real_eqv(double x, double y) {
  // 使用 memcmp 比较位模式（可以区分 +0 和 -0）
  if (memcmp(&x, &y, sizeof(double)) == 0) {
    return true;
  }
  // NaN 比较：两个 NaN 应该相等
  if (x != x && y != y) {
    return true;
  }
  return false;
}

bool scm_is_eqv_internal(SCM *lhs, SCM *rhs) {
  if (lhs == rhs) return true;
  // ...
  if (is_float(lhs)) {
    double x = scm_to_double(lhs);
    double y = scm_to_double(rhs);
    return real_eqv(x, y);
  }
  // ...
}
```

#### 2.3 添加精确/不精确数字区分

**改进**：区分精确数字和不精确数字。

```c
bool scm_is_eqv_internal(SCM *lhs, SCM *rhs) {
  if (lhs == rhs) return true;
  // ...
  // 精确数字和不精确数字不相等
  if ((is_num(lhs) && is_float(rhs)) || (is_float(lhs) && is_num(rhs))) {
    return false;  // 即使值相同，也不相等
  }
  // ...
}
```

#### 2.4 优化 `scm_c_is_equal` 的尾递归

**改进**：使用 `goto` 实现尾递归优化。

```c
bool scm_is_equal_internal(SCM *lhs, SCM *rhs) {
tailrecurse:
  if (lhs == rhs) return true;
  // ...
  if (is_pair(lhs) && is_pair(rhs)) {
    if (!scm_is_equal_internal(car(lhs), car(rhs))) {
      return false;
    }
    lhs = cdr(lhs);
    rhs = cdr(rhs);
    goto tailrecurse;  // 尾递归优化
  }
  // ...
}
```

### 3. 长期改进（低优先级）

#### 3.1 使用宏优化常见情况

**改进**：定义宏来内联简单比较。

```c
// eq.h
#define SCM_IS_EQ_PTR(x, y) ((x) == (y))

// 在代码中使用
if (SCM_IS_EQ_PTR(lhs, rhs)) {
  // ...
}
```

#### 3.2 添加性能分析

**改进**：添加性能计数器。

```c
// 统计比较函数调用次数
void collect_eq_stats(const char *func_name) {
  stats.eq_count++;
  // ...
}
```

#### 3.3 支持循环检测

**改进**：在 `equal?` 中添加循环检测。

```c
bool scm_is_equal_internal(SCM *lhs, SCM *rhs) {
  // 使用哈希表记录已比较的对
  std::set<std::pair<SCM*, SCM*>> visited;
  return scm_is_equal_internal_recursive(lhs, rhs, &visited);
}
```

## 改进方案总结

### 阶段 1：修复语义和统一接口（高优先级）

1. **修复 `scm_c_is_eq`**：只使用指针比较
2. **重命名或删除 `_eq`**：统一使用 `scm_is_eqv_internal` 或 `scm_is_equal_internal`
3. **创建 `eq.h` 头文件**：导出统一的比较函数
4. **修复代码库中的混用**：
   - `case.cc`：使用 `scm_is_eqv_internal`
   - `list.cc`：删除局部 `_eqv`，使用统一函数
   - `alist.cc`：删除 `is_eq_pointer`，使用统一函数
   - `hash_table.cc`：删除自定义函数，使用统一函数

### 阶段 2：优化和特殊处理（中优先级）

1. **优化 `scm_c_is_eqv`**：添加 NaN、+0/-0 处理
2. **区分精确/不精确数字**：在 `eqv?` 中区分
3. **优化 `equal?`**：添加尾递归优化

### 阶段 3：高级优化（低优先级）

1. **使用宏优化**：定义内联宏
2. **性能分析**：添加性能计数器
3. **循环检测**：在 `equal?` 中检测循环结构

## 实现示例

### 改进后的 `eq.cc`（简化版）

```c
// eq.cc
#include "pscm.h"
#include "eval.h"
#include "eq.h"

// 内部比较函数（返回 bool）
bool scm_is_eq_ptr(SCM *lhs, SCM *rhs) {
  return lhs == rhs;
}

// 浮点数比较（处理 NaN、+0/-0）
static bool real_eqv(double x, double y) {
  if (memcmp(&x, &y, sizeof(double)) == 0) {
    return true;
  }
  if (x != x && y != y) {  // 两个 NaN
    return true;
  }
  return false;
}

bool scm_is_eqv_internal(SCM *lhs, SCM *rhs) {
  // 1. 先检查指针相等
  if (lhs == rhs) {
    return true;
  }
  
  // 2. 类型不同
  if (lhs->type != rhs->type) {
    // 特殊处理：允许 NUM <-> FLOAT 比较（但值必须相等）
    if ((lhs->type == SCM::NUM && rhs->type == SCM::FLOAT) ||
        (lhs->type == SCM::FLOAT && rhs->type == SCM::NUM)) {
      return _number_eq(lhs, rhs);
    }
    return false;
  }
  
  // 3. 对数字使用值比较
  if (is_num(lhs) || is_float(lhs) || is_ratio(lhs)) {
    if (is_float(lhs)) {
      return real_eqv(scm_to_double(lhs), scm_to_double(rhs));
    }
    return _number_eq(lhs, rhs);
  }
  
  // 4. 对字符使用值比较
  if (is_char(lhs)) {
    return ptr_to_char(lhs->value) == ptr_to_char(rhs->value);
  }
  
  // 5. 对布尔值使用值比较
  if (is_bool(lhs)) {
    return is_true(lhs) == is_true(rhs);
  }
  
  // 6. 其他类型使用指针比较（和 eq? 一样）
  return false;
}

// equal? 的递归实现
static bool _equal_recursive(SCM *lhs, SCM *rhs) {
  if (lhs == rhs) {
    return true;
  }
  
  if (lhs->type != rhs->type) {
    // 特殊处理：允许 NUM <-> FLOAT 比较
    if ((lhs->type == SCM::NUM && rhs->type == SCM::FLOAT) ||
        (lhs->type == SCM::FLOAT && rhs->type == SCM::NUM)) {
      return _number_eq(lhs, rhs);
    }
    return false;
  }
  
  switch (lhs->type) {
  case SCM::LIST:
    return _list_equal(lhs, rhs);
  case SCM::VECTOR:
    return _vector_equal(lhs, rhs);
  case SCM::NUM:
  case SCM::FLOAT:
  case SCM::RATIO:
    return _number_eq(lhs, rhs);
  case SCM::CHAR:
    return ptr_to_char(lhs->value) == ptr_to_char(rhs->value);
  case SCM::BOOL:
    return is_true(lhs) == is_true(rhs);
  case SCM::SYM:
  case SCM::STR:
    return _sym_eq(lhs, rhs);
  default:
    // 其他类型使用指针比较
    return false;
  }
}

bool scm_is_equal_internal(SCM *lhs, SCM *rhs) {
tailrecurse:
  if (lhs == rhs) {
    return true;
  }
  
  // 递归比较列表
  if (is_pair(lhs) && is_pair(rhs)) {
    if (!_equal_recursive(car(lhs), car(rhs))) {
      return false;
    }
    lhs = cdr(lhs);
    rhs = cdr(rhs);
    goto tailrecurse;  // 尾递归优化
  }
  
  // 其他情况使用递归实现
  return _equal_recursive(lhs, rhs);
}

// Scheme 接口函数（返回 SCM*）
SCM *scm_c_is_eq(SCM *lhs, SCM *rhs) {
  return scm_is_eq_ptr(lhs, rhs) ? scm_bool_true() : scm_bool_false();
}

SCM *scm_c_is_eqv(SCM *lhs, SCM *rhs) {
  return scm_is_eqv_internal(lhs, rhs) ? scm_bool_true() : scm_bool_false();
}

SCM *scm_c_is_equal(SCM *lhs, SCM *rhs) {
  return scm_is_equal_internal(lhs, rhs) ? scm_bool_true() : scm_bool_false();
}

void init_eq() {
  scm_define_function("eq?", 2, 0, 0, scm_c_is_eq);
  scm_define_function("eqv?", 2, 0, 0, scm_c_is_eqv);
  scm_define_function("equal?", 2, 0, 0, scm_c_is_equal);
}
```

### 改进后的 `eq.h`

```c
// eq.h
#ifndef PSCM_EQ_H
#define PSCM_EQ_H

#include "pscm.h"

// 内部比较函数（返回 bool，用于 C++ 代码）
bool scm_is_eq_ptr(SCM *lhs, SCM *rhs);
bool scm_is_eqv_internal(SCM *lhs, SCM *rhs);
bool scm_is_equal_internal(SCM *lhs, SCM *rhs);

// Scheme 接口函数（返回 SCM*，用于 Scheme 调用）
SCM *scm_c_is_eq(SCM *lhs, SCM *rhs);
SCM *scm_c_is_eqv(SCM *lhs, SCM *rhs);
SCM *scm_c_is_equal(SCM *lhs, SCM *rhs);

#endif // PSCM_EQ_H
```

## 结论

pscm_cc 的 eq/eqv/equal 实现存在以下主要问题：

1. **语义错误**：`scm_c_is_eq` 对数字和字符使用值比较，不符合 R4RS
2. **语义混乱**：`_eq` 函数的语义不清晰，既不是 `eq?` 也不是 `eqv?`
3. **代码重复**：多处重复定义比较函数
4. **使用混乱**：代码库中混用不同的比较函数，导致语义不一致

**建议优先实现：**
1. **修复 `scm_c_is_eq`**：只使用指针比较
2. **统一比较函数接口**：创建 `eq.h`，导出统一的内部比较函数
3. **修复代码库中的混用**：统一使用 `eq.h` 中的函数
4. **删除重复定义**：删除 `list.cc`、`alist.cc`、`hash_table.cc` 中的重复定义

这些改进将提高代码的正确性、一致性和可维护性，同时保持与 R4RS 规范的兼容性。
