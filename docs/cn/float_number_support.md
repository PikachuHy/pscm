# pscm_cc 浮点数支持实现说明

::: tip 状态：已实现
浮点数支持功能已在 pscm_cc 中完整实现，包括浮点数类型、解析、打印、运算和类型提升。
:::

## Guile 1.8 的浮点数实现

### 数据结构

Guile 1.8 使用单独的数据结构来存储浮点数：

```c
typedef struct scm_t_double
{
  SCM type;
  SCM pad;
  double real;
} scm_t_double;
```

### 类型系统

Guile 1.8 中，数字类型分为：
- **Fixnum**（立即数）：小整数直接编码在 SCM 值中（tagged pointer）
- **Bignum**：大整数，使用 mpz_t 存储
- **Real**（浮点数）：使用 `scm_t_double` 结构存储 `double` 值
- **Complex**：复数
- **Fraction**：分数

浮点数使用单独的类型码 `scm_tc16_real`，与整数类型不同。

### 创建浮点数

```c
SCM
scm_from_double (double val)
{
  SCM z = scm_double_cell (scm_tc16_real, 0, 0, 0);
  SCM_REAL_VALUE (z) = val;
  return z;
}
```

### 访问浮点数值

```c
#define SCM_REAL_VALUE(x) (((scm_t_double *) SCM2PTR (x))->real)
```

## pscm_cc 实际架构

### 当前数字表示

```cpp
struct SCM {
  enum Type { NONE, NIL, LIST, PROC, CONT, FUNC, NUM, FLOAT, CHAR, BOOL, SYM, STR, MACRO } type;
  void *value;  // 对于 NUM 类型存储 int64_t，对于 FLOAT 类型存储 double
  SCM_SourceLocation *source_loc;
};
```

**实现状态**：
- ✅ `NUM` 类型支持整数（`int64_t`）
- ✅ `FLOAT` 类型支持浮点数（`double`）
- ✅ `value` 字段直接存储数值（通过 `void*` 转换）
- ✅ 可以明确区分整数和浮点数

### 数字操作

数字操作已支持整数和浮点数混合运算：
- ✅ `print.cc`: 支持打印整数和浮点数
- ✅ `number.cc`: 支持整数和浮点数运算，自动类型提升

## 架构调整方案

### 方案1：添加新的 FLOAT 类型，直接使用 void* 存储（推荐）

**优点**：
- 类型系统清晰，整数和浮点数明确区分
- 符合 Scheme 的语义（exact vs inexact numbers）
- 易于扩展（未来可以添加复数、分数等）
- 与 Guile 1.8 的设计理念一致
- **不需要额外的结构体**，直接使用 `void*` 存储 `double` 值

**实现**：

```cpp
struct SCM {
  enum Type { 
    NONE, NIL, LIST, PROC, CONT, FUNC, 
    NUM,      // 整数：value 存储 int64_t（通过 void* 转换）
    FLOAT,    // 浮点数：value 直接存储 double（通过 void* 转换）
    BOOL, SYM, STR, MACRO 
  } type;
  void *value;  // 对于 NUM 存储 int64_t，对于 FLOAT 存储 double
  SCM_SourceLocation *source_loc;
};

// 不需要额外的 SCM_Float 结构体！
```

**关键点**：
- `double` 是 8 字节，`void*` 在 64 位系统上也是 8 字节
- 可以通过类型转换直接存储和读取
- 使用 `union` 或 `reinterpret_cast` 进行安全转换

**修改点**：
1. **类型定义**（`pscm.h`）：
   - 添加 `FLOAT` 类型
   - 添加 `SCM_Float` 结构体
   - 添加 `is_float()` 辅助函数

2. **解析器**（`parse.cc`）：
   - 修改 `parse_number()` 支持浮点数解析
   - 检测小数点 `.` 和科学计数法 `e/E`

3. **打印**（`print.cc`）：
   - 添加浮点数打印逻辑
   - 处理特殊值（NaN, Infinity）

4. **数字操作**（`number.cc`）：
   - 修改所有数字操作函数，支持整数和浮点数混合运算
   - 类型提升：整数 + 浮点数 → 浮点数

5. **相等性比较**（`eq.cc`）：
   - 添加浮点数相等性比较

**示例代码**：

```cpp
// 创建浮点数 - 直接使用 void* 存储
SCM *scm_from_double(double val) {
  SCM *scm = new SCM();
  scm->type = SCM::FLOAT;
  // 使用 union 或 reinterpret_cast 将 double 转换为 void*
  union {
    double d;
    void *p;
  } u;
  u.d = val;
  scm->value = u.p;
  scm->source_loc = nullptr;
  return scm;
}

// 或者使用更直接的方法（C++）
SCM *scm_from_double(double val) {
  SCM *scm = new SCM();
  scm->type = SCM::FLOAT;
  scm->value = reinterpret_cast<void*>(*reinterpret_cast<uint64_t*>(&val));
  scm->source_loc = nullptr;
  return scm;
}

// 检查是否为浮点数
inline bool is_float(SCM *scm) {
  return scm->type == SCM::FLOAT;
}

// 获取浮点数值 - 从 void* 转换回 double
inline double scm_to_double(SCM *scm) {
  if (is_float(scm)) {
    // 方法1：使用 union
    union {
      double d;
      void *p;
    } u;
    u.p = scm->value;
    return u.d;
    
    // 方法2：使用 reinterpret_cast（C++）
    // uint64_t bits = reinterpret_cast<uint64_t>(scm->value);
    // return *reinterpret_cast<double*>(&bits);
  } else if (is_num(scm)) {
    return (double)(int64_t)scm->value;
  }
  // 错误处理
  return 0.0;
}
```

**更安全的实现（推荐）**：

```cpp
// 使用辅助函数进行类型转换
inline void* double_to_ptr(double val) {
  union {
    double d;
    void *p;
  } u;
  u.d = val;
  return u.p;
}

inline double ptr_to_double(void *ptr) {
  union {
    double d;
    void *p;
  } u;
  u.p = ptr;
  return u.d;
}

// 创建浮点数
SCM *scm_from_double(double val) {
  SCM *scm = new SCM();
  scm->type = SCM::FLOAT;
  scm->value = double_to_ptr(val);
  scm->source_loc = nullptr;
  return scm;
}

// 获取浮点数值
inline double scm_to_double(SCM *scm) {
  if (is_float(scm)) {
    return ptr_to_double(scm->value);
  } else if (is_num(scm)) {
    return (double)(int64_t)scm->value;
  }
  return 0.0;
}
```

### 方案2：使用联合体（Union）存储数字

**优点**：
- 内存效率高（不需要额外的结构体）
- 统一的数字类型接口

**缺点**：
- 需要额外的标记来区分整数和浮点数
- 类型系统不够清晰
- 扩展性较差

**实现**：

```cpp
struct SCM_Number {
  enum NumType { INTEGER, FLOAT } num_type;
  union {
    int64_t int_val;
    double float_val;
  };
};

struct SCM {
  enum Type { ..., NUM, ... } type;
  void *value;  // 对于 NUM，指向 SCM_Number
  // ...
};
```

### 方案3：在 NUM 类型中使用标记位

**优点**：
- 最小改动
- 保持现有代码结构

**缺点**：
- 类型系统不够清晰
- 需要额外的检查逻辑
- 扩展性差

**实现**：

```cpp
// 使用 value 的高位作为标记
// 例如：最高位为 1 表示浮点数
// 但这会限制整数的范围
```

### 方案4：统一数字类型，内部区分

**优点**：
- 统一的数字接口
- 易于实现类型提升

**缺点**：
- 需要修改所有数字操作代码
- 运行时类型检查开销

**实现**：

```cpp
struct SCM_Number {
  bool is_float;
  union {
    int64_t int_val;
    double float_val;
  };
};

struct SCM {
  enum Type { ..., NUM, ... } type;  // 统一使用 NUM
  void *value;  // 指向 SCM_Number
  // ...
};
```

## 推荐方案：方案1（添加 FLOAT 类型）

### 实现步骤

1. **修改类型定义**（`pscm.h`）：
   ```cpp
   enum Type { ..., NUM, FLOAT, ... };
   
   inline bool is_float(SCM *scm) {
     return scm->type == SCM::FLOAT;
   }
   
   // 辅助函数：double <-> void* 转换
   inline void* double_to_ptr(double val) {
     union {
       double d;
       void *p;
     } u;
     u.d = val;
     return u.p;
   }
   
   inline double ptr_to_double(void *ptr) {
     union {
       double d;
       void *p;
     } u;
     u.p = ptr;
     return u.d;
   }
   
   // 创建浮点数
   inline SCM *scm_from_double(double val) {
     SCM *scm = new SCM();
     scm->type = SCM::FLOAT;
     scm->value = double_to_ptr(val);
     scm->source_loc = nullptr;
     return scm;
   }
   
   // 获取浮点数值
   inline double scm_to_double(SCM *scm) {
     if (is_float(scm)) {
       return ptr_to_double(scm->value);
     } else if (is_num(scm)) {
       return (double)(int64_t)scm->value;
     }
     return 0.0;
   }
   ```
   
   **注意**：不需要定义 `SCM_Float` 结构体！

2. **修改解析器**（`parse.cc`）：
   ```cpp
   static SCM *parse_number(Parser *p) {
     // ... 解析整数部分 ...
     double value = ...; // 整数部分的值
     bool negative = ...;
     
     // 检查小数点
     if (*p->pos == '.') {
       // 解析小数部分
       double fractional = 0.0;
       double divisor = 1.0;
       p->pos++;
       while (isdigit((unsigned char)*p->pos)) {
         fractional = fractional * 10 + (*p->pos - '0');
         divisor *= 10;
         p->pos++;
       }
       value += fractional / divisor;
       
       // 检查科学计数法
       if (*p->pos == 'e' || *p->pos == 'E') {
         // 解析指数部分
         // ...
       }
       
       // 创建浮点数
       SCM *scm = new SCM();
       scm->type = SCM::FLOAT;
       scm->value = double_to_ptr(negative ? -value : value);
       scm->source_loc = nullptr;
       return scm;
     }
     
     // 整数情况
     // ...
   }
   ```

3. **修改打印函数**（`print.cc`）：
   ```cpp
   if (is_float(ast)) {
     double val = ptr_to_double(ast->value);
     printf("%g", val);
     return;
   }
   ```

4. **修改数字操作**（`number.cc`）：
   ```cpp
   // 类型提升函数
   static SCM *promote_to_float(SCM *num) {
     if (is_float(num)) return num;
     if (is_num(num)) {
       return scm_from_double((double)(int64_t)num->value);
     }
     return nullptr;
   }
   
   // 修改 BinaryOperator 支持混合类型
   template <typename Op>
   struct BinaryOperator {
     static SCM *run(SCM *lhs, SCM *rhs) {
       // 类型提升：如果任一操作数是浮点数，都转换为浮点数
       bool has_float = is_float(lhs) || is_float(rhs);
       if (has_float) {
         double d_lhs = scm_to_double(lhs);
         double d_rhs = scm_to_double(rhs);
         double ret = Op::run(d_lhs, d_rhs);
         return scm_from_double(ret);
       } else {
         // 整数运算
         // ...
       }
     }
   };
   ```

5. **修改相等性比较**（`eq.cc`）：
   ```cpp
   case SCM::FLOAT:
     // 浮点数相等性比较（考虑精度）
     break;
   ```

### 类型提升规则

- 整数 + 整数 → 整数
- 整数 + 浮点数 → 浮点数
- 浮点数 + 浮点数 → 浮点数
- 整数 / 整数 → 如果整除则整数，否则浮点数（或分数）

### 与整数存储方式对比

**整数（当前实现）**：
```cpp
scm->type = SCM::NUM;
scm->value = (void *)value;  // 直接转换 int64_t -> void*
// 读取：int64_t num = (int64_t)scm->value;
```

**浮点数（新实现）**：
```cpp
scm->type = SCM::FLOAT;
scm->value = double_to_ptr(value);  // 通过 union 转换 double -> void*
// 读取：double val = ptr_to_double(scm->value);
```

两者都直接存储在 `value` 字段中，存储方式一致。

### 内存管理

- **不需要额外的内存分配**：浮点数直接存储在 `value` 字段中，不需要 `new SCM_Float()`
- 与整数存储方式一致，都直接存储在 `value` 字段中
- 如果实现 GC，只需要释放 `SCM` 结构体本身，不需要额外处理浮点数值

## 实际实现

### 实现状态

✅ **已完全实现**，采用方案1（简化版）：添加 `FLOAT` 类型，**不需要定义 `SCM_Float` 结构体**，直接使用 `void*` 存储 `double` 值。

### 实现细节

1. **类型定义**（`pscm.h`）：
   - ✅ 已添加 `SCM::FLOAT` 类型
   - ✅ 已实现 `is_float()` 函数
   - ✅ 已实现 `double_to_ptr()` 和 `ptr_to_double()` 转换函数
   - ✅ 已实现 `scm_from_double()` 和 `scm_to_double()` 函数

2. **解析器**（`parse.cc`）：
   - ✅ 已修改 `parse_number()` 支持浮点数解析
   - ✅ 支持小数点 `.` 和科学计数法 `e/E`
   - ✅ 正确创建 `FLOAT` 类型的 SCM

3. **打印函数**（`print.cc`）：
   - ✅ 已添加浮点数打印逻辑
   - ✅ 使用 `%g` 格式打印浮点数

4. **数字操作**（`number.cc`）：
   - ✅ 已实现类型提升机制（`needs_float_promotion`）
   - ✅ 所有数字操作函数支持整数和浮点数混合运算
   - ✅ `BinaryOperator` 模板支持混合类型运算

5. **相等性比较**（`eq.cc`）：
   - ✅ 已支持浮点数相等性比较
   - ✅ 支持整数和浮点数之间的比较

### 实现代码示例

实际实现与推荐方案完全一致：

```cpp
// pscm.h 中的实现
inline void* double_to_ptr(double val) {
  union {
    double d;
    void *p;
  } u;
  u.d = val;
  return u.p;
}

inline double ptr_to_double(void *ptr) {
  union {
    double d;
    void *p;
  } u;
  u.p = ptr;
  return u.d;
}

inline SCM *scm_from_double(double val) {
  SCM *scm = new SCM();
  scm->type = SCM::FLOAT;
  scm->value = double_to_ptr(val);
  scm->source_loc = nullptr;
  return scm;
}

// number.cc 中的类型提升
static bool needs_float_promotion(SCM *lhs, SCM *rhs) {
  return is_float(lhs) || is_float(rhs);
}
```

## 总结

**实际实现采用方案1（简化版）**：添加 `FLOAT` 类型，但**不需要定义 `SCM_Float` 结构体**，直接使用 `void*` 存储 `double` 值。

**关键点**：
1. ✅ 类型系统层面区分整数和浮点数（通过 `SCM::FLOAT` 类型）
2. ✅ **直接使用 `void*` 存储 `double` 值**（通过 `union` 转换）
3. ✅ 实现类型提升机制（整数自动提升为浮点数）
4. ✅ 所有数字操作支持混合类型
5. ✅ 解析器支持浮点数语法（`12.7`, `1.2e3` 等）

**优势**：
- ✅ 不需要额外的内存分配（不需要 `new SCM_Float()`）
- ✅ 与整数存储方式一致（都直接存储在 `value` 字段中）
- ✅ 代码更简洁
- ✅ 性能更好（减少一次内存分配和指针解引用）

**注意事项**：
- ✅ 使用 `union` 进行类型转换是安全的（在 C/C++ 中）
- ✅ 确保 `double` 和 `void*` 大小相同（在 64 位系统上都是 8 字节）
- ✅ 在 32 位系统上可能需要特殊处理（`double` 是 8 字节，`void*` 是 4 字节）

### 32 位系统处理（如果需要）

如果需要在 32 位系统上支持，可以使用条件编译：

```cpp
#if __SIZEOF_POINTER__ == 4
// 32 位系统：需要分配内存
SCM *scm_from_double(double val) {
  double *d = new double(val);
  SCM *scm = new SCM();
  scm->type = SCM::FLOAT;
  scm->value = d;  // 存储指针
  scm->source_loc = nullptr;
  return scm;
}

inline double scm_to_double(SCM *scm) {
  if (is_float(scm)) {
    return *(double*)scm->value;  // 解引用指针
  } else if (is_num(scm)) {
    return (double)(int64_t)scm->value;
  }
  return 0.0;
}
#else
// 64 位系统：直接转换（使用上面的实现）
#endif
```

但在大多数现代系统上（包括 macOS、Linux、Windows 64 位），都是 64 位系统，可以直接使用简单的转换方式。

