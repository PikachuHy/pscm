# pscm_cc Smob 实现方案

## 概述

本文档描述在 `pscm_cc` 中实现 Smob（Small Object）特性的方案，参考 Guile 1.8 的设计。Smob 允许用户定义新的 Scheme 对象类型，这对于扩展 pscm_cc 的功能非常重要。

## 设计目标

- 提供与 Guile 1.8 兼容的 Smob API
- 支持用户定义新的 Scheme 对象类型
- 集成到现有的类型系统和 GC 机制（未来）
- 支持自定义标记、释放、打印和相等性比较函数
- 支持可调用的 Smob（apply 函数）

## Guile 1.8 的 Smob 实现分析

### 核心数据结构

```c
typedef struct scm_smob_descriptor
{
  char const *name;              // 类型名称
  size_t size;                   // 实例大小（0 表示不使用默认内存分配）
  SCM (*mark) (SCM);             // GC 标记函数
  size_t (*free) (SCM);          // GC 释放函数
  int (*print) (SCM, SCM, scm_print_state*);  // 打印函数
  SCM (*equalp) (SCM, SCM);      // 相等性比较函数
  SCM (*apply) ();               // 可调用对象的应用函数
  SCM (*apply_0) (SCM);
  SCM (*apply_1) (SCM, SCM);
  SCM (*apply_2) (SCM, SCM, SCM);
  SCM (*apply_3) (SCM, SCM, SCM, SCM);
  int gsubr_type;
} scm_smob_descriptor;
```

### 关键特性

1. **类型注册**：通过 `scm_make_smob_type` 注册新类型，返回类型标签（tag）
2. **数据存储**：
   - 当 `size > 0` 时，smob 存储指向外部内存块的指针（通过 `SCM_SMOB_DATA` 访问）
   - 当 `size = 0` 时，smob 直接存储值（通过 `SCM_SMOBNUM` 访问）
3. **类型标签**：每个 smob 类型有唯一的标签，格式为 `scm_tc7_smob + smob_num * 256`
4. **默认函数**：提供默认的 mark、free、print 函数
5. **可调用 Smob**：支持将 Smob 作为过程调用（通过 apply 函数）

## pscm_cc 实现方案

### 1. 类型系统扩展

在 `pscm.h` 中添加 `SMOB` 类型：

```c
enum Type { 
  NONE, NIL, LIST, PROC, CONT, FUNC, NUM, FLOAT, CHAR, BOOL, 
  SYM, STR, MACRO, HASH_TABLE, RATIO, VECTOR, PORT, PROMISE, 
  MODULE, SMOB  // 新增 SMOB 类型
};
```

### 2. 数据结构设计

#### Smob 描述符

```c
struct SCM_SmobDescriptor {
  const char *name;              // 类型名称
  size_t size;                   // 实例大小（0 表示直接存储值）
  SCM *(*mark)(SCM *);           // GC 标记函数（未来使用）
  size_t (*free)(SCM *);         // GC 释放函数（未来使用）
  void (*print)(SCM *, bool);    // 打印函数（write_mode）
  SCM *(*equalp)(SCM *, SCM *);  // 相等性比较函数
  // 可调用对象的应用函数（可选）
  SCM *(*apply_0)(SCM *);
  SCM *(*apply_1)(SCM *, SCM *);
  SCM *(*apply_2)(SCM *, SCM *, SCM *);
  SCM *(*apply_3)(SCM *, SCM *, SCM *, SCM *);
};
```

#### Smob 对象

```c
struct SCM_Smob {
  long tag;                      // 类型标签（用于查找描述符）
  void *data;                    // 数据指针（当 size > 0 时）
  int64_t data2;                 // 第二个数据字段（可选）
  int64_t data3;                 // 第三个数据字段（可选）
  int64_t flags;                 // 标志位（可选）
};
```

### 3. 类型注册表

```c
#define MAX_SMOB_COUNT 256

static long g_numsmob = 0;
static SCM_SmobDescriptor g_smobs[MAX_SMOB_COUNT];
```

### 4. 核心 API 设计

#### 类型注册

```c
// 创建新的 smob 类型
long scm_make_smob_type(const char *name, size_t size);

// 设置自定义函数
void scm_set_smob_mark(long tag, SCM *(*mark)(SCM *));
void scm_set_smob_free(long tag, size_t (*free)(SCM *));
void scm_set_smob_print(long tag, void (*print)(SCM *, bool));
void scm_set_smob_equalp(long tag, SCM *(*equalp)(SCM *, SCM *));
void scm_set_smob_apply(long tag, SCM *(*apply_0)(SCM *),
                        SCM *(*apply_1)(SCM *, SCM *),
                        SCM *(*apply_2)(SCM *, SCM *, SCM *),
                        SCM *(*apply_3)(SCM *, SCM *, SCM *, SCM *));
```

#### Smob 创建

```c
// 创建 smob（使用外部数据指针）
SCM *scm_make_smob(long tag, void *data);

// 创建 smob（直接存储值，size=0）
SCM *scm_make_smob_with_data(long tag, int64_t data);

// 创建 smob（带多个数据字段）
SCM *scm_make_smob_with_data2(long tag, int64_t data1, int64_t data2);
SCM *scm_make_smob_with_data3(long tag, int64_t data1, int64_t data2, int64_t data3);
```

#### Smob 访问

```c
// 获取 smob 数据
void *scm_smob_data(SCM *smob);
int64_t scm_smob_data2(SCM *smob);
int64_t scm_smob_data3(SCM *smob);
int64_t scm_smob_flags(SCM *smob);

// 设置 smob 数据
void scm_set_smob_data(SCM *smob, void *data);
void scm_set_smob_data2(SCM *smob, int64_t data);
void scm_set_smob_data3(SCM *smob, int64_t data);
void scm_set_smob_flags(SCM *smob, int64_t flags);

// 类型检查
bool is_smob(SCM *obj);
bool is_smob_type(SCM *obj, long tag);
void scm_assert_smob_type(long tag, SCM *val);
```

### 5. 类型标签设计

类型标签格式：`tag = base_tag + smob_num`

- `base_tag`：基础标签值（例如 0x1000000）
- `smob_num`：smob 类型编号（0-255）

```c
#define SCM_SMOB_BASE_TAG 0x1000000

static inline long make_smob_tag(long smob_num) {
  return SCM_SMOB_BASE_TAG + smob_num;
}

static inline long get_smob_num(long tag) {
  return tag - SCM_SMOB_BASE_TAG;
}
```

### 6. 默认函数实现

#### 默认打印函数

```c
static void default_smob_print(SCM *smob, bool write_mode) {
  SCM_Smob *s = cast<SCM_Smob>(smob);
  long num = get_smob_num(s->tag);
  const char *name = g_smobs[num].name ? g_smobs[num].name : "smob";
  
  printf("#<%s", name);
  if (g_smobs[num].size > 0) {
    printf(" %p", s->data);
  } else {
    printf(" %lld", (long long)s->data);
  }
  printf(">");
}
```

#### 默认相等性比较函数

```c
static SCM *default_smob_equalp(SCM *obj1, SCM *obj2) {
  // 默认使用指针相等性（eq?）
  return obj1 == obj2 ? scm_bool_true() : scm_bool_false();
}
```

#### 默认标记和释放函数

```c
// 标记函数（未来 GC 使用）
static SCM *default_smob_mark(SCM *smob) {
  // 默认不标记任何对象
  return scm_nil();
}

// 释放函数（未来 GC 使用）
static size_t default_smob_free(SCM *smob) {
  SCM_Smob *s = cast<SCM_Smob>(smob);
  long num = get_smob_num(s->tag);
  
  if (g_smobs[num].size > 0 && s->data) {
    // 释放外部内存（未来实现）
    // free(s->data);
  }
  return 0;
}
```

### 7. 集成到现有系统

#### 打印系统集成

在 `print.cc` 中添加 smob 打印支持：

```c
if (is_smob(ast)) {
  SCM_Smob *s = cast<SCM_Smob>(ast);
  long num = get_smob_num(s->tag);
  
  if (g_smobs[num].print) {
    g_smobs[num].print(ast, write_mode);
  } else {
    default_smob_print(ast, write_mode);
  }
  return;
}
```

#### 相等性比较集成

在 `eq.cc` 中添加 smob 相等性比较支持：

```c
if (is_smob(obj1) && is_smob(obj2)) {
  SCM_Smob *s1 = cast<SCM_Smob>(obj1);
  SCM_Smob *s2 = cast<SCM_Smob>(obj2);
  
  // 类型必须相同
  if (s1->tag != s2->tag) {
    return scm_bool_false();
  }
  
  long num = get_smob_num(s1->tag);
  if (g_smobs[num].equalp) {
    return g_smobs[num].equalp(obj1, obj2);
  } else {
    return default_smob_equalp(obj1, obj2);
  }
}
```

#### 可调用 Smob 集成

在 `eval.cc` 中添加可调用 smob 支持：

```c
if (is_smob(l->data)) {
  SCM_Smob *s = cast<SCM_Smob>(l->data);
  long num = get_smob_num(s->tag);
  
  // 检查是否有 apply 函数
  if (g_smobs[num].apply_0 || g_smobs[num].apply_1 || 
      g_smobs[num].apply_2 || g_smobs[num].apply_3) {
    // 根据参数数量调用相应的 apply 函数
    // ...
  }
}
```

### 8. 文件组织

创建新文件 `src/c/smob.cc` 和 `src/c/smob.h`：

- `smob.h`：声明所有 smob 相关的函数和数据结构
- `smob.cc`：实现 smob 类型注册、创建、访问等功能

### 9. 初始化

在 `init.cc` 中添加 smob 初始化：

```c
void init_smob() {
  // 初始化 smob 系统
  g_numsmob = 0;
  memset(g_smobs, 0, sizeof(g_smobs));
}
```

### 10. 测试用例

#### 基础测试

```scheme
;; 定义简单的 smob 类型
(define image-tag (scm-make-smob-type "image" 0))

;; 创建 smob
(define img (scm-make-smob-with-data image-tag 42))

;; 类型检查
(scm-smob? img)  ; => #t
(scm-smob-type? img image-tag)  ; => #t

;; 访问数据
(scm-smob-data img)  ; => 42

;; 打印
(write img)  ; => #<image 42>
```

#### 自定义打印函数测试

```scheme
(define point-tag (scm-make-smob-type "point" 0))

;; 设置自定义打印函数（通过 C 函数）
(scm-set-smob-print point-tag custom-point-print)

(define p (scm-make-smob-with-data2 point-tag 10 20))
(write p)  ; => #<point (10, 20)>
```

#### 相等性比较测试

```scheme
(define point-tag (scm-make-smob-type "point" 0))

;; 设置自定义相等性比较函数
(scm-set-smob-equalp point-tag custom-point-equalp)

(define p1 (scm-make-smob-with-data2 point-tag 10 20))
(define p2 (scm-make-smob-with-data2 point-tag 10 20))
(equal? p1 p2)  ; => #t（如果自定义函数比较坐标）
```

#### 可调用 Smob 测试

```scheme
(define counter-tag (scm-make-smob-type "counter" 0))

;; 设置 apply 函数（通过 C 函数）
(scm-set-smob-apply counter-tag counter-apply-0 counter-apply-1 ...)

(define c (scm-make-smob-with-data counter-tag 0))
(c)  ; => 调用 counter-apply-0
(c 5)  ; => 调用 counter-apply-1
```

## 实现步骤

1. **阶段 1：基础框架**
   - 添加 `SMOB` 类型到类型系统
   - 实现 `SCM_Smob` 和 `SCM_SmobDescriptor` 结构
   - 实现类型注册表
   - 实现基础的类型注册和创建函数

2. **阶段 2：访问函数**
   - 实现 smob 数据访问函数
   - 实现类型检查函数
   - 实现默认打印和相等性比较函数

3. **阶段 3：系统集成**
   - 集成到打印系统
   - 集成到相等性比较系统
   - 集成到求值器（可调用 smob）

4. **阶段 4：测试和文档**
   - 编写基础测试用例
   - 编写自定义函数测试用例
   - 更新文档

## 注意事项

1. **内存管理**：当前 pscm_cc 没有 GC，所以 `mark` 和 `free` 函数暂时不实现，但保留接口以便未来添加 GC 时使用。

2. **类型标签冲突**：确保 smob 类型标签不与现有类型冲突。

3. **错误处理**：添加适当的错误检查，如类型不匹配、smob 数量超限等。

4. **兼容性**：尽量保持与 Guile 1.8 的 API 兼容，但可以简化一些复杂的特性（如 apply 函数的参数适配）。

## 参考

- Guile 1.8 源码：`guile/libguile/smob.h`、`guile/libguile/smob.c`
- Guile 文档：`guile/doc/ref/api-smobs.texi`
- Guile 示例：`guile/doc/example-smob/`

