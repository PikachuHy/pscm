#pragma once

#include "pscm.h"

// Forward declarations
struct SCM_Smob;
struct SCM_SmobDescriptor;

// Smob descriptor structure
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

// Smob object structure
struct SCM_Smob {
  long tag;                      // 类型标签（用于查找描述符）
  void *data;                    // 数据指针（当 size > 0 时）
  int64_t data2;                 // 第二个数据字段（可选）
  int64_t data3;                 // 第三个数据字段（可选）
  int64_t flags;                 // 标志位（可选）
};

// Type tag constants
#define SCM_SMOB_BASE_TAG 0x1000000
#define MAX_SMOB_COUNT 256

// Type tag helpers (implemented as inline functions)
static inline long make_smob_tag(long smob_num) {
  return SCM_SMOB_BASE_TAG + smob_num;
}

static inline long get_smob_num(long tag) {
  return tag - SCM_SMOB_BASE_TAG;
}

// Type checking (is_smob is defined in pscm.h)
static inline bool is_smob_type(SCM *scm, long tag) {
  if (!is_smob(scm)) {
    return false;
  }
  SCM_Smob *s = cast<SCM_Smob>(scm);
  return s->tag == tag;
}

// Smob type registration
long scm_make_smob_type(const char *name, size_t size);
void scm_set_smob_mark(long tag, SCM *(*mark)(SCM *));
void scm_set_smob_free(long tag, size_t (*free)(SCM *));
void scm_set_smob_print(long tag, void (*print)(SCM *, bool));
void scm_set_smob_equalp(long tag, SCM *(*equalp)(SCM *, SCM *));
void scm_set_smob_apply(long tag, 
                        SCM *(*apply_0)(SCM *),
                        SCM *(*apply_1)(SCM *, SCM *),
                        SCM *(*apply_2)(SCM *, SCM *, SCM *),
                        SCM *(*apply_3)(SCM *, SCM *, SCM *, SCM *));

// Smob creation
SCM *scm_make_smob(long tag, void *data);
SCM *scm_make_smob_with_data(long tag, int64_t data);
SCM *scm_make_smob_with_data2(long tag, int64_t data1, int64_t data2);
SCM *scm_make_smob_with_data3(long tag, int64_t data1, int64_t data2, int64_t data3);

// Smob access
void *scm_smob_data(SCM *smob);
int64_t scm_smob_data2(SCM *smob);
int64_t scm_smob_data3(SCM *smob);
int64_t scm_smob_flags(SCM *smob);
void scm_set_smob_data(SCM *smob, void *data);
void scm_set_smob_data2(SCM *smob, int64_t data);
void scm_set_smob_data3(SCM *smob, int64_t data);
void scm_set_smob_flags(SCM *smob, int64_t flags);

// Type assertion
void scm_assert_smob_type(long tag, SCM *val);

// Initialization
void init_smob();

// Helper to get smob descriptor
SCM_SmobDescriptor *scm_get_smob_descriptor(long tag);

