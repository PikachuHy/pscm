#include "pscm.h"
#include "eval.h"
#include "smob.h"
#include <string.h>
#include <stdio.h>

// Smob type registry
static long g_numsmob = 0;
static SCM_SmobDescriptor g_smobs[MAX_SMOB_COUNT];

// Default functions
static SCM *default_smob_mark(SCM *smob) {
  // Default mark function: mark nothing
  // This will be used when GC is implemented
  return scm_nil();
}

static size_t default_smob_free(SCM *smob) {
  // Default free function: free external data if size > 0
  SCM_Smob *s = cast<SCM_Smob>(smob);
  long num = get_smob_num(s->tag);
  
  if (g_smobs[num].size > 0 && s->data) {
    // Free external memory (when GC is implemented)
    // For now, we don't free anything since there's no GC
    // free(s->data);
  }
  return 0;
}

// Default print function (exported for use in print.cc)
void default_smob_print(SCM *smob, bool write_mode) {
  SCM_Smob *s = cast<SCM_Smob>(smob);
  long num = get_smob_num(s->tag);
  const char *name = g_smobs[num].name ? g_smobs[num].name : "smob";
  
  printf("#<%s", name);
  if (g_smobs[num].size > 0) {
    printf(" %p", s->data);
  } else {
    printf(" %lld", (long long)(intptr_t)s->data);
  }
  printf(">");
}

static SCM *default_smob_equalp(SCM *obj1, SCM *obj2) {
  // Default equality: pointer equality (eq?)
  return obj1 == obj2 ? scm_bool_true() : scm_bool_false();
}

// Get smob descriptor
SCM_SmobDescriptor *scm_get_smob_descriptor(long tag) {
  long num = get_smob_num(tag);
  if (num < 0 || num >= g_numsmob) {
    return nullptr;
  }
  return &g_smobs[num];
}

// Create a new smob type
long scm_make_smob_type(const char *name, size_t size) {
  if (g_numsmob >= MAX_SMOB_COUNT) {
    eval_error("scm-make-smob-type: maximum number of smobs exceeded");
    return 0;
  }
  
  long new_smob = g_numsmob++;
  
  // Initialize descriptor
  g_smobs[new_smob].name = name;
  g_smobs[new_smob].size = size;
  g_smobs[new_smob].mark = default_smob_mark;
  if (size > 0) {
    g_smobs[new_smob].free = default_smob_free;
  } else {
    g_smobs[new_smob].free = nullptr;
  }
  g_smobs[new_smob].print = default_smob_print;
  g_smobs[new_smob].equalp = default_smob_equalp;
  g_smobs[new_smob].apply_0 = nullptr;
  g_smobs[new_smob].apply_1 = nullptr;
  g_smobs[new_smob].apply_2 = nullptr;
  g_smobs[new_smob].apply_3 = nullptr;
  
  return make_smob_tag(new_smob);
}

// Set custom functions
void scm_set_smob_mark(long tag, SCM *(*mark)(SCM *)) {
  long num = get_smob_num(tag);
  if (num < 0 || num >= g_numsmob) {
    eval_error("scm-set-smob-mark: invalid smob tag");
    return;
  }
  g_smobs[num].mark = mark;
}

void scm_set_smob_free(long tag, size_t (*free)(SCM *)) {
  long num = get_smob_num(tag);
  if (num < 0 || num >= g_numsmob) {
    eval_error("scm-set-smob-free: invalid smob tag");
    return;
  }
  g_smobs[num].free = free;
}

void scm_set_smob_print(long tag, void (*print)(SCM *, bool)) {
  long num = get_smob_num(tag);
  if (num < 0 || num >= g_numsmob) {
    eval_error("scm-set-smob-print: invalid smob tag");
    return;
  }
  g_smobs[num].print = print;
}

void scm_set_smob_equalp(long tag, SCM *(*equalp)(SCM *, SCM *)) {
  long num = get_smob_num(tag);
  if (num < 0 || num >= g_numsmob) {
    eval_error("scm-set-smob-equalp: invalid smob tag");
    return;
  }
  g_smobs[num].equalp = equalp;
}

void scm_set_smob_apply(long tag, 
                        SCM *(*apply_0)(SCM *),
                        SCM *(*apply_1)(SCM *, SCM *),
                        SCM *(*apply_2)(SCM *, SCM *, SCM *),
                        SCM *(*apply_3)(SCM *, SCM *, SCM *, SCM *)) {
  long num = get_smob_num(tag);
  if (num < 0 || num >= g_numsmob) {
    eval_error("scm-set-smob-apply: invalid smob tag");
    return;
  }
  g_smobs[num].apply_0 = apply_0;
  g_smobs[num].apply_1 = apply_1;
  g_smobs[num].apply_2 = apply_2;
  g_smobs[num].apply_3 = apply_3;
}

// Create smob with external data pointer
SCM *scm_make_smob(long tag, void *data) {
  long num = get_smob_num(tag);
  if (num < 0 || num >= g_numsmob) {
    eval_error("scm-make-smob: invalid smob tag");
    return scm_none();
  }
  
  auto smob = new SCM_Smob();
  smob->tag = tag;
  smob->data = data;
  smob->data2 = 0;
  smob->data3 = 0;
  smob->flags = 0;
  
  SCM *scm = new SCM();
  scm->type = SCM::SMOB;
  scm->value = smob;
  scm->source_loc = nullptr;
  
  return scm;
}

// Create smob with direct data value (size=0)
SCM *scm_make_smob_with_data(long tag, int64_t data) {
  long num = get_smob_num(tag);
  if (num < 0 || num >= g_numsmob) {
    eval_error("scm-make-smob-with-data: invalid smob tag");
    return scm_none();
  }
  
  auto smob = new SCM_Smob();
  smob->tag = tag;
  smob->data = (void *)(intptr_t)data;  // Store value in pointer
  smob->data2 = 0;
  smob->data3 = 0;
  smob->flags = 0;
  
  SCM *scm = new SCM();
  scm->type = SCM::SMOB;
  scm->value = smob;
  scm->source_loc = nullptr;
  
  return scm;
}

// Create smob with two data fields
SCM *scm_make_smob_with_data2(long tag, int64_t data1, int64_t data2) {
  long num = get_smob_num(tag);
  if (num < 0 || num >= g_numsmob) {
    eval_error("scm-make-smob-with-data2: invalid smob tag");
    return scm_none();
  }
  
  auto smob = new SCM_Smob();
  smob->tag = tag;
  smob->data = (void *)(intptr_t)data1;
  smob->data2 = data2;
  smob->data3 = 0;
  smob->flags = 0;
  
  SCM *scm = new SCM();
  scm->type = SCM::SMOB;
  scm->value = smob;
  scm->source_loc = nullptr;
  
  return scm;
}

// Create smob with three data fields
SCM *scm_make_smob_with_data3(long tag, int64_t data1, int64_t data2, int64_t data3) {
  long num = get_smob_num(tag);
  if (num < 0 || num >= g_numsmob) {
    eval_error("scm-make-smob-with-data3: invalid smob tag");
    return scm_none();
  }
  
  auto smob = new SCM_Smob();
  smob->tag = tag;
  smob->data = (void *)(intptr_t)data1;
  smob->data2 = data2;
  smob->data3 = data3;
  smob->flags = 0;
  
  SCM *scm = new SCM();
  scm->type = SCM::SMOB;
  scm->value = smob;
  scm->source_loc = nullptr;
  
  return scm;
}

// Access smob data
void *scm_smob_data(SCM *smob) {
  if (!is_smob(smob)) {
    eval_error("scm-smob-data: expected smob");
    return nullptr;
  }
  SCM_Smob *s = cast<SCM_Smob>(smob);
  return s->data;
}

int64_t scm_smob_data2(SCM *smob) {
  if (!is_smob(smob)) {
    eval_error("scm-smob-data2: expected smob");
    return 0;
  }
  SCM_Smob *s = cast<SCM_Smob>(smob);
  return s->data2;
}

int64_t scm_smob_data3(SCM *smob) {
  if (!is_smob(smob)) {
    eval_error("scm-smob-data3: expected smob");
    return 0;
  }
  SCM_Smob *s = cast<SCM_Smob>(smob);
  return s->data3;
}

int64_t scm_smob_flags(SCM *smob) {
  if (!is_smob(smob)) {
    eval_error("scm-smob-flags: expected smob");
    return 0;
  }
  SCM_Smob *s = cast<SCM_Smob>(smob);
  return s->flags;
}

// Set smob data
void scm_set_smob_data(SCM *smob, void *data) {
  if (!is_smob(smob)) {
    eval_error("scm-set-smob-data: expected smob");
    return;
  }
  SCM_Smob *s = cast<SCM_Smob>(smob);
  s->data = data;
}

void scm_set_smob_data2(SCM *smob, int64_t data) {
  if (!is_smob(smob)) {
    eval_error("scm-set-smob-data2: expected smob");
    return;
  }
  SCM_Smob *s = cast<SCM_Smob>(smob);
  s->data2 = data;
}

void scm_set_smob_data3(SCM *smob, int64_t data) {
  if (!is_smob(smob)) {
    eval_error("scm-set-smob-data3: expected smob");
    return;
  }
  SCM_Smob *s = cast<SCM_Smob>(smob);
  s->data3 = data;
}

void scm_set_smob_flags(SCM *smob, int64_t flags) {
  if (!is_smob(smob)) {
    eval_error("scm-set-smob-flags: expected smob");
    return;
  }
  SCM_Smob *s = cast<SCM_Smob>(smob);
  s->flags = flags;
}

// Type assertion
void scm_assert_smob_type(long tag, SCM *val) {
  if (!is_smob_type(val, tag)) {
    SCM_SmobDescriptor *desc = scm_get_smob_descriptor(tag);
    const char *name = desc && desc->name ? desc->name : "smob";
    eval_error("scm-assert-smob-type: expected %s", name);
  }
}

// Helper function to extract int64_t from SCM number
static inline int64_t ptr_to_num(SCM *scm) {
  if (!is_num(scm)) {
    return 0;
  }
  return (int64_t)(intptr_t)scm->value;
}

// Scheme wrappers for smob functions
SCM *scm_c_make_smob_type(SCM *name, SCM *size) {
  if (!is_str(name)) {
    eval_error("scm-make-smob-type: expected string for name");
    return scm_none();
  }
  if (!is_num(size)) {
    eval_error("scm-make-smob-type: expected number for size");
    return scm_none();
  }
  
  SCM_String *name_str = cast<SCM_String>(name);
  int64_t size_val = ptr_to_num(size);
  
  // Create null-terminated string for name
  char *name_cstr = new char[name_str->len + 1];
  memcpy(name_cstr, name_str->data, name_str->len);
  name_cstr[name_str->len] = '\0';
  
  long tag = scm_make_smob_type(name_cstr, (size_t)size_val);
  
  SCM *result = new SCM();
  result->type = SCM::NUM;
  result->value = (void *)(intptr_t)tag;
  result->source_loc = nullptr;
  
  return result;
}

SCM *scm_c_make_smob(SCM *tag, SCM *data) {
  if (!is_num(tag)) {
    eval_error("scm-make-smob: expected number for tag");
    return scm_none();
  }
  
  long tag_val = (long)ptr_to_num(tag);
  void *data_ptr = nullptr;
  
  if (is_num(data)) {
    // For size=0 smobs, store value in data pointer
    data_ptr = (void *)(intptr_t)ptr_to_num(data);
  } else {
    // For size>0 smobs, data should be a pointer (not supported from Scheme yet)
    eval_error("scm-make-smob: data must be a number for size=0 smobs");
    return scm_none();
  }
  
  return scm_make_smob_with_data(tag_val, (int64_t)(intptr_t)data_ptr);
}

SCM *scm_c_smob_data(SCM *smob) {
  if (!is_smob(smob)) {
    eval_error("scm-smob-data: expected smob");
    return scm_none();
  }
  
  SCM_Smob *s = cast<SCM_Smob>(smob);
  SCM_SmobDescriptor *desc = scm_get_smob_descriptor(s->tag);
  
  if (desc && desc->size > 0) {
    // For size>0 smobs, return pointer (not supported from Scheme yet)
    eval_error("scm-smob-data: cannot get pointer data from Scheme");
    return scm_none();
  } else {
    // For size=0 smobs, return the stored value
    int64_t val = (int64_t)(intptr_t)s->data;
    SCM *result = new SCM();
    result->type = SCM::NUM;
    result->value = (void *)(intptr_t)val;
    result->source_loc = nullptr;
    return result;
  }
}

SCM *scm_c_smob_p(SCM *obj) {
  return is_smob(obj) ? scm_bool_true() : scm_bool_false();
}

// Initialization
void init_smob() {
  g_numsmob = 0;
  memset(g_smobs, 0, sizeof(g_smobs));
  
  // Register Scheme functions
  scm_define_function("scm-make-smob-type", 2, 0, 0, scm_c_make_smob_type);
  scm_define_function("scm-make-smob", 2, 0, 0, scm_c_make_smob);
  scm_define_function("scm-smob-data", 1, 0, 0, scm_c_smob_data);
  scm_define_function("scm-smob?", 1, 0, 0, scm_c_smob_p);
}
