#pragma once

#include "pscm.h"

// Forward declarations
struct SCM_Smob;
struct SCM_SmobDescriptor;
struct scm_print_state;

// Smob descriptor structure
struct SCM_SmobDescriptor {
  const char *name;              // Type name
  size_t size;                   // Instance size (0 means direct value storage)
  SCM *(*mark)(SCM *);           // GC mark function (for future use)
  size_t (*free)(SCM *);         // GC free function (for future use)
  void (*print)(SCM *, SCM *, scm_print_state *);  // Print function (obj, port, pstate)
  SCM *(*equalp)(SCM *, SCM *);  // Equality comparison function
  // Applicable object functions (optional)
  SCM *(*apply_0)(SCM *);
  SCM *(*apply_1)(SCM *, SCM *);
  SCM *(*apply_2)(SCM *, SCM *, SCM *);
  SCM *(*apply_3)(SCM *, SCM *, SCM *, SCM *);
};

// Smob object structure
struct SCM_Smob {
  long tag;                      // Type tag (used to find descriptor)
  void *data;                    // Data pointer (when size > 0)
  int64_t data2;                 // Second data field (optional)
  int64_t data3;                 // Third data field (optional)
  int64_t flags;                 // Flags field (optional)
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
void scm_set_smob_print(long tag, void (*print)(SCM *, SCM *, scm_print_state *));
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

// Default print function (exported for use in print.cc)
void default_smob_print(SCM *smob, SCM *port, scm_print_state *pstate);

