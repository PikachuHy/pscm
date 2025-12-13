#include "pscm.h"
#include "eval.h"

// Helper function for make-vector implementation
static SCM *scm_c_make_vector_impl(SCM *len_scm, SCM *fill_scm) {
  if (!is_num(len_scm)) {
    eval_error("make-vector: first argument must be an integer");
    return nullptr;
  }
  int64_t len = (int64_t)len_scm->value;
  if (len < 0) {
    eval_error("make-vector: length must be non-negative");
    return nullptr;
  }
  
  // Get optional fill value (default to unspecified/none)
  SCM *fill = fill_scm ? fill_scm : scm_none();
  
  // Create vector
  SCM_Vector *vec = new SCM_Vector();
  vec->length = (size_t)len;
  if (len > 0) {
    vec->elements = new SCM*[len];
    for (size_t i = 0; i < (size_t)len; i++) {
      vec->elements[i] = fill;
    }
  } else {
    vec->elements = nullptr;
  }
  
  return wrap(vec);
}

// Wrapper for variable arguments (make-vector k [fill])
SCM *scm_c_make_vector(SCM_List *args) {
  if (!args || !args->data) {
    eval_error("make-vector: requires at least 1 argument");
    return nullptr;
  }
  
  SCM *len_scm = args->data;
  SCM *fill_scm = (args->next && args->next->data) ? args->next->data : nullptr;
  return scm_c_make_vector_impl(len_scm, fill_scm);
}

// vector: Create a vector from arguments
SCM *scm_c_vector(SCM_List *args) {
  SCM_Vector *vec = new SCM_Vector();
  size_t count = 0;
  SCM_List *current = args;
  
  // Count arguments
  while (current) {
    count++;
    current = current->next;
  }
  
  vec->length = count;
  if (count > 0) {
    vec->elements = new SCM*[count];
    current = args;
    size_t i = 0;
    while (current) {
      vec->elements[i++] = current->data;
      current = current->next;
    }
  } else {
    vec->elements = nullptr;
  }
  
  return wrap(vec);
}

// vector-length: Return the length of a vector
SCM *scm_c_vector_length(SCM *vec) {
  if (!is_vector(vec)) {
    eval_error("vector-length: expected vector");
    return nullptr;
  }
  SCM_Vector *v = cast<SCM_Vector>(vec);
  SCM *scm = new SCM();
  scm->type = SCM::NUM;
  scm->value = (void*)(int64_t)v->length;
  scm->source_loc = nullptr;
  return scm;
}

// vector-ref: Return element at index k of vector
SCM *scm_c_vector_ref(SCM *vec, SCM *k) {
  if (!is_vector(vec)) {
    eval_error("vector-ref: expected vector");
    return nullptr;
  }
  if (!is_num(k)) {
    eval_error("vector-ref: expected integer index");
    return nullptr;
  }
  
  SCM_Vector *v = cast<SCM_Vector>(vec);
  int64_t idx = (int64_t)k->value;
  
  if (idx < 0 || (size_t)idx >= v->length) {
    eval_error("vector-ref: index out of range");
    return nullptr;
  }
  
  return v->elements[idx];
}

// vector-set!: Set element at index k of vector to obj
SCM *scm_c_vector_set(SCM *vec, SCM *k, SCM *obj) {
  if (!is_vector(vec)) {
    eval_error("vector-set!: expected vector");
    return nullptr;
  }
  if (!is_num(k)) {
    eval_error("vector-set!: expected integer index");
    return nullptr;
  }
  
  SCM_Vector *v = cast<SCM_Vector>(vec);
  int64_t idx = (int64_t)k->value;
  
  if (idx < 0 || (size_t)idx >= v->length) {
    eval_error("vector-set!: index out of range");
    return nullptr;
  }
  
  v->elements[idx] = obj;
  return scm_none();
}

// vector->list: Convert vector to list
SCM *scm_c_vector_to_list(SCM *vec) {
  if (!is_vector(vec)) {
    eval_error("vector->list: expected vector");
    return nullptr;
  }
  
  SCM_Vector *v = cast<SCM_Vector>(vec);
  
  if (v->length == 0) {
    return scm_nil();
  }
  
  // Build list from end to beginning
  SCM_List *result = nullptr;
  for (int i = (int)v->length - 1; i >= 0; i--) {
    SCM_List *cell = new SCM_List();
    cell->data = v->elements[i];
    cell->next = result;
    cell->is_dotted = false;
    result = cell;
  }
  
  return result ? wrap(result) : scm_nil();
}

// list->vector: Convert list to vector
SCM *scm_c_list_to_vector(SCM *list) {
  if (is_nil(list)) {
    // Empty list: return empty vector
    SCM_Vector *vec = new SCM_Vector();
    vec->length = 0;
    vec->elements = nullptr;
    return wrap(vec);
  }
  
  if (!is_pair(list)) {
    eval_error("list->vector: expected list");
    return nullptr;
  }
  
  // Count elements
  size_t count = 0;
  SCM_List *current = cast<SCM_List>(list);
  while (current) {
    count++;
    if (!current->next) {
      break;
    }
    current = current->next;
  }
  
  // Create vector
  SCM_Vector *vec = new SCM_Vector();
  vec->length = count;
  if (count > 0) {
    vec->elements = new SCM*[count];
    current = cast<SCM_List>(list);
    size_t i = 0;
    while (current && i < count) {
      vec->elements[i++] = current->data;
      current = current->next;
    }
  } else {
    vec->elements = nullptr;
  }
  
  return wrap(vec);
}

void init_vector() {
  scm_define_vararg_function("make-vector", scm_c_make_vector);
  scm_define_vararg_function("vector", scm_c_vector);
  scm_define_function("vector-length", 1, 0, 0, scm_c_vector_length);
  scm_define_function("vector-ref", 2, 0, 0, scm_c_vector_ref);
  scm_define_function("vector-set!", 3, 0, 0, scm_c_vector_set);
  scm_define_function("vector->list", 1, 0, 0, scm_c_vector_to_list);
  scm_define_function("list->vector", 1, 0, 0, scm_c_list_to_vector);
}
