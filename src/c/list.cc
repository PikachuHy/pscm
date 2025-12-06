#include "pscm.h"

SCM *scm_list1(SCM *arg) {
  auto l = make_list(arg);
  return wrap(l);
}

SCM *scm_list2(SCM *arg1, SCM *arg2) {
  auto l = make_list(arg1, arg2);
  return wrap(l);
}

SCM *scm_list3(SCM *arg1, SCM *arg2, SCM *arg3) {
  auto l = make_list(arg1, arg2, arg3);
  return wrap(l);
}

static void _concat(SCM_List *& it, SCM_List *l) {
  while (l) {
    it->next = make_list(l->data);
    it = it->next;
    l = l->next;
  }
}

SCM *scm_concat_list2(SCM *arg1, SCM *arg2) {
  assert(is_pair(arg1));
  assert(is_pair(arg2));
  SCM_List dummay_list;
  auto it = &dummay_list;
  it->next = NULL;
  _concat(it, cast<SCM_List>(arg1));
  _concat(it, cast<SCM_List>(arg2));
  if (dummay_list.next) {
    return wrap(dummay_list.next);
  }
  return scm_nil();
}

// Generic list function that takes variable number of arguments
static SCM *scm_list_impl(SCM_List *args) {
  if (!args) {
    return scm_nil();
  }
  SCM_List dummy;
  dummy.data = nullptr;
  dummy.next = nullptr;
  SCM_List *tail = &dummy;
  SCM_List *current = args;
  
  while (current) {
    SCM_List *node = make_list(current->data);
    tail->next = node;
    tail = node;
    current = current->next;
  }
  
  if (dummy.next) {
    return wrap(dummy.next);
  }
  return scm_nil();
}

SCM *scm_list(SCM_List *args) {
  return scm_list_impl(args);
}

// cons function: (cons a b) -> (a . b)
SCM *scm_cons(SCM *car_val, SCM *cdr_val) {
  SCM_List *l = make_list(car_val);
  if (is_pair(cdr_val)) {
    // If cdr is a list, append it
    l->next = cast<SCM_List>(cdr_val);
  } else if (is_nil(cdr_val)) {
    // If cdr is nil, just return the list with car
    // (already done)
  } else {
    // Otherwise, create a proper pair
    l->next = make_list(cdr_val);
  }
  return wrap(l);
}

// append function: (append list1 list2 ...) -> concatenated list
SCM *scm_append(SCM_List *args) {
  if (!args) {
    return scm_nil();
  }
  
  // If only one argument, return it
  if (!args->next) {
    return args->data;
  }
  
  // Build result by concatenating all lists
  SCM_List dummy;
  dummy.data = nullptr;
  dummy.next = nullptr;
  SCM_List *tail = &dummy;
  
  SCM_List *current = args;
  while (current) {
    if (is_pair(current->data)) {
      SCM_List *src = cast<SCM_List>(current->data);
      while (src) {
        SCM_List *node = make_list(src->data);
        tail->next = node;
        tail = node;
        src = src->next;
      }
    } else if (!is_nil(current->data)) {
      // Non-list, non-nil: just add it
      SCM_List *node = make_list(current->data);
      tail->next = node;
      tail = node;
    }
    current = current->next;
  }
  
  if (dummy.next) {
    return wrap(dummy.next);
  }
  return scm_nil();
}
