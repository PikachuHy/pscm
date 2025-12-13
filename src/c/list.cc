#include "pscm.h"
#include "eval.h"

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
  SCM_List dummy = make_list_dummy();
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
    // Check if the appended list is a dotted pair
    // If so, preserve the is_dotted flag
    SCM_List *last = l->next;
    while (last->next) {
      last = last->next;
    }
    // If the last node of cdr is dotted, keep it dotted
    // Otherwise, it's a proper list, so no change needed
  } else if (is_nil(cdr_val)) {
    // If cdr is nil, just return the list with car
    // (already done, l->next is nullptr)
  } else {
    // Otherwise, create a dotted pair: (car . cdr)
    l->next = make_list(cdr_val);
    l->next->is_dotted = true;  // Mark as dotted pair's cdr
  }
  return wrap(l);
}

// Helper function to copy a list element
static SCM_List *_copy_list_element(SCM *data) {
  return make_list(data);
}

// Helper function to check if a list is a dotted pair by checking is_dotted flag
static bool _is_dotted_pair(SCM_List *l) {
  if (!l || !l->next) {
    return false;
  }
  // Find the last node and check its is_dotted flag
  SCM_List *last = l;
  while (last->next) {
    last = last->next;
  }
  return last->is_dotted;
}

// append function: (append list1 list2 ...) -> concatenated list
// If the last argument is a dotted pair, the result will also be a dotted pair
SCM *scm_append(SCM_List *args) {
  if (!args) {
    return scm_nil();
  }
  
  // If only one argument, return it
  if (!args->next) {
    return args->data;
  }
  
  // Find the last argument to check if it's a dotted pair
  // We need to check the actual value, not the expression, so we check after
  // processing all arguments
  SCM_List *last_arg = args;
  while (last_arg->next) {
    last_arg = last_arg->next;
  }
  // Check if the last argument evaluates to a dotted pair
  // This is done after we've processed all arguments, so we can check the actual value
  bool last_is_dotted_pair = false;
  SCM *last_cdr = nullptr;
  if (is_pair(last_arg->data)) {
    SCM_List *last_list = cast<SCM_List>(last_arg->data);
    if (_is_dotted_pair(last_list)) {
      last_is_dotted_pair = true;
      // Find the last node to get the cdr
      SCM_List *last_node = last_list;
      while (last_node->next) {
        last_node = last_node->next;
      }
      last_cdr = last_node->data;
    }
  }
  
  // Build result by concatenating all lists
  SCM_List dummy = make_list_dummy();
  SCM_List *tail = &dummy;
  
  SCM_List *current = args;
  while (current) {
    if (is_pair(current->data)) {
      SCM_List *src = cast<SCM_List>(current->data);
      if (current == last_arg && last_is_dotted_pair) {
        // Last argument is a dotted pair: copy car, but preserve the cdr
        // For (c . d), src points to the pair, src->data is c, src->next->data is d
        // We copy c, but not d (which will be the final cdr)
        if (src) {
          SCM_List *node = _copy_list_element(src->data);
          tail->next = node;
          tail = node;
        }
        // The cdr (d) will be appended below, creating the dotted pair structure
      } else {
        // Regular list: copy all elements
        while (src) {
          SCM_List *node = _copy_list_element(src->data);
          tail->next = node;
          tail = node;
          src = src->next;
        }
      }
    } else if (!is_nil(current->data)) {
      // Non-list, non-nil: just add it
      SCM_List *node = _copy_list_element(current->data);
      tail->next = node;
      tail = node;
    }
    current = current->next;
  }
  
  // If last argument was a dotted pair, append its cdr as the final cdr
  // This creates a dotted pair structure: the last node's next points to a node with the cdr,
  // and that node's next is nullptr, and is_dotted = true
  if (last_is_dotted_pair && last_cdr) {
    // Create a node for the cdr, but don't update tail
    // This makes tail->next point to a node with data=cdr and next=nullptr
    // which is the structure for a dotted pair
    SCM_List *node = _copy_list_element(last_cdr);
    node->is_dotted = true;  // Mark as dotted pair's cdr
    tail->next = node;
    // tail remains pointing to the previous node, so tail->next->next == nullptr
    // This creates the dotted pair structure
  }
  
  if (dummy.next) {
    return wrap(dummy.next);
  }
  return scm_nil();
}

// list-head: Copy the first k elements from lst into a new list
SCM *scm_c_list_head(SCM *lst, SCM *k) {
  if (!is_pair(lst)) {
    eval_error("list-head: first argument must be a pair");
    return nullptr;
  }
  if (!is_num(k)) {
    eval_error("list-head: second argument must be a number");
    return nullptr;
  }
  
  size_t count = (size_t)(int64_t)k->value;
  SCM_List dummy = make_list_dummy();
  SCM_List *tail = &dummy;
  SCM_List *current = cast<SCM_List>(lst);
  
  while (count > 0 && current) {
    SCM_List *node = make_list(current->data);
    tail->next = node;
    tail = node;
    current = current->next;
    count--;
  }
  
  if (count > 0) {
    eval_error("list-head: list too short");
    return nullptr;
  }
  
  if (dummy.next) {
    return wrap(dummy.next);
  }
  return scm_nil();
}

// list-tail: Return the tail of lst beginning with its kth element (shared structure)
SCM *scm_c_list_tail(SCM *lst, SCM *k) {
  if (!is_pair(lst) && !is_nil(lst)) {
    eval_error("list-tail: first argument must be a pair or nil");
    return nullptr;
  }
  if (!is_num(k)) {
    eval_error("list-tail: second argument must be a number");
    return nullptr;
  }
  
  size_t count = (size_t)(int64_t)k->value;
  SCM_List *current = cast<SCM_List>(lst);
  
  while (count > 0) {
    if (!current || is_nil(wrap(current))) {
      eval_error("list-tail: list too short");
      return nullptr;
    }
    current = current->next;
    count--;
  }
  
  if (current) {
    return wrap(current);
  }
  return scm_nil();
}

// set-car!: Store value in the car field of pair
SCM *scm_c_set_car(SCM *pair, SCM *value) {
  if (!is_pair(pair)) {
    eval_error("set-car!: first argument must be a pair");
    return nullptr;
  }
  
  auto l = cast<SCM_List>(pair);
  l->data = value;
  
  return scm_none();
}

// set-cdr!: Store value in the cdr field of pair
SCM *scm_c_set_cdr(SCM *pair, SCM *value) {
  if (!is_pair(pair)) {
    eval_error("set-cdr!: first argument must be a pair");
    return nullptr;
  }
  
  auto l = cast<SCM_List>(pair);
  if (is_pair(value)) {
    l->next = cast<SCM_List>(value);
  } else if (is_nil(value)) {
    l->next = nullptr;
  } else {
    // Create a dotted pair: (car . value)
    l->next = make_list(value);
    l->next->is_dotted = true;
  }
  
  return scm_none();
}

// last-pair: Return the last pair in lst (using Floyd's cycle detection)
SCM *scm_c_last_pair(SCM *lst) {
  if (is_nil(lst)) {
    return scm_nil();
  }
  if (!is_pair(lst)) {
    eval_error("last-pair: argument must be a pair or nil");
    return nullptr;
  }
  
  SCM_List *tortoise = cast<SCM_List>(lst);
  SCM_List *hare = cast<SCM_List>(lst);
  
  // Floyd's cycle detection algorithm
  while (true) {
    // Move hare two steps
    if (!hare->next || is_nil(wrap(hare->next))) {
      // Reached end, return tortoise (last pair)
      return wrap(tortoise);
    }
    hare = hare->next;
    if (!hare->next || is_nil(wrap(hare->next))) {
      // Reached end, return hare (last pair)
      return wrap(hare);
    }
    hare = hare->next;
    
    // Move tortoise one step
    tortoise = tortoise->next;
    
    // Check for cycle
    if (hare == tortoise) {
      eval_error("last-pair: circular structure detected");
      return nullptr;
    }
  }
}

SCM *scm_c_reverse(SCM *lst) {
  if (is_nil(lst)) {
    return scm_nil();
  }
  if (!is_pair(lst)) {
    eval_error("reverse: argument must be a pair or nil");
    return nullptr;
  }
  
  SCM_List *result = nullptr;
  SCM_List *current = cast<SCM_List>(lst);
  
  while (current) {
    auto new_node = make_list(current->data);
    new_node->next = result;
    result = new_node;
    
    if (!current->next || is_nil(wrap(current->next))) {
      break;
    }
    current = current->next;
  }
  
  if (result) {
    return wrap(result);
  }
  return scm_nil();
}

SCM *scm_c_list_ref(SCM *lst, SCM *k) {
  if (is_nil(lst)) {
    eval_error("list-ref: index out of range");
    return nullptr;
  }
  if (!is_pair(lst)) {
    eval_error("list-ref: first argument must be a list");
    return nullptr;
  }
  if (!is_num(k)) {
    eval_error("list-ref: second argument must be an integer");
    return nullptr;
  }
  
  int64_t idx = (int64_t)k->value;
  if (idx < 0) {
    eval_error("list-ref: index must be non-negative");
    return nullptr;
  }
  
  SCM_List *current = cast<SCM_List>(lst);
  int64_t i = 0;
  
  while (current) {
    if (i == idx) {
      return current->data;
    }
    i++;
    if (!current->next || is_nil(wrap(current->next))) {
      break;
    }
    current = current->next;
  }
  
  eval_error("list-ref: index out of range");
  return nullptr;
}

SCM *scm_c_length(SCM *lst) {
  if (is_nil(lst)) {
    SCM *result = new SCM();
    result->type = SCM::NUM;
    result->value = (void *)0;
    result->source_loc = nullptr;
    return result;
  }
  if (!is_pair(lst)) {
    eval_error("length: argument must be a list");
    return nullptr;
  }
  
  int count = 0;
  SCM_List *current = cast<SCM_List>(lst);
  
  while (current) {
    count++;
    if (!current->next || is_nil(wrap(current->next))) {
      break;
    }
    current = current->next;
  }
  
  SCM *result = new SCM();
  result->type = SCM::NUM;
  result->value = (void *)(intptr_t)count;
  result->source_loc = nullptr;
  return result;
}

// memq: Return the first sublist of list whose car is obj (using eq? for comparison)
// Returns #f if obj is not found
SCM *scm_c_memq(SCM *obj, SCM *lst) {
  if (is_nil(lst)) {
    return scm_bool_false();
  }
  if (!is_pair(lst)) {
    eval_error("memq: second argument must be a list");
    return nullptr;
  }
  
  extern bool _eq(SCM *lhs, SCM *rhs);
  SCM_List *current = cast<SCM_List>(lst);
  
  while (current) {
    if (current->data && _eq(obj, current->data)) {
      // Found a match, return the sublist starting from this element
      return wrap(current);
    }
    if (!current->next || is_nil(wrap(current->next))) {
      break;
    }
    current = current->next;
  }
  
  // Not found, return #f
  return scm_bool_false();
}

void init_list() {
  scm_define_function("car", 1, 0, 0, car);
  scm_define_function("cdr", 1, 0, 0, cdr);
  scm_define_function("cadr", 1, 0, 0, cadr);
  scm_define_function("cddr", 1, 0, 0, cddr);
  scm_define_function("caddr", 1, 0, 0, caddr);
  scm_define_function("cons", 2, 0, 0, scm_cons);
  scm_define_vararg_function("list", scm_list);
  scm_define_vararg_function("append", scm_append);
  scm_define_function("list-head", 2, 0, 0, scm_c_list_head);
  scm_define_function("list-tail", 2, 0, 0, scm_c_list_tail);
  scm_define_function("list-ref", 2, 0, 0, scm_c_list_ref);
  scm_define_function("set-car!", 2, 0, 0, scm_c_set_car);
  scm_define_function("set-cdr!", 2, 0, 0, scm_c_set_cdr);
  scm_define_function("last-pair", 1, 0, 0, scm_c_last_pair);
  scm_define_function("reverse", 1, 0, 0, scm_c_reverse);
  scm_define_function("length", 1, 0, 0, scm_c_length);
  scm_define_function("memq", 2, 0, 0, scm_c_memq);
}
