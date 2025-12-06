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
