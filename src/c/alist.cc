#include "pscm.h"
#include <type_traits>
bool _eq(SCM *lhs, SCM *rhs);

SCM *scm_c_assv(SCM *key, SCM *alist) {
  assert(key);
  assert(alist);
  assert(is_pair(alist));
  auto l = cast<SCM_List>(alist);
  auto it = l;
  while (it) {
    if (_eq(key, car(it->data))) {
      return it->data;
    }
    it = it->next;
  }
  return scm_bool_false();
}

void init_alist() {
  scm_define_function("assv", 2, 0, 0, scm_c_assv);
}