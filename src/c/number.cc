#include "pscm.h"
#include <type_traits>

SCM *scm_c_is_negative(SCM *arg) {
  assert(is_num(arg));
  int64_t val = (int64_t)arg->value;
  if (val < 0) {
    return scm_bool_true();
  }
  return scm_bool_false();
}

template <typename Op>
struct BinaryOperator {
  static SCM *run(SCM *lhs, SCM *rhs) {
    assert(is_num(lhs));
    assert(is_num(rhs));
    int64_t n_lhs = (int64_t)lhs->value;
    int64_t n_rhs = (int64_t)rhs->value;
    auto ret = Op::run(n_lhs, n_rhs);
    if constexpr (std::is_same_v<decltype(ret), bool>) {
      if (ret) {
        return scm_bool_true();
      }
      else {
        return scm_bool_false();
      }
    }
    else {
      SCM *data = new SCM();
      data->type = SCM::NUM;
      data->value = (void *)ret;
      return data;
    }
  }
};

template <typename Ret, typename T>
struct AddOp {
  static Ret run(T lhs, T rhs) {
    return lhs + rhs;
  }
};

template <typename Ret, typename T>
struct MinusOp {
  static Ret run(T lhs, T rhs) {
    return lhs - rhs;
  }
};

template <typename Ret, typename T>
struct MulOp {
  static Ret run(T lhs, T rhs) {
    return lhs * rhs;
  }
};

template <typename Ret, typename T>
struct LtEqOp {
  static Ret run(T lhs, T rhs) {
    return lhs <= rhs;
  }
};

template <typename Ret, typename T>
struct GtEqOp {
  static Ret run(T lhs, T rhs) {
    return lhs >= rhs;
  }
};

template <typename Ret, typename T>
struct GtOp {
  static Ret run(T lhs, T rhs) {
    return lhs > rhs;
  }
};

template <typename Ret, typename T>
struct LtOp {
  static Ret run(T lhs, T rhs) {
    return lhs < rhs;
  }
};

template <typename Ret, typename T>
struct EqOp {
  static Ret run(T lhs, T rhs) {
    return lhs == rhs;
  }
};

SCM *scm_c_eq_number(SCM *lhs, SCM *rhs) {
  return BinaryOperator<EqOp<bool, int64_t>>::run(lhs, rhs);
}

SCM *scm_c_add_number(SCM *lhs, SCM *rhs) {
  return BinaryOperator<AddOp<int64_t, int64_t>>::run(lhs, rhs);
}

SCM *scm_c_minus_number(SCM *lhs, SCM *rhs) {
  return BinaryOperator<MinusOp<int64_t, int64_t>>::run(lhs, rhs);
}

SCM *scm_c_mul_number(SCM *lhs, SCM *rhs) {
  return BinaryOperator<MulOp<int64_t, int64_t>>::run(lhs, rhs);
}

SCM *scm_c_lt_eq_number(SCM *lhs, SCM *rhs) {
  return BinaryOperator<LtEqOp<bool, int64_t>>::run(lhs, rhs);
}

SCM *scm_c_gt_eq_number(SCM *lhs, SCM *rhs) {
  return BinaryOperator<GtEqOp<bool, int64_t>>::run(lhs, rhs);
}

SCM *scm_c_lt_number(SCM *lhs, SCM *rhs) {
  return BinaryOperator<LtOp<bool, int64_t>>::run(lhs, rhs);
}

SCM *scm_c_gt_number(SCM *lhs, SCM *rhs) {
  return BinaryOperator<GtOp<bool, int64_t>>::run(lhs, rhs);
}

SCM *scm_c_abs(SCM *arg) {
  assert(is_num(arg));
  int64_t val = (int64_t)arg->value;
  int64_t abs_val = val < 0 ? -val : val;
  SCM *data = new SCM();
  data->type = SCM::NUM;
  data->value = (void *)abs_val;
  return data;
}

bool _number_eq(SCM *lhs, SCM *rhs) {
  auto ret = BinaryOperator<EqOp<bool, int64_t>>::run(lhs, rhs);
  return is_true(ret);
}

SCM *_create_num(int64_t val) {
  SCM *data = new SCM();
  data->type = SCM::NUM;
  data->value = (void *)val;
  return data;
}

void init_number() {
  scm_define_function("negative?", 1, 0, 0, scm_c_is_negative);
  scm_define_generic_function("+", scm_c_add_number, _create_num(0));
  scm_define_function("=", 2, 0, 0, scm_c_eq_number);
  scm_define_function("-", 2, 0, 0, scm_c_minus_number);
  scm_define_function("*", 2, 0, 0, scm_c_mul_number);
  scm_define_function("<=", 2, 0, 0, scm_c_lt_eq_number);
  scm_define_function(">=", 2, 0, 0, scm_c_gt_eq_number);
  scm_define_function("<", 2, 0, 0, scm_c_lt_number);
  scm_define_function(">", 2, 0, 0, scm_c_gt_number);
  scm_define_function("abs", 1, 0, 0, scm_c_abs);
}