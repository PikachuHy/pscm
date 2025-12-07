#include "pscm.h"
#include "eval.h"
#include <type_traits>

SCM *scm_c_is_negative(SCM *arg) {
  assert(is_num(arg));
  int64_t val = (int64_t)arg->value;
  if (val < 0) {
    return scm_bool_true();
  }
  return scm_bool_false();
}

// Type promotion: if either operand is float, promote both to float
static bool needs_float_promotion(SCM *lhs, SCM *rhs) {
  return is_float(lhs) || is_float(rhs);
}

template <typename Op>
struct BinaryOperator {
  static SCM *run(SCM *lhs, SCM *rhs) {
    // Check if we need float promotion
    if (needs_float_promotion(lhs, rhs)) {
      // Float operation: convert both to double and use double arithmetic
      double d_lhs = scm_to_double(lhs);
      double d_rhs = scm_to_double(rhs);
      // For floating point operations, we need to use double arithmetic
      // Create a double version of the operation by calling the template with double types
      // Since Op is templated, we can instantiate it with double
      using DoubleRet = decltype(Op::run(0.0, 0.0));
      auto ret = Op::run(d_lhs, d_rhs);
      if constexpr (std::is_same_v<DoubleRet, bool>) {
        return ret ? scm_bool_true() : scm_bool_false();
      } else {
        return scm_from_double(ret);
      }
    } else {
      // Integer operation
      assert(is_num(lhs));
      assert(is_num(rhs));
      int64_t n_lhs = (int64_t)lhs->value;
      int64_t n_rhs = (int64_t)rhs->value;
      auto ret = Op::run(n_lhs, n_rhs);
      if constexpr (std::is_same_v<decltype(ret), bool>) {
        return ret ? scm_bool_true() : scm_bool_false();
      } else {
        SCM *data = new SCM();
        data->type = SCM::NUM;
        data->value = (void *)ret;
        return data;
      }
    }
  }
};

template <typename Ret, typename T>
struct AddOp {
  static Ret run(T lhs, T rhs) {
    return lhs + rhs;
  }
};

// Specialization for double to ensure proper floating-point arithmetic
template <>
struct AddOp<double, double> {
  static double run(double lhs, double rhs) {
    return lhs + rhs;
  }
};

template <typename Ret, typename T>
struct MinusOp {
  static Ret run(T lhs, T rhs) {
    return lhs - rhs;
  }
};

template <>
struct MinusOp<double, double> {
  static double run(double lhs, double rhs) {
    return lhs - rhs;
  }
};

template <typename Ret, typename T>
struct MulOp {
  static Ret run(T lhs, T rhs) {
    return lhs * rhs;
  }
};

template <>
struct MulOp<double, double> {
  static double run(double lhs, double rhs) {
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
  // Use double comparison for mixed types
  if (needs_float_promotion(lhs, rhs)) {
    double d_lhs = scm_to_double(lhs);
    double d_rhs = scm_to_double(rhs);
    return (d_lhs == d_rhs) ? scm_bool_true() : scm_bool_false();
  }
  return BinaryOperator<EqOp<bool, int64_t>>::run(lhs, rhs);
}

SCM *scm_c_add_number(SCM *lhs, SCM *rhs) {
  // Handle float promotion directly
  if (needs_float_promotion(lhs, rhs)) {
    double d_lhs = scm_to_double(lhs);
    double d_rhs = scm_to_double(rhs);
    return scm_from_double(d_lhs + d_rhs);
  }
  return BinaryOperator<AddOp<int64_t, int64_t>>::run(lhs, rhs);
}

SCM *scm_c_minus_number(SCM *lhs, SCM *rhs) {
  // Handle float promotion directly
  if (needs_float_promotion(lhs, rhs)) {
    double d_lhs = scm_to_double(lhs);
    double d_rhs = scm_to_double(rhs);
    return scm_from_double(d_lhs - d_rhs);
  }
  return BinaryOperator<MinusOp<int64_t, int64_t>>::run(lhs, rhs);
}

// Unary minus: negate a number (supports both int and float)
SCM *scm_c_negate_number(SCM *arg) {
  if (is_float(arg)) {
    double val = ptr_to_double(arg->value);
    return scm_from_double(-val);
  } else if (is_num(arg)) {
    int64_t val = (int64_t)arg->value;
    SCM *data = new SCM();
    data->type = SCM::NUM;
    data->value = (void *)(-val);
    return data;
  }
  eval_error("negate: expected number");
  return nullptr;
}

// Wrapper for - that handles both unary and binary cases
SCM *scm_c_minus_wrapper(SCM_List *args) {
  // args is already the argument list (without function name)
  // For (- 1), args points to the list node containing 1
  // args can be nullptr if no arguments provided
  if (!args) {
    eval_error("-: requires at least one argument");
    return nullptr;
  }
  
  // Count arguments
  int count = 0;
  SCM_List *current = args;
  while (current) {
    count++;
    current = current->next;
  }
  
  if (count == 0) {
    eval_error("-: requires at least one argument");
    return nullptr;
  }
  
  if (count == 1) {
    // Unary: negate
    if (!args->data) {
      eval_error("-: invalid argument");
      return nullptr;
    }
    // Check if argument is a number
    if (!is_num(args->data) && !is_float(args->data)) {
      eval_error("-: expected number");
      return nullptr;
    }
    return scm_c_negate_number(args->data);
  } else {
    // Binary or more: subtract from first
    SCM *result = args->data;
    current = args->next;
    while (current) {
      result = scm_c_minus_number(result, current->data);
      current = current->next;
    }
    return result;
  }
}

SCM *scm_c_mul_number(SCM *lhs, SCM *rhs) {
  // Handle float promotion directly
  if (needs_float_promotion(lhs, rhs)) {
    double d_lhs = scm_to_double(lhs);
    double d_rhs = scm_to_double(rhs);
    return scm_from_double(d_lhs * d_rhs);
  }
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
  // BinaryOperator already handles float promotion
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
  scm_define_vararg_function("-", scm_c_minus_wrapper);
  scm_define_generic_function("*", scm_c_mul_number, _create_num(1));
  scm_define_function("<=", 2, 0, 0, scm_c_lt_eq_number);
  scm_define_function(">=", 2, 0, 0, scm_c_gt_eq_number);
  scm_define_function("<", 2, 0, 0, scm_c_lt_number);
  scm_define_function(">", 2, 0, 0, scm_c_gt_number);
  scm_define_function("abs", 1, 0, 0, scm_c_abs);
}