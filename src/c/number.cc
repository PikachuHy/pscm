#include "pscm.h"
#include "eval.h"
#include <type_traits>

// Forward declaration
SCM *scm_make_ratio(int64_t numerator, int64_t denominator);

SCM *scm_c_is_negative(SCM *arg) {
  if (!is_num(arg) && !is_float(arg)) {
    eval_error("negative?: expected number");
    return nullptr;
  }
  double val = scm_to_double(arg);
  if (val < 0) {
    return scm_bool_true();
  }
  return scm_bool_false();
}

SCM *scm_c_is_zero(SCM *arg) {
  if (!is_num(arg) && !is_float(arg) && !is_ratio(arg)) {
    eval_error("zero?: expected number");
    return nullptr;
  }
  double val = scm_to_double(arg);
  if (val == 0.0) {
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
  // Handle ratio (rational number) multiplication
  if (is_ratio(lhs) || is_ratio(rhs)) {
    // Convert both to numerator/denominator form
    int64_t num1, den1, num2, den2;
    
    if (is_num(lhs)) {
      num1 = (int64_t)lhs->value;
      den1 = 1;
    } else if (is_ratio(lhs)) {
      SCM_Rational *rat = cast<SCM_Rational>(lhs);
      num1 = rat->numerator;
      den1 = rat->denominator;
    } else {
      // For floats, use float arithmetic
      double d_lhs = scm_to_double(lhs);
      double d_rhs = scm_to_double(rhs);
      return scm_from_double(d_lhs * d_rhs);
    }
    
    if (is_num(rhs)) {
      num2 = (int64_t)rhs->value;
      den2 = 1;
    } else if (is_ratio(rhs)) {
      SCM_Rational *rat = cast<SCM_Rational>(rhs);
      num2 = rat->numerator;
      den2 = rat->denominator;
    } else {
      // For floats, use float arithmetic
      double d_lhs = scm_to_double(lhs);
      double d_rhs = scm_to_double(rhs);
      return scm_from_double(d_lhs * d_rhs);
    }
    
    // Multiply: (a/b) * (c/d) = (a*c)/(b*d)
    return scm_make_ratio(num1 * num2, den1 * den2);
  }
  
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

// Helper function to compute GCD
static int64_t gcd(int64_t a, int64_t b) {
  a = a < 0 ? -a : a;
  b = b < 0 ? -b : b;
  while (b != 0) {
    int64_t temp = b;
    b = a % b;
    a = temp;
  }
  return a;
}

// Create a rational number (fraction)
SCM *scm_make_ratio(int64_t numerator, int64_t denominator) {
  if (denominator == 0) {
    eval_error("make-ratio: division by zero");
    return nullptr;
  }
  
  // Handle special cases
  if (numerator == 0) {
    return _create_num(0);
  }
  
  // Normalize sign: denominator should be positive
  if (denominator < 0) {
    numerator = -numerator;
    denominator = -denominator;
  }
  
  // Simplify the fraction using GCD
  int64_t g = gcd(numerator, denominator);
  numerator /= g;
  denominator /= g;
  
  // If denominator is 1, return an integer
  if (denominator == 1) {
    return _create_num(numerator);
  }
  
  // Create rational number
  SCM_Rational *rat = new SCM_Rational();
  rat->numerator = numerator;
  rat->denominator = denominator;
  
  SCM *data = new SCM();
  data->type = SCM::RATIO;
  data->value = rat;
  data->source_loc = nullptr;
  return data;
}

SCM *scm_c_expt(SCM *base, SCM *exp) {
  assert(is_num(base));
  assert(is_num(exp));
  int64_t n1 = (int64_t)base->value;
  int64_t n2 = (int64_t)exp->value;
  
  // Handle special cases
  if (n1 == 0 && n2 == 0) {
    // 0^0 = 1
    return _create_num(1);
  }
  else if (n1 == 0) {
    // 0^n = 0 (for n != 0)
    return _create_num(0);
  }
  else if (n2 == 0) {
    // n^0 = 1
    return _create_num(1);
  }
  
  // Handle negative exponent
  bool is_negative = false;
  if (n2 < 0) {
    is_negative = true;
    n2 = -n2;
  }
  
  // Calculate base^n2
  int64_t ret = n1;
  for (int64_t i = 0; i < n2 - 1; ++i) {
    ret *= n1;
  }
  
  // Handle negative exponent result
  if (is_negative) {
    // For negative exponents, we return 1/result
    // But since we only support integers, we can only return exact results
    // when result is 1 or -1
    if (ret == 1) {
      return _create_num(1);
    }
    else if (ret == -1) {
      return _create_num(-1);
    }
    else {
      // For other cases, we can't represent fractions, so return 0
      // This is a limitation of integer-only arithmetic
      return _create_num(0);
    }
  }
  
  return _create_num(ret);
}

SCM *scm_c_div(SCM_List *args) {
  if (!args || !args->next) {
    eval_error("/: requires at least one argument");
    return nullptr;
  }
  
  SCM *first = args->next->data;
  if (!is_num(first) && !is_float(first) && !is_ratio(first)) {
    eval_error("/: expected number");
    return nullptr;
  }
  
  // Convert first argument to numerator/denominator form
  int64_t numerator, denominator;
  if (is_num(first)) {
    numerator = (int64_t)first->value;
    denominator = 1;
  } else if (is_ratio(first)) {
      SCM_Rational *rat = cast<SCM_Rational>(first);
    numerator = rat->numerator;
    denominator = rat->denominator;
  } else {
    // For floats, we'll use rational approximation
    // But for now, let's handle it as integer division if possible
    double val = scm_to_double(first);
    if (val == (double)(int64_t)val) {
      numerator = (int64_t)val;
      denominator = 1;
    } else {
      // For non-integer floats, return float result
      double result = val;
      SCM_List *current = args->next->next;
      while (current) {
        SCM *arg = current->data;
        if (!is_num(arg) && !is_float(arg) && !is_ratio(arg)) {
          eval_error("/: expected number");
          return nullptr;
        }
        double divisor = scm_to_double(arg);
        if (divisor == 0.0) {
          eval_error("/: division by zero");
          return nullptr;
        }
        result /= divisor;
        current = current->next;
      }
      return scm_from_double(result);
    }
  }
  
  // If only one argument, return 1/arg
  if (!args->next->next) {
    if (numerator == 0) {
      eval_error("/: division by zero");
      return nullptr;
    }
    return scm_make_ratio(denominator, numerator);
  }
  
  // Process remaining arguments
  SCM_List *current = args->next->next;
  while (current) {
    SCM *arg = current->data;
    if (!is_num(arg) && !is_float(arg) && !is_ratio(arg)) {
      eval_error("/: expected number");
      return nullptr;
    }
    
    int64_t div_num, div_den;
    if (is_num(arg)) {
      div_num = (int64_t)arg->value;
      div_den = 1;
    } else if (is_ratio(arg)) {
      SCM_Rational *rat = cast<SCM_Rational>(arg);
      div_num = rat->numerator;
      div_den = rat->denominator;
    } else {
      // For floats, convert to rational if possible
      double val = scm_to_double(arg);
      if (val == (double)(int64_t)val) {
        div_num = (int64_t)val;
        div_den = 1;
      } else {
        // For non-integer floats, use float arithmetic
        double result = scm_to_double(first);
        SCM_List *iter = args->next->next;
        while (iter) {
          result /= scm_to_double(iter->data);
          iter = iter->next;
        }
        return scm_from_double(result);
      }
    }
    
    if (div_num == 0) {
      eval_error("/: division by zero");
      return nullptr;
    }
    
    // Multiply by reciprocal: (n/d) / (a/b) = (n/d) * (b/a) = (n*b)/(d*a)
    numerator = numerator * div_den;
    denominator = denominator * div_num;
    
    current = current->next;
  }
  
  return scm_make_ratio(numerator, denominator);
}

void init_number() {
  scm_define_function("negative?", 1, 0, 0, scm_c_is_negative);
  scm_define_function("zero?", 1, 0, 0, scm_c_is_zero);
  scm_define_generic_function("+", scm_c_add_number, _create_num(0));
  scm_define_function("=", 2, 0, 0, scm_c_eq_number);
  scm_define_vararg_function("-", scm_c_minus_wrapper);
  scm_define_generic_function("*", scm_c_mul_number, _create_num(1));
  scm_define_vararg_function("/", scm_c_div);
  scm_define_function("<=", 2, 0, 0, scm_c_lt_eq_number);
  scm_define_function(">=", 2, 0, 0, scm_c_gt_eq_number);
  scm_define_function("<", 2, 0, 0, scm_c_lt_number);
  scm_define_function(">", 2, 0, 0, scm_c_gt_number);
  scm_define_function("abs", 1, 0, 0, scm_c_abs);
  scm_define_function("expt", 2, 0, 0, scm_c_expt);
}