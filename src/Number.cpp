//
// Created by PikachuHy on 2023/2/23.
//
#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/Number.h"
#include "pscm/Exception.h"
#include "pscm/common_def.h"
#include "pscm/misc/ICUCompat.h"
#include <numeric>
#include <spdlog/fmt/fmt.h>
#include <sstream>
#include <string_view>
#endif
using namespace std::string_literals;

namespace pscm {
PSCM_INLINE_LOG_DECLARE("pscm.core.Number");

std::ostream& operator<<(std::ostream& out, const Number& num) {
  if (num.data_.index() == 1) {
    return out << std::get<1>(num.data_);
  }
  else if (num.data_.index() == 2) {
    return out << std::get<2>(num.data_);
  }
  else if (num.data_.index() == 3) {
    return out << std::get<3>(num.data_);
  }
  else if (num.data_.index() == 4) {
    return out << std::get<4>(num.data_);
  }
  else {
    PSCM_THROW_EXCEPTION("invalid number index: " + pscm::to_string(num.data_.index()) + ", update needed");
  }
  return out;
}

UString Complex::to_string() const{
  UString out;
  out += pscm::to_string(real_part_);
  if (imag_part_ > 0) {
    out += "+";
  }
  out += pscm::to_string(imag_part_);
  out += "i";
  return out;
}

UString Rational::to_string() const{
  UString out;
  out += pscm::to_string(numerator_);
  out += "/";
  out += pscm::to_string(denominator_);
  return out;
}

Number Number::operator-(const Number& num) {
  if (data_.index() == 1) {
    if (num.data_.index() == 1) {
      auto data = std::get<1>(data_) - std::get<1>(num.data_);
      return Number(data);
    }
    else if (num.data_.index() == 2) {
      auto data = std::get<1>(data_) - std::get<2>(num.data_);
      return Number(data);
    }
  }
  else if (data_.index() == 2) {
    if (num.data_.index() == 1) {
      auto data = std::get<2>(data_) - std::get<1>(num.data_);
      return Number(data);
    }
    else if (num.data_.index() == 2) {
      auto data = std::get<2>(data_) - std::get<2>(num.data_);
      return Number(data);
    }
  }
  else {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
  }
  PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
}

Number Number::operator/(const Number& num) {
  if (num.data_.index() != 1) {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
  }
  auto n1 = std::get<1>(data_);
  auto n2 = std::get<1>(num.data_);
  if (n2 == 0) {
    PSCM_THROW_EXCEPTION("bad expresion: " + pscm::to_string(n1) + "/" + pscm::to_string(n2));
  }
  auto data = n1 / n2;
  if (data * n2 == n1) {
    return Number(data);
  }
  else {
    // use rational
    auto m = std::gcd(n1, n2);
    return Number(Rational(n1 / m, n2 / m));
  }
}

bool Number::operator<(const Number& num) const {
  if (data_.index() == 1) {
    if (num.data_.index() == 1) {
      auto a = std::get<1>(data_);
      auto b = std::get<1>(num.data_);
      return a < b;
    }
    else if (num.data_.index() == 2) {
      auto a = std::get<1>(data_);
      auto b = std::get<2>(num.data_);
      return a < b;
    }
    else {
      PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + num.to_string());
    }
  }
  else if (data_.index() == 2) {
    if (num.data_.index() == 1) {
      auto a = std::get<2>(data_);
      auto b = std::get<1>(num.data_);
      return a < b;
    }
    else if (num.data_.index() == 2) {
      auto a = std::get<2>(data_);
      auto b = std::get<2>(num.data_);
      return a < b;
    }
    else {
      PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + num.to_string());
    }
  }
  else {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + num.to_string());
  }
}

bool Number::operator>(const Number& num) const {
  PSCM_ASSERT(data_.index() == 1);
  if (num.data_.index() != 1) {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
  }
  auto a = std::get<1>(data_);
  auto b = std::get<1>(num.data_);
  return a > b;
}

bool Number::operator<=(const Number& num) const {
  PSCM_ASSERT(data_.index() == 1);
  if (num.data_.index() != 1) {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
  }
  auto a = std::get<1>(data_);
  auto b = std::get<1>(num.data_);
  return a <= b;
}

bool Number::operator>=(const Number& num) const {

  PSCM_ASSERT(data_.index() == 1);
  if (num.data_.index() != 1) {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
  }
  auto a = std::get<1>(data_);
  auto b = std::get<1>(num.data_);
  return a >= b;
}

void Number::inplace_add(const Number& num) {
  if (is_int()) {
    if (num.is_int()) {
      data_ = to_int() + num.to_int();
    }
    else if (num.is_float()) {
      data_ = to_int() + num.to_float();
    }
    else {
      PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
    }
  }
  else if (is_float()) {
    if (num.is_int()) {
      data_ = to_float() + num.to_int();
    }
    else if (num.is_float()) {
      data_ = to_float() + num.to_float();
    }
    else {
      PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
    }
  }
  else {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
  }
}

void Number::inplace_minus(const Number& num) {
  if (num.data_.index() != 1) {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
  }
  data_ = std::get<1>(data_) - std::get<1>(num.data_);
}

void Number::inplace_mul(const Number& num) {
  if (is_int()) {
    if (num.is_int()) {
      data_ = to_int() * num.to_int();
    }
    else if (num.is_float()) {
      data_ = to_int() * num.to_float();
    }
    else if (num.is_rational()) {
      auto n0 = to_int();
      auto n1 = n0 * num.to_rational().numerator_;
      auto n2 = num.to_rational().denominator_;
      auto m = std::gcd(n1, n2);
      n1 /= m;
      n2 /= m;
      if (n2 == 1) {
        data_ = n1;
      }
      else {
        data_ = Rational(n1, n2);
      }
    }
    else {
      PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
    }
  }
  else if (is_float()) {
    if (num.is_int()) {
      data_ = to_float() * num.to_int();
    }
    else if (num.is_float()) {
      data_ = to_float() * num.to_float();
    }
    else {
      PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
    }
  }
  else if (is_rational()) {
    if (num.is_int()) {
      auto n0 = num.to_int();
      auto n1 = n0 * to_rational().numerator_;
      auto n2 = to_rational().denominator_;
      auto m = std::gcd(n1, n2);
      n1 /= m;
      n2 /= m;
      if (n2 == 1) {
        data_ = n1;
      }
      else {
        data_ = Rational(n1, n2);
      }
    }
    else if (num.is_float()) {
      auto n0 = num.to_float();
      auto n1 = n0 * to_rational().numerator_;
      auto n2 = to_rational().denominator_;
      data_ = n1 / n2;
    }
    else {
      PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
    }
  }
  else {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
  }
}

void Number::inplace_div(const Number& num) {
  if (num.data_.index() != 1) {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
  }
  data_ = std::get<1>(data_) / std::get<1>(num.data_);
}

int64_t Number::to_int() const {
  PSCM_ASSERT(data_.index() == 1);
  return std::get<1>(data_);
}

double Number::to_float() const {
  PSCM_ASSERT(data_.index() == 2);
  return std::get<2>(data_);
}

Rational Number::to_rational() const {
  PSCM_ASSERT(data_.index() == 3);
  return std::get<3>(data_);
}

Complex Number::to_complex() const {
  PSCM_ASSERT(data_.index() == 4);
  return std::get<4>(data_);
}

void Number::display() const {
  PSCM_ASSERT(data_.index() != 0);
  if (data_.index() == 1) {
    std::cout << std::get<1>(data_);
  }
  else if (data_.index() == 2) {
    std::cout << std::get<2>(data_);
  }
  else if (data_.index() == 3) {
    std::cout << std::get<3>(data_);
  }
  else {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
  }
}

UString Number::to_string() const{
  PSCM_ASSERT(!std::holds_alternative<std::monostate>(data_));
  if (std::holds_alternative<int64_t>(data_)) {
    return pscm::to_string(std::get<int64_t>(data_));
  }
  else if (std::holds_alternative<double>(data_)) {
    return pscm::to_string(std::get<double>(data_));
  }
  else if (std::holds_alternative<Rational>(data_)) {
    return std::get<Rational>(data_).to_string();
  }
  else if (std::holds_alternative<Complex>(data_)) {
    return std::get<Complex>(data_).to_string();
  }
  else {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
  }

}

bool Number::is_zero() const {
  if (data_.index() == 1) {
    return std::get<1>(data_) == 0;
  }
  if (data_.index() == 2) {
    return std::get<2>(data_) == 0;
  }
  PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
}
} // namespace pscm