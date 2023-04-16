//
// Created by PikachuHy on 2023/2/23.
//

#include "pscm/Number.h"
#include "pscm/Exception.h"
#include "pscm/common_def.h"
#include <sstream>
#include <string_view>
using namespace std::string_literals;

namespace pscm {
std::ostream& operator<<(std::ostream& out, const Number& num) {
  if (num.data_.index() == 1) {
    return out << std::get<1>(num.data_);
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const Complex& num) {
  out << num.real_part_;
  if (num.imag_part_ > 0) {
    out << "+";
  }
  out << num.imag_part_;
  out << "i";
  return out;
}

std::ostream& operator<<(std::ostream& out, const Rational& num) {
  out << num.numerator_;
  out << "/";
  out << num.denominator_;
  return out;
}

Number Number::operator-(const Number& num) {
  if (num.data_.index() != 1) {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
  }
  auto data = std::get<1>(data_) - std::get<1>(num.data_);
  return Number(data);
}

Number Number::operator/(const Number& num) {
  if (num.data_.index() != 1) {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
  }
  auto data = std::get<1>(data_) / std::get<1>(num.data_);
  return Number(data);
}

bool Number::operator<(const Number& num) const {
  PSCM_ASSERT(data_.index() == 1);
  if (num.data_.index() != 1) {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
  }
  auto a = std::get<1>(data_);
  auto b = std::get<1>(num.data_);
  return a < b;
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
  else {
    PSCM_THROW_EXCEPTION("Invalid number type: not supported now, " + to_string());
  }
}

std::string Number::to_string() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
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