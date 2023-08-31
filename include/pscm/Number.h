//
// Created by PikachuHy on 2023/2/23.
//

#pragma once
#include "compat.h"
#include "pscm/misc/ICUCompat.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <variant>

namespace pscm {
class Complex {
public:
  Complex()
      : real_part_(0)
      , imag_part_(0) {
  }

  Complex(double real_part, double imag_part)
      : real_part_(real_part)
      , imag_part_(imag_part) {
  }

  bool operator==(const Complex& other) const {
    return real_part_ == other.real_part_ && imag_part_ == other.imag_part_;
  }

  [[nodiscard]] double imag_part() const {
    return imag_part_;
  }

  UString to_string() const;

private:
  double real_part_;
  double imag_part_;
};
class Number;

class Rational {
public:
  Rational()
      : numerator_(0)
      , denominator_(0) {
  }

  Rational(int64_t numerator, int64_t denominator)
      : numerator_(numerator)
      , denominator_(denominator) {
  }

  bool operator==(const Rational& other) const {
    return numerator_ == other.numerator_ && denominator_ == other.denominator_;
  }

  [[nodiscard]] std::int64_t to_int() const {
    return numerator_ / denominator_;
  }

  [[nodiscard]] double to_float() const {
    return 1.0 * numerator_ / denominator_;
  }

  UString to_string() const;

private:
  int64_t numerator_;
  int64_t denominator_;
  friend class Number;
};

class Number {
public:
  Number() {
  }

  Number(std::uint32_t val) {
    data_ = val;
  }

  Number(int32_t val) {
    data_ = val;
  }

  Number(int64_t val) {
    data_ = val;
  }

  Number(double val) {
    data_ = val;
  }

  Number(Rational val) {
    data_ = val;
  }

  Number(Complex val) {
    data_ = val;
  }

  bool is_int() const {
    return data_.index() == 1;
  }

  int64_t to_int() const;

  bool is_float() const {
    return data_.index() == 2;
  }

  double to_float() const;

  bool is_rational() const {
    return data_.index() == 3;
  }

  Rational to_rational() const;

  bool is_complex() const {
    return data_.index() == 4;
  }

  Complex to_complex() const;

  void inplace_add(const Number& num);
  void inplace_minus(const Number& num);
  void inplace_mul(const Number& num);
  void inplace_div(const Number& num);

  bool operator==(const Number& other) const {
    return this->data_ == other.data_;
  }

  void display() const;
  [[nodiscard]] UString to_string() const;
  friend std::ostream& operator<<(std::ostream& out, const Number& num);
  Number operator-(const Number& num);
  Number operator/(const Number& num);
  bool operator<(const Number& num) const;
  bool operator>(const Number& num) const;
  bool operator<=(const Number& num) const;
  bool operator>=(const Number& num) const;

  [[nodiscard]] bool is_zero() const;

private:
  std::variant<std::monostate, int64_t, double, Rational, Complex> data_;
};

Number operator""_num(const char *data, std::size_t len);
} // namespace pscm
