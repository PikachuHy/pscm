//
// Created by PikachuHy on 2023/2/23.
//

#pragma once
#include <cstdlib>
#include <iostream>
#include <variant>

namespace pscm {
class Number {
public:
  Number(int32_t val) {
    data_ = val;
  }

  Number(int64_t val) {
    data_ = val;
  }

  Number(double val) {
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

  void inplace_add(const Number& num);
  void inplace_minus(const Number& num);
  void inplace_mul(const Number& num);
  void inplace_div(const Number& num);

  bool operator==(const Number& other) const {
    return this->data_ == other.data_;
  }

  void display() const;
  [[nodiscard]] std::string to_string() const;
  friend std::ostream& operator<<(std::ostream& out, const Number& num);
  Number operator-(const Number& num);
  Number operator/(const Number& num);
  bool operator<(const Number& num) const;
  bool operator>(const Number& num) const;
  bool operator<=(const Number& num) const;
  bool operator>=(const Number& num) const;

  [[nodiscard]] bool is_zero() const;

private:
  std::variant<std::monostate, int64_t, double> data_;
};

Number operator""_num(const char *data, std::size_t len);
} // namespace pscm
