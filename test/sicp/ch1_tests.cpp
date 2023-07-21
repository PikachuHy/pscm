//
// Created by PikachuHy on 2023/2/23.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#ifdef PSCM_USE_CXX20_MODULES
import pscm;
#else
#include <pscm/Number.h>
#include <pscm/Scheme.h>
#include <string>
#endif
using namespace doctest;
using namespace pscm;
using namespace std::string_literals;

TEST_CASE("testing page 7") {
  Scheme scm;
  auto ret = scm.eval("486");
  CHECK(ret == "486"_num);
}

TEST_CASE("testing page 7, with register machine") {
  Scheme scm(true);
  auto ret = scm.eval("486");
  CHECK(ret == "486"_num);
}

TEST_CASE("testing page 8, +") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(+ 137 349)");
  CHECK(ret == "486"_num);
}

TEST_CASE("testing page 8, +, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(+ 137 349)");
  CHECK(ret == "486"_num);
}

TEST_CASE("testing page 8, -") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(- 1000 334)");
  CHECK(ret == "666"_num);
}

TEST_CASE("testing page 8, -, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(- 1000 334)");
  CHECK(ret == "666"_num);
}

TEST_CASE("testing page 8, *") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(* 5 99)");
  CHECK(ret == "495"_num);
}

TEST_CASE("testing page 8, *, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(* 5 99)");
  CHECK(ret == "495"_num);
}

TEST_CASE("testing page 8, /") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(/ 10 5)");
  CHECK(ret == "2"_num);
}

TEST_CASE("testing page 8, /, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(/ 10 5)");
  CHECK(ret == "2"_num);
}

TEST_CASE("testing page 8, float-point add") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(+ 2.7 10)");
  CHECK(ret == "12.7"_num);
}

TEST_CASE("testing page 8, float-point add, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(+ 2.7 10)");
  CHECK(ret == "12.7"_num);
}

TEST_CASE("testing page 9, add more") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(+ 21 35 12 7)");
  CHECK(ret == "75"_num);
}

TEST_CASE("testing page 9, add more, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(+ 21 35 12 7)");
  CHECK(ret == "75"_num);
}

TEST_CASE("testing page 9, multiply more") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(* 25 4 12)");
  CHECK(ret == "1200"_num);
}

TEST_CASE("testing page 9, multiply more, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(* 25 4 12)");
  CHECK(ret == "1200"_num);
}

TEST_CASE("testing page 9, mix + * -") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(+ (* 3 5) (- 10 6))");
  CHECK(ret == "19"_num);
}

TEST_CASE("testing page 9, mix + * -, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(+ (* 3 5) (- 10 6))");
  CHECK(ret == "19"_num);
}

TEST_CASE("testing page 9, mix + * -+ * + + -") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(+ (* 3 (+ (* 2 4) (+ 3 5))) (+ (- 10 7) 6))");
  CHECK(ret == "57"_num);
}

TEST_CASE("testing page 9, mix + * -+ * + + -, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(+ (* 3 (+ (* 2 4) (+ 3 5))) (+ (- 10 7) 6))");
  CHECK(ret == "57"_num);
}

TEST_CASE("testing page 10, define") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(define size 2)");
  REQUIRE(ret.is_none());
  ret = scm.eval("size");
  CHECK(ret == "2"_num);
  ret = scm.eval("(* 5 size)");
  CHECK(ret == "10"_num);
}

TEST_CASE("testing page 10, define, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(define size 2)");
  REQUIRE_MESSAGE(ret.is_none(), "ret is "s, ret.to_string());
  ret = scm.eval("size");
  CHECK(ret == "2"_num);
  ret = scm.eval("(* 5 size)");
  CHECK(ret == "10"_num);
}

TEST_CASE("testing page 11, define") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(define pi 3.14159)");
  REQUIRE(ret.is_none());
  ret = scm.eval("(define radius 10)");
  REQUIRE(ret.is_none());
  ret = scm.eval("(* pi (* radius radius))");
  REQUIRE(ret == "314.159"_num);
  ret = scm.eval("(define circumference (* 2 pi radius))");
  REQUIRE(ret.is_none());
  ret = scm.eval("circumference");
  CHECK(ret == "62.8318"_num);
}

TEST_CASE("testing page 11, define, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(define pi 3.14159)");
  REQUIRE(ret.is_none());
  ret = scm.eval("(define radius 10)");
  REQUIRE(ret.is_none());
  ret = scm.eval("(* pi (* radius radius))");
  REQUIRE(ret == "314.159"_num);
  ret = scm.eval("(define circumference (* 2 pi radius))");
  REQUIRE(ret.is_none());
  ret = scm.eval("circumference");
  CHECK(ret == "62.8318"_num);
}

TEST_CASE("testing page 16, define procedure") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(define (square x) (* x x))");
  REQUIRE(ret.is_none());
  ret = scm.eval("square");
  CHECK(ret.is_proc());
  ret = scm.eval("(square 21)");
  CHECK(ret == "441"_num);
  ret = scm.eval("(square (+ 2 5))");
  CHECK(ret == "49"_num);
  ret = scm.eval("(square (square 3))");
  CHECK(ret == "81"_num);
  ret = scm.eval(R"(
(define (sum-of-squares x y)
  (+ (square x) (square y))))
)");
  REQUIRE(ret.is_none());
  ret = scm.eval("sum-of-squares");
  CHECK(ret.is_proc());
  ret = scm.eval("(sum-of-squares 3 4)");
  CHECK(ret == "25"_num);
  ret = scm.eval(R"(
(define (f a)
  (sum-of-squares (+ a 1) (* a 2)))
)");
  REQUIRE(ret.is_none());
  ret = scm.eval("(f 5)");
  CHECK(ret == "136"_num);
}

TEST_CASE("testing page 16, define procedure, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(define (square x) (* x x))");
  REQUIRE_MESSAGE(ret.is_none(), ret.to_string());
  ret = scm.eval("square");
  CHECK(ret.is_proc());
  ret = scm.eval("(square 21)");
  CHECK(ret == "441"_num);
  ret = scm.eval("(square (+ 2 5))");
  CHECK(ret == "49"_num);
  ret = scm.eval("(square (square 3))");
  CHECK(ret == "81"_num);
  ret = scm.eval(R"(
(define (sum-of-squares x y)
  (+ (square x) (square y))))
)");
  REQUIRE(ret.is_none());
  ret = scm.eval("sum-of-squares");
  CHECK(ret.is_proc());
  ret = scm.eval("(sum-of-squares 3 4)");
  CHECK(ret == "25"_num);
  ret = scm.eval(R"(
(define (f a)
  (sum-of-squares (+ a 1) (* a 2)))
)");
  REQUIRE(ret.is_none());
  ret = scm.eval("(f 5)");
  CHECK(ret == "136"_num);
}

TEST_CASE("testing page 22, cond") {
  Scheme scm;
  Cell ret;
  ret = scm.eval(R"(
(define (abs x)
  (cond ((> x 0) x)
        ((= x 0) 0)
        ((< x 0) (- x))))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(abs -1)");
  CHECK(ret == "1"_num);
  ret = scm.eval("(abs 0)");
  CHECK(ret == "0"_num);
  ret = scm.eval("(abs 1)");
  CHECK(ret == "1"_num);
}

TEST_CASE("testing page 22, cond, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval(R"(
(define (abs x)
  (cond ((> x 0) x)
        ((= x 0) 0)
        ((< x 0) (- x))))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(abs -1)");
  CHECK(ret == "1"_num);
  ret = scm.eval("(abs 0)");
  CHECK(ret == "0"_num);
  ret = scm.eval("(abs 1)");
  CHECK(ret == "1"_num);
}

TEST_CASE("testing page 24, cond") {
  Scheme scm;
  Cell ret;
  ret = scm.eval(R"(
(define (abs x)
  (cond ((< x 0) (- x))
        (else x)))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(abs -1)");
  CHECK(ret == "1"_num);
  ret = scm.eval("(abs 0)");
  CHECK(ret == "0"_num);
  ret = scm.eval("(abs 1)");
  CHECK(ret == "1"_num);
}

TEST_CASE("testing page 24, cond, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval(R"(
(define (abs x)
  (cond ((< x 0) (- x))
        (else x)))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(abs -1)");
  CHECK(ret == "1"_num);
  ret = scm.eval("(abs 0)");
  CHECK(ret == "0"_num);
  ret = scm.eval("(abs 1)");
  CHECK(ret == "1"_num);
}

TEST_CASE("testing page 24, if") {
  Scheme scm;
  Cell ret;
  ret = scm.eval(R"(
(define (abs x)
  (if (< x 0)
      (- x)
      x))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(abs -1)");
  CHECK(ret == "1"_num);
}

TEST_CASE("testing page 24, if, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval(R"(
(define (abs x)
  (if (< x 0)
      (- x)
      x))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(abs -1)");
  CHECK(ret == "1"_num);
}

TEST_CASE("testing page 25, and") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(define x 1)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(and (> x 5) (< x 10))");
  CHECK(ret == Cell::bool_false());
  ret = scm.eval("(define x 5)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(and (> x 5) (< x 10))");
  CHECK(ret == Cell::bool_false());
  ret = scm.eval("(define x 7)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(and (> x 5) (< x 10))");
  CHECK(ret == Cell::bool_true());
  ret = scm.eval("(define x 10)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(and (> x 5) (< x 10))");
  CHECK(ret == Cell::bool_false());
  ret = scm.eval("(define x 11)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(and (> x 5) (< x 10))");
  CHECK(ret == Cell::bool_false());
}

TEST_CASE("testing page 25, and, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(define x 1)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(and (> x 5) (< x 10))");
  CHECK(ret == Cell::bool_false());
  ret = scm.eval("(define x 5)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(and (> x 5) (< x 10))");
  CHECK(ret == Cell::bool_false());
  ret = scm.eval("(define x 7)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(and (> x 5) (< x 10))");
  CHECK(ret == Cell::bool_true());
  ret = scm.eval("(define x 10)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(and (> x 5) (< x 10))");
  CHECK(ret == Cell::bool_false());
  ret = scm.eval("(define x 11)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(and (> x 5) (< x 10))");
  CHECK(ret == Cell::bool_false());
}

TEST_CASE("testing page 25, or") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(define (>= x y) (or (> x y) (= x y)))");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(>= 2 2)");
  CHECK(ret == Cell::bool_true());
  ret = scm.eval("(>= 3 2)");
  CHECK(ret == Cell::bool_true());
  ret = scm.eval("(>= 1 2)");
  CHECK(ret == Cell::bool_false());
}

TEST_CASE("testing page 25, or, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(define (>= x y) (or (> x y) (= x y)))");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(>= 2 2)");
  CHECK(ret == Cell::bool_true());
  ret = scm.eval("(>= 3 2)");
  CHECK(ret == Cell::bool_true());
  ret = scm.eval("(>= 1 2)");
  CHECK(ret == Cell::bool_false());
}

TEST_CASE("testing page 26, not") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(define (>= x y) (not (< x y)))");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(>= 2 2)");
  CHECK(ret == Cell::bool_true());
  ret = scm.eval("(>= 3 2)");
  CHECK(ret == Cell::bool_true());
  ret = scm.eval("(>= 1 2)");
  CHECK(ret == Cell::bool_false());
}

TEST_CASE("testing page 26, not, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(define (>= x y) (or (> x y) (= x y)))");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(>= 2 2)");
  CHECK(ret == Cell::bool_true());
  ret = scm.eval("(>= 3 2)");
  CHECK(ret == Cell::bool_true());
  ret = scm.eval("(>= 1 2)");
  CHECK(ret == Cell::bool_false());
}

TEST_CASE("testing page 26, Exercise 1.1") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("10");
  CHECK(ret == "10"_num);
  ret = scm.eval("(+ 5 3 4)");
  CHECK(ret == "12"_num);
  ret = scm.eval("(- 9 1)");
  CHECK(ret == "8"_num);
  ret = scm.eval("(- 6 2)");
  CHECK(ret == "4"_num);
  ret = scm.eval("(+ (* 2 4) (- 4 6))");
  CHECK(ret == "6"_num);
  ret = scm.eval("(define a 3)"); // -> 3
  CHECK(ret == Cell::none());
  ret = scm.eval("(define b (+ a 1))"); // -> 4
  CHECK(ret == Cell::none());
  ret = scm.eval("(+ a b (* a b))");
  CHECK(ret == "19"_num);
  ret = scm.eval("(= a b)");
  CHECK(ret == Cell::bool_false());
  ret = scm.eval(R"(
(if (and (> b a) (< b (* a b)))
   b
   a)
)");
  CHECK(ret == "4"_num);

  ret = scm.eval(R"(
(cond ((= a 4) 6)
      ((= b 4) (+ 6 7 a))
      (else 25))
)");
  CHECK(ret == "16"_num);

  ret = scm.eval("(+ 2 (if (> b a) b a))");
  CHECK(ret == "6"_num);

  ret = scm.eval(R"(
(* (cond ((> a b) a)
         ((< a b) b)
         (else -1))
   (+ a 1))
)");
  CHECK(ret == "16"_num);
}

TEST_CASE("testing page 26, Exercise 1.1, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("10");
  CHECK(ret == "10"_num);
  ret = scm.eval("(+ 5 3 4)");
  CHECK(ret == "12"_num);
  ret = scm.eval("(- 9 1)");
  CHECK(ret == "8"_num);
  ret = scm.eval("(- 6 2)");
  CHECK(ret == "4"_num);
  ret = scm.eval("(+ (* 2 4) (- 4 6))");
  CHECK(ret == "6"_num);
  ret = scm.eval("(define a 3)"); // -> 3
  CHECK(ret == Cell::none());
  ret = scm.eval("(define b (+ a 1))"); // -> 4
  CHECK(ret == Cell::none());
  ret = scm.eval("(+ a b (* a b))");
  CHECK(ret == "19"_num);
  ret = scm.eval("(= a b)");
  CHECK(ret == Cell::bool_false());
  ret = scm.eval(R"(
(if (and (> b a) (< b (* a b)))
   b
   a)
)");
  CHECK(ret == "4"_num);

  ret = scm.eval(R"(
(cond ((= a 4) 6)
      ((= b 4) (+ 6 7 a))
      (else 25))
)");
  CHECK(ret == "16"_num);

  ret = scm.eval("(+ 2 (if (> b a) b a))");
  CHECK(ret == "6"_num);

  ret = scm.eval(R"(
(* (cond ((> a b) a)
         ((< a b) b)
         (else -1))
   (+ a 1))
)");
  CHECK(ret == "16"_num);
}

TEST_CASE("testing page 41") {
  Scheme scm;
  Cell ret;
  ret = scm.eval(R"(
(define (factorial n)
  (if (= n 1)
      1
      (* n (factorial (- n 1)))))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(factorial 6)");
  CHECK(ret == "720"_num);
}

TEST_CASE("testing page 41, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval(R"(
(define (factorial n)
  (if (= n 1)
      1
      (* n (factorial (- n 1)))))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(factorial 6)");
  CHECK(ret == "720"_num);
}

TEST_CASE("testing page 43") {
  Scheme scm;
  Cell ret;
  ret = scm.eval(R"(
(define (factorial n)
  (fact-iter 1 1 n))
)");
  ret = scm.eval(R"(
(define (fact-iter product counter max-count)
  (if (> counter max-count)
      product
      (fact-iter (* counter product)
                 (+ counter 1)
                 max-count)))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(factorial 6)");
  CHECK(ret == "720"_num);
}

TEST_CASE("testing page 43, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval(R"(
(define (factorial n)
  (fact-iter 1 1 n))
)");
  ret = scm.eval(R"(
(define (fact-iter product counter max-count)
  (if (> counter max-count)
      product
      (fact-iter (* counter product)
                 (+ counter 1)
                 max-count)))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(factorial 6)");
  CHECK(ret == "720"_num);
}

TEST_CASE("testing page 43, nested proc") {
  Scheme scm;
  Cell ret;
  ret = scm.eval(R"(
(define (factorial n)
  (define (iter product counter)
    (if (> counter n)
        product
        (iter (* counter product)
              (+ counter 1))))
  (iter 1 1))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(factorial 6)");
  CHECK(ret == "720"_num);
}

TEST_CASE("testing page 43, nested proc, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval(R"(
(define (factorial n)
  (define (iter product counter)
    (if (> counter n)
        product
        (iter (* counter product)
              (+ counter 1))))
  (iter 1 1))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(factorial 6)");
  CHECK(ret == "720"_num);
}

TEST_CASE("testing page 48") {
  Scheme scm;
  Cell ret;
  ret = scm.eval(R"(
(define (fib n)
  (cond ((= n 0) 0)
        ((= n 1) 1)
        (else (+ (fib (- n 1))
                 (fib (- n 2))))))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(fib 5)");
  CHECK(ret == "5"_num);
  ret = scm.eval("(fib 6)");
  CHECK(ret == "8"_num);
  ret = scm.eval("(fib 7)");
  CHECK(ret == "13"_num);
  ret = scm.eval("(fib 8)");
  CHECK(ret == "21"_num);
}

TEST_CASE("testing page 48, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval(R"(
(define (fib n)
  (cond ((= n 0) 0)
        ((= n 1) 1)
        (else (+ (fib (- n 1))
                 (fib (- n 2))))))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(fib 5)");
  CHECK(ret == "5"_num);
  ret = scm.eval("(fib 6)");
  CHECK(ret == "8"_num);
  ret = scm.eval("(fib 7)");
  CHECK(ret == "13"_num);
  ret = scm.eval("(fib 8)");
  CHECK(ret == "21"_num);
}

TEST_CASE("testing page 50") {
  Scheme scm;
  Cell ret;
  ret = scm.eval(R"(
(define (fib n)
  (fib-iter 1 0 n))
)");
  ret = scm.eval(R"(
(define (fib-iter a b count)
  (if (= count 0)
      b
      (fib-iter (+ a b) a (- count 1))))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(fib 5)");
  CHECK(ret == "5"_num);
  ret = scm.eval("(fib 6)");
  CHECK(ret == "8"_num);
  ret = scm.eval("(fib 7)");
  CHECK(ret == "13"_num);
  ret = scm.eval("(fib 8)");
  CHECK(ret == "21"_num);
}

TEST_CASE("testing page 50, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval(R"(
(define (fib n)
  (fib-iter 1 0 n))
)");
  ret = scm.eval(R"(
(define (fib-iter a b count)
  (if (= count 0)
      b
      (fib-iter (+ a b) a (- count 1))))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval("(fib 5)");
  CHECK(ret == "5"_num);
  ret = scm.eval("(fib 6)");
  CHECK(ret == "8"_num);
  ret = scm.eval("(fib 7)");
  CHECK(ret == "13"_num);
  ret = scm.eval("(fib 8)");
  CHECK(ret == "21"_num);
}
