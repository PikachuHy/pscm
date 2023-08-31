//
// Created by PikachuHy on 2023/3/12.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#ifdef PSCM_USE_CXX20_MODULES
import pscm;
#else
#include <pscm/Displayable.h>
#include <pscm/Number.h>
#include <pscm/Pair.h>
#include <pscm/Scheme.h>
#include <pscm/Symbol.h>
#include <pscm/scm_utils.h>
#include <string>
#endif
using namespace doctest;
using namespace pscm;
using namespace std::string_literals;
using namespace doctest;

TEST_CASE("testing quote, num") {
  Scheme scm(true);
  auto ret = scm.eval("'1");
  CHECK(ret == "1"_num);
}

TEST_CASE("testing quote, list") {
  Scheme scm(true);
  auto ret = scm.eval("'(0 1 2)");
  auto expected = list(0, 1, 2);
  std::cout << expected.to_string() << std::endl;
  CHECK(ret == Cell(expected));
}

TEST_CASE("testing quote, list 2") {
  Scheme scm(true);
  auto ret = scm.eval("'(45)");
  auto expected = list(45);
  std::cout << expected.to_string() << std::endl;
  CHECK(ret == Cell(expected));
}

TEST_CASE("testing proc, +") {
  Scheme scm(true);
  auto ret = scm.eval("(+ (+ (+ 3) 2 ))");
  REQUIRE_MESSAGE(ret == 5, ret);
}

TEST_CASE("testing proc, -") {
  Scheme scm(true);
  auto ret = scm.eval("(- 1)");
  CHECK(ret == "-1"_num);
}

TEST_CASE("testing proc, body 0 expr") {
  Scheme scm(true);
  auto ret = scm.eval("((lambda () 6))");
  REQUIRE_MESSAGE(ret == 6, ret);
}

TEST_CASE("testing proc, body 1 expr") {
  Scheme scm(true);
  auto ret = scm.eval("((lambda () (* 2 3)))");
  REQUIRE_MESSAGE(ret == 6, ret);
}

TEST_CASE("testing proc, body 2 expr") {
  Scheme scm(true);
  auto ret = scm.eval("((lambda () (* 2 3) (* 3 4)))");
  REQUIRE_MESSAGE(ret == 12, ret);
}

TEST_CASE("testing for-each") {
  Scheme scm(true);
  auto ret = scm.eval("for-each");
  REQUIRE(ret.is_proc());
  ret = scm.eval(R"(
(for-each (lambda (x) x) '(54))
)");
  CHECK(ret == Cell::none());
}

TEST_CASE("testing for-each, 2") {
  Scheme scm(true);
  auto ret = scm.eval(R"(
(for-each (lambda (x) x) '(54 0))
)");
  CHECK(ret == Cell::none());
}

TEST_CASE("testing apply, 1") {
  Scheme scm(true);
  auto ret = scm.eval("(apply + '())");
  CHECK(ret == "0"_num);
}

TEST_CASE("testing negative, -3") {
  Scheme scm(true);
  auto ret = scm.eval("-3");
  CHECK(ret == "-3"_num);
}

TEST_CASE("testing for-each, 3") {
  Scheme scm(true);
  auto ret = scm.eval(R"(
(for-each (lambda (x) x) '(-3))
)");
  CHECK(ret == Cell::none());
}

TEST_CASE("testing for-each, 4") {
  Scheme scm(true);
  auto ret = scm.eval(R"(
(for-each (lambda (x) x) '(54 0 37 -3 245 19))
)");
  CHECK(ret == Cell::none());
}

TEST_CASE("testing let, 1") {
  Scheme scm(true);
  auto ret = scm.eval(R"(
(let ((x 2) (y 3))
  (* x y))
)");
  CHECK(ret == "6"_num);
}

TEST_CASE("testing let, 2") {
  Scheme scm(true);
  auto ret = scm.eval(R"(
(let ((x 2) (y 3))
  (let ((x 7)
        (z (+ x y)))
    (* z x)))
)");
  CHECK(ret == "35"_num);
}

TEST_CASE("testing let*") {
  Scheme scm(true);
  auto ret = scm.eval(R"(
(let ((x 2) (y 3))
  (let* ((x 7)
        (z (+ x y)))
    (* z x)))
)");
  CHECK(ret == "70"_num);
}

TEST_CASE("testing let_star, 2") {
  Scheme scm(true);
  auto ret = scm.eval(R"(
(let* ((a 1)
       (b 2))
  (+ a b))
)");
  CHECK(ret == "3"_num);
}

TEST_CASE("testing current continuation, define f first") {
  Scheme scm(true);
  auto ret = scm.eval("(define (f a) 2)");
  REQUIRE_MESSAGE(ret.is_none(), ret);
  ret = scm.eval("(call/cc f)");
  CHECK(ret == "2"_num);
}

TEST_CASE("testing current continuation") {
  Scheme scm(true);
  auto ret = scm.eval("(define cc #f)");
  REQUIRE_MESSAGE(ret.is_none(), ret);
  ret = scm.eval("cc");
  REQUIRE_MESSAGE(ret.is_bool(), ret);
  REQUIRE_MESSAGE(!ret.to_bool(), ret);
  ret = scm.eval("(+ (call/cc (lambda (return) (set! cc return) (* 2 3)))(+ 1 7))");
  REQUIRE_MESSAGE(ret == 14, ret);
  ret = scm.eval("cc");
  REQUIRE_MESSAGE(ret.is_cont(), ret);
  ret = scm.eval("(cc 10)");
  CHECK(ret == "18"_num);
  ret = scm.eval("(cc (* 2 3)))");
  CHECK(ret == "14"_num);
}

TEST_CASE("testing exit") {
  Scheme scm(true);
  auto ret = scm.eval(R"(
(call/cc
 (lambda (exit)
   (for-each (lambda (x)
	       (if (negative? x) (exit x)))
	     '(54 0 37 -3 245 19))
   #t))
)");
  CHECK(ret == "-3"_num);
}

TEST_CASE("testing normal func") {
  Scheme scm(true);
  auto ret = scm.eval(R"(
(define (f return)
  (return 2)
  3)
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval(R"(
(f (lambda (x) x))
)");
  CHECK(ret == "3"_num);
}

TEST_CASE("testing call/cc") {
  Scheme scm(true);
  auto ret = scm.eval(R"(
(define (f return)
  (return 2)
  3)
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval(R"(
(call/cc f)
)");
  CHECK(ret == "2"_num);
}

TEST_CASE("testing call/cc 2") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval(R"(
(define c #f)
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval(R"(
(call/cc
  (lambda (c0)
          (set! c c0)
          'talk1))
)");
  REQUIRE(ret == "talk1"_sym);
  ret = scm.eval("(c 'talk2)");
  CHECK(ret == "talk2"_sym);
}

TEST_CASE("testing generator") {
  Scheme scm(true);
  auto ret = scm.eval(R"(
(define (generate-one-element-at-a-time lst)
  (define (control-state return)
    (for-each
      (lambda (element)
        (set! return (call/cc
          (lambda (resume-here)
            (set! control-state resume-here)
            (return element)))))
      lst)
    (return 'you-fell-off-the-end))
  (define (generator)
    (call/cc control-state))
 generator)
)");
  REQUIRE_MESSAGE(ret.is_none(), ret);
  ret = scm.eval("(generate-one-element-at-a-time '(0 1 2))");
  REQUIRE(ret.is_proc());
  std::cout << ret.to_string() << std::endl;
  ret = scm.eval("(define generate-digit (generate-one-element-at-a-time '(0 1 2)))");
  REQUIRE_MESSAGE(ret.is_none(), ret);
  ret = scm.eval("(generate-digit)");
  REQUIRE(ret == "0"_num);
  ret = scm.eval("(generate-digit)");
  REQUIRE(ret == "1"_num);
  ret = scm.eval("(generate-digit)");
  REQUIRE(ret == "2"_num);
  ret = scm.eval("(generate-digit)");
  REQUIRE(ret == "you-fell-off-the-end"_sym);
}

TEST_CASE("testing values, 1") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval(R"(
(call-with-values (lambda () (values 4 5))
         (lambda (a b) b))
)");
  CHECK(ret == "5"_num);
}

TEST_CASE("testing values, 2") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval(R"(
(call-with-values * -)
)");
  CHECK(ret == "-1"_num);
}

TEST_CASE("testing values, 1, with callcc") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval(R"(
(define (values . things)
  (call/cc
    (lambda (cont) (apply cont things))))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval(R"(
(call-with-values (lambda () (values 4 5))
         (lambda (a b) b))
)");
  CHECK(ret == "5"_num);
}

TEST_CASE("testing values, 2, with callcc") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval(R"(
(define (values . things)
  (call/cc
    (lambda (cont) (apply cont things))))
)");
  REQUIRE(ret == Cell::none());
  ret = scm.eval(R"(
(call-with-values * -)
)");
  CHECK(ret == "-1"_num);
}

TEST_CASE("testing call/cc, procedure?") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(call/cc procedure?)");
  CHECK(ret == Cell::bool_true());
}

TEST_CASE("testing call/cc, procedure? 2") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(procedure? procedure?)");
  CHECK(ret == Cell::bool_true());
}

TEST_CASE("testing call/cc, boolean?") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(call/cc boolean?)");
  CHECK(ret == Cell::bool_false());
}

TEST_CASE("testing call/cc, nested call") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(define (f c) c)");
  ret = scm.eval("(call/cc f)");
  CHECK(ret.is_cont());
  ret = scm.eval("(call/cc (call/cc f))");
  CHECK(ret.is_cont());
}
