#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "core/Scheme.h"
#include "core/Value.h"
#include "doctest/doctest.h"
using namespace doctest;
using namespace pscm::core;
using namespace std::string_literals;
using namespace doctest;

TEST_CASE("testing add") {
  std::string code = R"(
(+ 2 3)
)";
  Scheme scm;
  auto ret = scm.eval(code.c_str());
  CHECK(ret);
  CHECK(ret->to_string() == "5"s);
}

TEST_CASE("testing add 3") {
  std::string code = R"(
(+ 2 3 4)
)";
  Scheme scm;
  auto ret = scm.eval(code.c_str());
  CHECK(ret);
  CHECK(ret->to_string() == "9"s);
}

TEST_CASE("testing minus") {
  std::string code = R"(
(- 2)
)";
  Scheme scm;
  auto ret = scm.eval(code.c_str());
  CHECK(ret);
  CHECK(ret->to_string() == "-2"s);
}

TEST_CASE("testing minus, 2") {
  std::string code = R"(
(- 2 3)
)";
  Scheme scm;
  auto ret = scm.eval(code.c_str());
  CHECK(ret);
  CHECK(ret->to_string() == "-1"s);
}

TEST_CASE("testing function") {
  std::string code = R"(
(define (sum a b c)
  (+ a b c))
(sum 1 2 3)
)";
  Scheme scm;
  auto ret = scm.eval(code.c_str());
  CHECK(ret);
  CHECK(ret->to_string() == "6"s);
}

TEST_CASE("testing cond") {
  std::string code = R"(
(cond ((> 3 2) 100)
      ((< 3 2) 200))
)";
  Scheme scm;
  auto ret = scm.eval(code.c_str());
  CHECK(ret);
  CHECK(ret->to_string() == "100"s);
}

TEST_CASE("testing function") {
  // 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
  std::string code = R"(
(define (fib n)
  (cond ((< n 2) 0)
        ((< n 4) 1)
        (else (+ (fib (- n 1)) (fib (- n 2))))))
(fib 10)
)";
  Scheme scm;
  auto ret = scm.eval(code.c_str());
  CHECK(ret);
  CHECK(ret->to_string() == "34"s);
}

TEST_CASE("testing loop, map, abs function") {
  // (map abs '(4 -5 6))
  std::string code = R"(
(define (abs n)
        (cond ((< n 0) (- n))
              (else n)))
(abs -5)
)";
  Scheme scm;
  auto ret = scm.eval(code.c_str());
  CHECK(ret);
  CHECK(ret->to_string() == "5"s);
}

TEST_CASE("testing loop, map") {
  // (map abs '(4 -5 6))
  std::string code = R"(
(define (abs n)
        (cond ((< n 0) (- n))
              (else n)))
(map abs '(4 -5 6))
)";
  Scheme scm;
  scm.set_logger_level(0);
  auto ret = scm.eval(code.c_str());
  CHECK(ret);
  CHECK(ret->to_string() == "(4 5 6)"s);
}

TEST_CASE("testing loop, map, in func") {
  std::string code = R"(
(define (abs n)
        (cond ((< n 0) (- n))
              (else n)))
(define (map-fn list) (map abs list))
(map-fn '(4 -5 6))
)";
  Scheme scm;
  auto ret = scm.eval(code.c_str());
  CHECK(ret);
  CHECK(ret->to_string() == "(4 5 6)"s);
}

TEST_CASE("testing list, car") {
  std::string code = R"(
(car '(4 -5 6))
)";
  Scheme scm;
  auto ret = scm.eval(code.c_str());
  CHECK(ret);
  CHECK(ret->to_string() == "4"s);
}