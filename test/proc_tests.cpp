
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#ifdef PSCM_USE_CXX20_MODULES
import pscm;
#else
#include <fstream>
#include <pscm/Char.h>
#include <pscm/Number.h>
#include <pscm/Pair.h>
#include <pscm/Parser.h>
#include <pscm/Scheme.h>
#include <pscm/Str.h>
#include <pscm/Symbol.h>
#include <pscm/scm_utils.h>
#include <sstream>
#include <string>
#endif
using namespace doctest;
using namespace pscm;
using namespace std::string_literals;
using namespace doctest;

TEST_CASE("testing user defined macro") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(noop)
)");
    CHECK(ret == Cell::bool_false());
  };
  {
    Scheme scm;
    f(scm);
  }
  {
    Scheme scm(true);
    f(scm);
  }
}

TEST_CASE("testing procedure-name, 1") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define (a b) 2)
)");
    ret = scm.eval("(procedure-name a)");
    CHECK(ret == "a"_sym);
    ret = scm.eval(R"(
(define b a)
)");
    ret = scm.eval("(procedure-name b)");
    CHECK(ret == "a"_sym);
  };
  {
    Scheme scm;
    f(scm);
  }
  {
    Scheme scm(true);
    f(scm);
  }
}

TEST_CASE("testing procedure-name, 2") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define c (lambda () '2))
)");
    ret = scm.eval("(procedure-name c)");
    CHECK(ret == "c"_sym);
  };
  {
    Scheme scm;
    f(scm);
  }
  {
    Scheme scm(true);
    f(scm);
  }
}

TEST_CASE("testing nested procedure") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define ((c b) a) '2)
)");
    ret = scm.eval("(c 2)");
    CHECK(ret.is_proc());
    ret = scm.eval("((c 2) 3)");
    CHECK(ret == 2);
  };
  {
    Scheme scm;
    f(scm);
  }
  {
    Scheme scm(true);
    f(scm);
  }
}

TEST_CASE("testing page 43, nested proc") {
  auto f = [](Scheme& scm) {
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
  };
  {
    Scheme scm;
    f(scm);
  }
  {
    Scheme scm(true);
    f(scm);
  }
}

TEST_CASE("testing page 43, nested proc") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define ((define-property* which) opt decl)
  `(,which (list ,@opt)))
)");
    REQUIRE(ret == Cell::none());
    ret = scm.eval("(define-property* 'a)");
    CHECK(ret.is_proc());
    ret = scm.eval("((define-property* 'a) '(c d) 'e)");
    auto expect = scm.eval("'(a (list c d))");
    CHECK(ret == expect);
  };
  {
    Scheme scm;
    f(scm);
  }
  {
    Scheme scm(true);
    f(scm);
  }
}