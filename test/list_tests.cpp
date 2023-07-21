#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#ifdef PSCM_USE_CXX20_MODULES
import pscm;
#else
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

TEST_CASE("testing append empty") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(append)
)");
    CHECK(ret == nil);
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

TEST_CASE("testing list only") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(list)
)");
    CHECK(ret == nil);
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

TEST_CASE("testing list-head") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define a '(a b c d))
)");
    auto a = "a"_sym;
    auto b = "b"_sym;
    auto c = "c"_sym;
    auto d = "d"_sym;
    ret = scm.eval("a");
    CHECK(ret == list(&a, &b, &c, &d));
    ret = scm.eval("(define b (list-head a 2))");
    ret = scm.eval("b");
    CHECK(ret == list(&a, &b));
    ret = scm.eval("(set-cdr! b 2)");
    ret = scm.eval("b");
    auto n2 = "2"_num;
    CHECK(ret == Cell(cons(&a, &n2)));
    ret = scm.eval("a");
    CHECK(ret == list(&a, &b, &c, &d));
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

TEST_CASE("testing list-tail") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define a '(a b c d))
)");
    auto a = "a"_sym;
    auto b = "b"_sym;
    auto c = "c"_sym;
    auto d = "d"_sym;
    ret = scm.eval("a");
    CHECK(ret == list(&a, &b, &c, &d));
    ret = scm.eval("(define b (list-tail a 2))");
    ret = scm.eval("b");
    CHECK(ret == list(&c, &d));
    ret = scm.eval("(set-cdr! b 2)");
    ret = scm.eval("b");
    auto n2 = "2"_num;
    CHECK(ret == Cell(cons(&c, &n2)));
    ret = scm.eval("a");
    CHECK(ret == list(&a, &b, &c, &d));
    ret = scm.eval("(list-tail a 3)");
    CHECK(ret == list(&b, &c, &d));
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

TEST_CASE("testing last-pair") {
  auto f = [](Scheme& scm) {
    Cell ret;
    auto a = "a"_sym;
    auto b = "b"_sym;
    auto c = "c"_sym;
    auto d = "d"_sym;
    ret = scm.eval("(last-pair '(a b c d))");
    CHECK(ret == list(&d));
    ret = scm.eval("(last-pair '())");
    CHECK(ret == Cell::nil());
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