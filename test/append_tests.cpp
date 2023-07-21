
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

TEST_CASE("testing append, 0") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define a '(1 2 3))
)");
    ret = scm.eval("(append a 4)");
    ret = scm.eval(R"(
a
)");
    CHECK(ret == list(1, 2, 3));
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

TEST_CASE("testing append, 1") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(append '(a b) '(c . d))");
    Cell ret2 = scm.eval("`(a b c . d)");
    CHECK(ret == ret2);
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

TEST_CASE("testing append, ()") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(apply append '())");
    CHECK(ret == nil);
    ret = scm.eval("(apply append '(()))");
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

TEST_CASE("testing append, append 3 elements") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(append '(1 2) '(3 4) '(5 6))");
    auto expected = Parser("(1 2 3 4 5 6)").parse();
    CHECK(ret == expected);
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

TEST_CASE("testing append, apply append 3 elements") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(apply append '((1 2) (3 4) (5 6)))");
    auto expected = Parser("(1 2 3 4 5 6)").parse();
    CHECK(ret == expected);
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