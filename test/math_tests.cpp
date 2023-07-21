//
// Created by PikachuHy on 2023/2/23.
//
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

TEST_CASE("testing expt -1 -255") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(expt -1 -255)
)");
    CHECK(ret == -1);
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

TEST_CASE("testing number?") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(number? 3)
)");
    CHECK(ret == Cell::bool_true());
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

TEST_CASE("testing /") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(/ 1 3)
)");
    Number num(Rational(1, 3));
    CHECK(ret == Cell(&num));
    ret = scm.eval(R"(
(* (/ 1 3) 100)
)");
    Number num2(Rational(100, 3));
    CHECK(ret == Cell(&num2));
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