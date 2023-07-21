
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

TEST_CASE("testing keywords, 0") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
:use
)");
    CHECK(ret.is_keyword());
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

TEST_CASE("testing keywords, ==") {
  auto f = [](Scheme& scm) {
    Cell ret;
    auto keyword1 = scm.eval(R"(
:use
)");
    CHECK(keyword1.is_keyword());
    auto keyword2 = scm.eval(R"(
:use
)");
    CHECK(keyword2.is_keyword());
    CHECK(keyword1 == keyword2);
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

TEST_CASE("testing keywords, eqv?") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(eqv? :check-mark :check-mark)
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
