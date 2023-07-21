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

TEST_CASE("testing map, 2 args") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(map (lambda (x y) (+ x y))
          '(1 2)
          '(3 4))
)");
    CHECK(ret == list(4, 6));
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

TEST_CASE("testing map, 1 args") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(map abs '(4 -5 6))
)");
    CHECK(ret == list(4, 5, 6));
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

TEST_CASE("testing map let set!") {
  auto f = [](Scheme& scm) {
    Cell ret;

    ret = scm.eval(R"(
(let ((count 0))
(map (lambda (ignored)
       (set! count (+ count 1))
       count)
     '(a b c)))
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