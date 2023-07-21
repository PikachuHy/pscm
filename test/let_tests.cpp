
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

TEST_CASE("testing let, 0") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define x '(1 3 5 7 9))
)");
    ret = scm.eval("(define sum 0)");
    ret = scm.eval(R"(
(do ((x x (cdr x)))
    ((null? x))
    (set! sum (+ sum (car x))))
)");
    ret = scm.eval("sum");
    CHECK(ret == "25"_num);
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

TEST_CASE("testing let, 2") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(let ((x '(1 3 5 7 9))
                   (sum 0))
               (do ((x x (cdr x)))
                   ((null? x))
                 (set! sum (+ sum (car x))))
               sum)
)");
    CHECK(ret == "25"_num);
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
