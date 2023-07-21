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

TEST_CASE("testing apply +") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(apply + '(3 4))
)");
    CHECK(ret == 7);
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

TEST_CASE("testing apply proc +") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define (test expect fun . args)
  ((lambda (res)
           (cond ((not (equal? expect res)) #f)
                  (else #t)))
    (apply fun args)))
)
)");
    REQUIRE(ret == Cell::none());
    ret = scm.eval("(test 7 apply + '(3 4))");
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

TEST_CASE("testing apply + list") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(apply + 10 (list 3 4))");
    CHECK(ret == 17);
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

TEST_CASE("testing apply lambda 1 arg") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(apply (lambda (x) (+ x x)) '(3))
)");
    CHECK(ret == 6);
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