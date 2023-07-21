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

TEST_CASE("testing for_each, 2 args") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(for-each (lambda (x y) (+ x y))
          '(1 2)
          '(3 4))
)");
    CHECK(ret == Cell::none());
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

TEST_CASE("testing for_each, 1 args") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(for-each abs '(4 -5 6))
)");
    CHECK(ret == Cell::none());
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

TEST_CASE("testing for_each, internal test, 0") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define (test . args) #t)
)");
    ret = scm.eval(R"(
(for-each (lambda (x y) (list x y)) (list 'a) (list 'b))
)");
    CHECK(ret == Cell::none());
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

TEST_CASE("testing for_each, internal test, 1") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define (test . args) #t)
)");
    ret = scm.eval(R"(
      (for-each (lambda (x y)
                  (for-each (lambda (f)
                                    (test #f f x))
                            (list null?)))
                (list '()  '(test))
                (list '()  '(t . t)))
    )");
    CHECK(ret == Cell::none());
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

TEST_CASE("testing for_each, internal test, 1") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define (test . args) #t)
)");
    ret = scm.eval(R"(
      (for-each (lambda (x y)
                  (for-each (lambda (f)
                                    (test #f f x))
                            (list null? '(test))))
                (list '()  '(test))
                (list '()  '(t . t)))
    )");
    CHECK(ret == Cell::none());
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
