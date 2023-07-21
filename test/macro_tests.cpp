
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
(define-macro (a b) `(+ 1 ,b)))
)");
    ret = scm.eval("(a 2)");
    CHECK(ret == 3);
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

TEST_CASE("testing user defined do") {

  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define (str-copy v)
    (do ((i (- 2 1) (- i 1)))
        ((< i 0) v)
        (+ 1)))
)");
    ret = scm.eval(R"(
    (define-macro (do bindings test-and-result . body)
      (let ((variables (map car bindings))
            (inits     (map cadr bindings))
            (steps     (map (lambda (clause)
                              (if (null? (cddr clause))
                                  (car clause)
                                  (caddr clause)))
                            bindings))
            (test   (car test-and-result))
            (result (cdr test-and-result))
            (loop   (gensym)))

        `(letrec ((,loop (lambda ,variables
                           (if ,test
                               ,(if (not (null? result))
                                    `(begin . ,result))
                               (begin
                                 ,@body
                                 (,loop . ,steps))))))
           (,loop . ,inits)) ))
    )");
    ret = scm.eval(R"(
(str-copy "a")
)");
    CHECK(ret == "a"_str);
    ret = scm.eval(R"(
(str-copy "a")
)");
    CHECK(ret == "a"_str);
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