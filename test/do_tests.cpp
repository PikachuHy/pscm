#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#ifdef PSCM_USE_CXX20_MODULES
import pscm;
#else
#include <pscm/Number.h>
#include <pscm/Pair.h>
#include <pscm/Scheme.h>
#include <pscm/Str.h>
#include <pscm/Symbol.h>
#include <pscm/scm_utils.h>
#include <string>
#endif
using namespace doctest;
using namespace pscm;
using namespace std::string_literals;
using namespace doctest;

TEST_CASE("testing do, 0") {
  Scheme scm;
  scm.eval("(define v \" abc \")");
  scm.eval(R"(
(do ((i (- (string-length v) 1) (- i 1)))
        ((< i 0) v)
     #t)
    )");
  scm.eval(R"(
(do ((i (- (string-length v) 1) (- i 1)))
        ((< i 0) v)
     #t)
    )");
}

TEST_CASE("testing do, 2") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(define s \" abc \")");
  ret = scm.eval(R"(
(let ((v (make-string (string-length s))))
    (do ((i (- (string-length v) 1) (- i 1)))
        ((< i 0) v)

     (string-set! v i (string-ref s i))

     ))
    )");
  CHECK(ret == pscm::String(" abc "));
  ret = scm.eval(R"(
(let ((v (make-string (string-length s))))
    (do ((i (- (string-length v) 1) (- i 1)))
        ((< i 0) v)

     (string-set! v i (string-ref s i))

     ))
    )");
  CHECK(ret == pscm::String(" abc "));
}

TEST_CASE("testing do, 3") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(define s \" abc \")");
  ret = scm.eval(R"(
(define (str-copy s)
  (let ((v (make-string (string-length s))))
    (do ((i (- (string-length v) 1) (- i 1)))
        ((< i 0) v)
      (string-set! v i (string-ref s i)))))
    )");
  CHECK(ret == Cell::none());
  ret = scm.eval(R"(
(str-copy s)
    )");
  CHECK(ret == pscm::String(" abc "));
  ret = scm.eval(R"(
(str-copy s)
    )");
  CHECK(ret == pscm::String(" abc "));
}