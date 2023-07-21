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

TEST_CASE("testing cond, empty") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(cond ((eq? 1 2))
      ((eq? 1 1))
      (else 'a))
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

TEST_CASE("testing 4.2.1, Conditionals, cond") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(cond ((> 3 2) 'greater)
      ((< 3 2) 'less))
)");
    CHECK(ret == "greater"_sym);
    ret = scm.eval(R"(
    (cond ((> 3 3) 'greater)
          ((< 3 3) 'less)
          (else 'equal))
    )");
    CHECK(ret == "equal"_sym);
    ret = scm.eval(R"(
    (cond ((assv 'b '((a 1) (b 2))) => cadr)
          (else #f))
    )");
    CHECK(ret == 2);
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

TEST_CASE("testing cond, multi statements") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(cond ((> 3 2) (+ 1) (+ 2)))
)");
    CHECK(ret == 2);
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

TEST_CASE("testing cond, =>") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(let ((=> 1)) (cond (#t => 'ok)))
)");
    CHECK(ret == "ok"_sym);
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

TEST_CASE("testing cond, else multi clause") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(cond
  ((= 1 2) '())
  (else 1 2 3))
)");
    CHECK(ret == "3"_num);
  };
  {
    Scheme scm;
    f(scm);
  }
  {
    // Scheme scm(true);
    // f(scm);
  }
}
