#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <pscm/Number.h>
#include <pscm/Pair.h>
#include <pscm/Scheme.h>
#include <pscm/Symbol.h>
#include <pscm/scm_utils.h>
#include <string>
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
