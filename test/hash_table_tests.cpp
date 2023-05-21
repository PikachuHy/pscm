//
// Created by PikachuHy on 2023/5/21.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <pscm/Number.h>
#include <pscm/Pair.h>
#include <pscm/Scheme.h>
#include <pscm/Str.h>
#include <pscm/Symbol.h>
#include <pscm/scm_utils.h>
#include <string>
using namespace doctest;
using namespace pscm;
using namespace std::string_literals;
using namespace doctest;

TEST_CASE("testing hash-table") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define h (make-hash-table 31))
)");
    ret = scm.eval("h");
    CHECK(ret.is_hash_table());
    ret = scm.eval(R"(
(hash-set! h 'foo "bar")
)");
    CHECK(ret == "bar"_str);
    ret = scm.eval(R"(
(hash-ref h 'foo)
)");
    CHECK(ret == "bar"_str);
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
