
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <filesystem>
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
using namespace doctest;
using namespace pscm;
using namespace std::string_literals;
using namespace doctest;
namespace fs = std::filesystem;

TEST_CASE("testing append, 0") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define a '(1 2 3))
)");
    ret = scm.eval("(append a 4)");
    ret = scm.eval(R"(
a
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

TEST_CASE("testing append, 1") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(append '(a b) '(c . d))");
    Cell ret2 = scm.eval("`(a b c . d)");
    CHECK(ret == ret2);
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