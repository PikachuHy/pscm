
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

TEST_CASE("testing eval") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(define (a b) 1)");
    ret = scm.eval("(eval '(a 1))");
    CHECK(ret == "1"_num);
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