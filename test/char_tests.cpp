//
// Created by PikachuHy on 2023/2/23.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#ifdef PSCM_USE_CXX20_MODULES
import pscm;
#else
#include <pscm/Char.h>
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

TEST_CASE("integer->char, 59") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(integer->char 59)
)");
    CHECK(ret == Char::from(';'));
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

TEST_CASE("char->integer, .Aa") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(char->integer #\.)
)");
    CHECK(ret == char('.'));
    ret = scm.eval(R"(
(char->integer #\A)
)");
    CHECK(ret == char('A'));
    ret = scm.eval(R"(
(char->integer #\a)
)");
    CHECK(ret == char('a'));
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

TEST_CASE("(integer->char (char->integer)), .Aa") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(integer->char (char->integer #\.))
)");
    CHECK(ret == Char::from('.'));
    ret = scm.eval(R"(
(integer->char (char->integer #\A))
)");
    CHECK(ret == Char::from('A'));
    ret = scm.eval(R"(
(integer->char (char->integer #\a))
)");
    CHECK(ret == Char::from('a'));
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

TEST_CASE("testing .") {

  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
#\.
)");
    CHECK(ret == Char::from('.'));
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

TEST_CASE("testing test expr") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
`(test #\. integer->char (char->integer #\.))
)");
    CHECK(cadr(ret) == Char::from('.'));
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
