//
// Created by PikachuHy on 2023/4/5.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#ifdef PSCM_USE_CXX20_MODULES
import pscm;
#else
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

TEST_CASE("testing parse .") {
  Parser parser("(a b . (c d))");
  auto ret = parser.parse();
  auto a = "a"_sym;
  auto b = "b"_sym;
  auto c = "c"_sym;
  auto d = "d"_sym;
  CHECK(ret == list(a, b, c, d));
}

// TEST_CASE("testing parse stream") {
//   std::stringstream ss;
//   ss << "(a b . (c d))";
//   Parser parser((std::istream *)&ss);
//   auto ret = parser.parse();
//   auto a = "a"_sym;
//   auto b = "b"_sym;
//   auto c = "c"_sym;
//   auto d = "d"_sym;
//   CHECK(ret == list(a, b, c, d));
// }

// TEST_CASE("testing parse stream 2") {
//   std::stringstream ss;
//   ss << "a b c d";
//   Parser parser((std::istream *)&ss);

//   auto a = "a"_sym;
//   auto b = "b"_sym;
//   auto c = "c"_sym;
//   auto d = "d"_sym;
//   auto ret = parser.next();
//   CHECK(ret == a);
//   ret = parser.next();
//   CHECK(ret == b);
// }

TEST_CASE("testing parse #\\ ") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
#\
)");
    CHECK(ret == Char::from(' '));
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

TEST_CASE("testing parse #\\Space") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
#\Space
)");
    CHECK(ret == Char::from(' '));
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

TEST_CASE("testing parse #\\ #\\Space") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(eqv? '#\ #\Space)
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

TEST_CASE("testing parse \"") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
"The word \"recursion\" has many meanings."
)");
    CHECK(ret == pscm::String("The word \"recursion\" has many meanings."));
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

TEST_CASE("testing ;;") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define (float-print-test x)
              ;;   (else (display xx) (newline))
              '2
              )
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

TEST_CASE("testing parse ,@") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
`(,@(cdr '(c)))
)");
    CHECK(ret == nil);
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
#ifndef WASM_PLATFORM
TEST_CASE("testing parse 3.3I") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(string->number "3.3I")
)");
    CHECK(ret == Cell::bool_false());
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
#endif
TEST_CASE("testing parse #i-i") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(string->number "#i-i")
)");
    CHECK(ret == Number(Complex(0, -1)));
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