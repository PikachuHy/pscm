//
// Created by PikachuHy on 2023/3/4.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#ifdef PSCM_USE_CXX20_MODULES
import pscm;
#else
#include "unicode/ustream.h"
#include <pscm/Cell.h>
#include <pscm/Number.h>
#include <pscm/Pair.h>
#include <pscm/Procedure.h>
#include <pscm/Symbol.h>
#endif
using namespace pscm;
using namespace doctest;

TEST_CASE("testing to_string, pair") {
  Cell cell = cons(new Number(1), new Number(2));
  auto s = cell.to_string();
  CHECK(s == "(1 . 2)");
}

TEST_CASE("testing to_string, list") {
  Cell cell = cons(new Number(1), cons(new Number(2), nil));
  auto s = cell.to_string();
  CHECK(s == "(1 2)");
}

TEST_CASE("testing to_string, proc") {
  auto sym = new Symbol("square");
  Cell args = cons(new Symbol("x"), nil);
  Cell proc = new Procedure(sym, args, nil, nullptr);
  auto s = proc.to_string();
  CHECK(s == "#<procedure square (x)>");
}