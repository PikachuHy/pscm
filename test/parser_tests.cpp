//
// Created by PikachuHy on 2023/4/5.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <pscm/Number.h>
#include <pscm/Pair.h>
#include <pscm/Parser.h>
#include <pscm/Scheme.h>
#include <pscm/Symbol.h>
#include <pscm/scm_utils.h>
#include <string>
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