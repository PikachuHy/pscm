//
// Created by PikachuHy on 2023/2/23.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#ifdef PSCM_USE_CXX20_MODULES
import pscm;
#else
#include <pscm/Number.h>
#endif
using namespace doctest;
using namespace pscm;
#if defined(__APPLE__) && defined(PSCM_STD_COMPAT)
// FIXME: maybe some bug on macOS
//  when using https://github.com/mpark/variant.git
//  with bazel

#else
TEST_CASE("testing _num, int") {
  auto num = "123"_num;
  CHECK(num == Number(123));
}

TEST_CASE("testing _num, float") {
  auto num = "12.7"_num;
  CHECK(num == Number(12.7));
}

TEST_CASE("testing _num, negative") {
  auto num = "-3"_num;
  REQUIRE(num.is_int());
  CHECK(num == Number(-3));
}
#endif
