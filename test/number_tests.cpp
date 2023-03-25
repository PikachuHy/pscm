//
// Created by PikachuHy on 2023/2/23.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#define private public
#include <pscm/Number.h>
using namespace doctest;
using namespace pscm;

TEST_CASE("testing _num, int") {
  auto num = "123"_num;
  REQUIRE(num.data_.index() == 1);
  CHECK(std::get<1>(num.data_) == 123);
}

TEST_CASE("testing _num, float") {
  auto num = "12.7"_num;
  REQUIRE(num.data_.index() == 2);
  CHECK(std::get<2>(num.data_) == 12.7);
}

TEST_CASE("testing _num, negative") {
  auto num = "-3"_num;
  REQUIRE(num.data_.index() == 1);
  CHECK(std::get<1>(num.data_) == -3);
}
