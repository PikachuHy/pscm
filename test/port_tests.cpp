
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

TEST_CASE("testing call-with-output-file") {

  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(call-with-output-file "tmp.txt"
    (lambda (port)
            (write 1 port)
            (newline port)))
)");
    CHECK(fs::exists("tmp.txt"));
    std::fstream fin;
    fin.open("tmp.txt", std::ios::in);
    REQUIRE(fin.is_open());
    Parser parser(&fin);
    ret = parser.next();
    CHECK(ret == 1);
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

TEST_CASE("testing call-with-input-file") {

  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(call-with-output-file "tmp.txt" 
    (lambda (port)
            (write 1 port)
            (newline port)))
)");
    CHECK(fs::exists("tmp.txt"));
    ret = scm.eval(R"(
(call-with-input-file "tmp.txt"
    (lambda (port)
        (input-port? port)))
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

TEST_CASE("testing write \"") {

  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(call-with-output-file "tmp2.txt"
    (lambda (port)
            (write "te \" \" st" port)
            (newline port)))
)");
    CHECK(fs::exists("tmp2.txt"));
    std::fstream fin;
    fin.open("tmp2.txt", std::ios::in);
    REQUIRE(fin.is_open());
    Parser parser(&fin);
    ret = parser.next();
    CHECK(ret == "te \" \" st"_str);
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
