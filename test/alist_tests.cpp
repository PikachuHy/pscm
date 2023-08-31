#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#ifdef PSCM_USE_CXX20_MODULES
import pscm;
#else
#include <pscm/Number.h>
#include <pscm/Pair.h>
#include <pscm/Parser.h>
#include <pscm/Scheme.h>
#include <pscm/Str.h>
#include <pscm/Symbol.h>
#include <pscm/scm_utils.h>
#include <string>
#endif
using namespace doctest;
using namespace pscm;
using namespace std::string_literals;
using namespace doctest;

namespace pscm{
doctest::String toString(const Cell& value) {
  std::string str;
  value.to_string().toUTF8String(str);
  return {str.data(), static_cast<doctest::String::size_type>(str.size())};
}}

TEST_CASE("testing acons") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(acons 3 "pay gas bill" '())
)");
    auto expected = Parser(R"(((3 . "pay gas bill")))").parse();
    CHECK(ret == expected);
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

TEST_CASE("testing acons, 1") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(define task-list '())");
    ret = scm.eval(R"(
(set! task-list (acons 3 "pay gas bill" '()))
)");
    ret = scm.eval(R"(
(acons 3 "tidy bedroom" task-list)
)");
    auto expected = Parser(R"(((3 . "tidy bedroom") (3 . "pay gas bill")))").parse();
    CHECK(ret == expected);
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

TEST_CASE("testing assoc-set!") {
  auto f = [](Scheme& scm) {
    Cell ret;
    // (("mary" . "34 Elm Road") ("james" . "16 Bow Street"))
    ret = scm.eval(R"(
(define address-list (acons "mary" "34 Elm Road" (acons "james" "16 Bow Street" '())))
)");
    ret = scm.eval("address-list");
    auto expected = Parser(R"(
(("mary" . "34 Elm Road") ("james" . "16 Bow Street"))
    )")
                        .parse();
    CHECK(ret == expected);
    ret = scm.eval(R"(
(assoc-set! address-list "james" "1a London Road")
)");
    expected = Parser(R"(
(("mary" . "34 Elm Road") ("james" . "1a London Road"))
    )")
                   .parse();
    CHECK(ret == expected);
    ret = scm.eval("address-list");
    CHECK(ret == expected);
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

TEST_CASE("testing assoc-set!, 2") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define address-list (acons "mary" "34 Elm Road" (acons "james" "1a London Road" '())))
)");
    ret = scm.eval("address-list");
    auto expected = Parser(R"(
(("mary" . "34 Elm Road") ("james" . "1a London Road"))
    )")
                        .parse();
    CHECK(ret == expected);
    ret = scm.eval(R"(
(assoc-set! address-list "bob" "11 Newington Avenue")
)");
    auto expected2 = Parser(R"(
(("bob" . "11 Newington Avenue") ("mary" . "34 Elm Road")
 ("james" . "1a London Road"))
    )")
                         .parse();
    CHECK(ret == expected2);
    ret = scm.eval("address-list");
    CHECK(ret == expected);
    ret = scm.eval(R"(
      (set! address-list
      (assoc-set! address-list "bob" "11 Newington Avenue"))
    )");
    ret = scm.eval("address-list");
    CHECK(ret == expected2);
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

TEST_CASE("testing assoc-remove!") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define address-list '(("bob" . "11 Newington Avenue") ("mary" . "34 Elm Road")
 ("james" . "1a London Road")))
)");
    ret = scm.eval(R"(
      (set! address-list (assoc-remove! address-list "mary"))
    )");
    ret = scm.eval("address-list");
    auto expected = Parser(R"(
(("bob" . "11 Newington Avenue") ("james" . "1a London Road"))
    )")
                        .parse();
    CHECK(ret == expected);
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

TEST_CASE("testing assoc-remove!, 2") {
  auto f = [](Scheme& scm) {
    Cell ret;
    scm.eval_all(R"(

      (define address-list '())
(set! address-list (assq-set! address-list "mary" "11 Elm Street"))
(set! address-list (assq-set! address-list "mary" "57 Pine Drive"))
    )");
    ret = scm.eval(R"(
address-list
)");
    auto expected = Parser(R"(
(("mary" . "57 Pine Drive") ("mary" . "11 Elm Street"))
    )")
                        .parse();
    CHECK(ret == expected);
    ret = scm.eval(R"(
      (set! address-list (assoc-remove! address-list "mary"))
    )");
    ret = scm.eval("address-list");
    auto expected2 = Parser(R"(
( ("mary" . "11 Elm Street"))
    )")
                         .parse();
    CHECK(ret == expected2);
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

TEST_CASE("testing alist") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
      (define capitals '(("New York" . "Albany")
                   ("Oregon"   . "Salem")
                   ("Florida"  . "Miami")))

    )");
    ret = scm.eval(R"(
      (assoc "Oregon" capitals)
    )");
    auto expected = Parser(R"(("Oregon" . "Salem"))").parse();
    CHECK(ret == expected);
    ret = scm.eval(R"(
      (assoc-ref capitals "Oregon")
    )");
    CHECK(ret == "Salem"_str);
    scm.eval(R"(
      (set! capitals
      (assoc-set! capitals "South Dakota" "Pierre"))
    )");
    ret = scm.eval("capitals");
    expected = Parser(R"(
      (("South Dakota" . "Pierre")
    ("New York" . "Albany")
    ("Oregon" . "Salem")
    ("Florida" . "Miami"))
    )")
                   .parse();
    CHECK(ret == expected);
    scm.eval(R"(
(set! capitals
      (assoc-set! capitals "Florida" "Tallahassee"))
    )");
    expected = Parser(R"(
(("South Dakota" . "Pierre")
    ("New York" . "Albany")
    ("Oregon" . "Salem")
    ("Florida" . "Tallahassee"))
    )")
                   .parse();
    CHECK(ret == expected);
    scm.eval(R"(
      (set! capitals
      (assoc-remove! capitals "Oregon"))
    )");

    expected = Parser(R"(
(("South Dakota" . "Pierre")
    ("New York" . "Albany")
    ("Florida" . "Tallahassee"))
    )")
                   .parse();
    CHECK(ret == expected);
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
