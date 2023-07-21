//
// Created by PikachuHy on 2023/5/21.
//
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

TEST_CASE("testing hash-table") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define h (make-hash-table 31))
)");
    ret = scm.eval("h");
    CHECK(ret.is_hash_table());
    ret = scm.eval(R"(
(hash-set! h 'foo "bar")
)");
    CHECK(ret == "bar"_str);
    ret = scm.eval(R"(
(hash-ref h 'foo)
)");
    CHECK(ret == "bar"_str);
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

TEST_CASE("testing hash-set!") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define h (make-hash-table 31))
)");
    ret = scm.eval("h");
    CHECK(ret.is_hash_table());
    ret = scm.eval(R"(
(hash-set! h :check-mark "bar")
)");
    CHECK(ret == "bar"_str);
    ret = scm.eval(R"(
(hash-ref h :check-mark)
)");
    CHECK(ret == "bar"_str);
  };
  {
    Scheme scm;
    f(scm);
  }
  {
    // Scheme scm(true);
    // f(scm);
  }
}

TEST_CASE("testing hash-table, hash-fold") {
  auto f = [](Scheme& scm) {
    Cell ret;
    scm.eval_all(R"(
      (define h (make-hash-table 31))
      (hash-set! h 'foo "bar")
      (hash-set! h 'braz "zonk")
    )");
    ret = scm.eval("(hash-fold (lambda (key value seed) (+ 1 seed)) 0 h)");
    CHECK(ret == "2"_num);
  };
  {
    Scheme scm;
    f(scm);
  }
  {
    // Scheme scm(true);
    // f(scm);
  }
}

TEST_CASE("testing hash-table, hash-fold, 2") {
  auto f = [](Scheme& scm) {
    Cell ret;
    scm.eval_all(R"(
      (define h (make-hash-table 31))
      (hash-set! h 'foo "bar")
      (hash-set! h 'braz "zonk")
    )");
    ret = scm.eval("(hashq-create-handle! h 'frob #f)");
    auto frob = "frob"_sym;
    CHECK(ret == Cell(cons(&frob, Cell::bool_false())));
    ret = scm.eval("(hashq-get-handle h 'foo)");
    auto foo = "foo"_sym;
    auto bar = "bar"_str;
    CHECK(ret == Cell(cons(&foo, &bar)));
    ret = scm.eval("(hashq-get-handle h 'not-there)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(hash-fold (lambda (key value seed) (+ 1 seed)) 0 h)");
    CHECK(ret == "3"_num);
  };
  {
    Scheme scm;
    f(scm);
  }
  {
    // Scheme scm(true);
    // f(scm);
  }
}

TEST_CASE("testing hash-table, hash-remove") {
  auto f = [](Scheme& scm) {
    Cell ret;
    scm.eval_all(R"(
      (define h (make-hash-table 31))
      (hash-set! h 'foo "bar")
      (hash-set! h 'braz "zonk")
      (hashq-create-handle! h 'frob #f)
    )");
    ret = scm.eval("(hash-fold (lambda (key value seed) (+ 1 seed)) 0 h)");
    CHECK(ret == "3"_num);
    scm.eval("(hash-remove! h 'foo)");
    ret = scm.eval("(hash-ref h 'foo)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(hash-fold (lambda (key value seed) (+ 1 seed)) 0 h)");
    CHECK(ret == "2"_num);
  };
  {
    Scheme scm;
    f(scm);
  }
  {
    // Scheme scm(true);
    // f(scm);
  }
}

TEST_CASE("testing hash-table, hash-fold") {
  auto f = [](Scheme& scm) {
    Cell ret;
    scm.eval(R"(
(define (ahash-table->list h)
	(hash-fold acons '() (car h)))
    )");
    scm.eval_all(R"(
      (define h (make-hash-table 31))
      (hash-set! h 'foo "bar")
      (hash-set! h 'braz "zonk")
    )");
    ret = scm.eval("(ahash-table->list (cons h '()))");
    auto expected = Parser(R"(
      ((braz . "zonk") (foo . "bar"))
    )")
                        .parse();
    CHECK(ret == expected);
  };
  {
    Scheme scm;
    f(scm);
  }
  {
    // Scheme scm(true);
    // f(scm);
  }
}
