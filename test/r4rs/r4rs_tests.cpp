//
// Created by PikachuHy on 2023/3/25.
//
#define DOCTEST_CONFIG_IMPLEMENT
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
#endif
using namespace pscm;
using namespace doctest;

TEST_CASE("testing 1.3.4, Evaluation examples") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(* 5 8)");
  CHECK(ret == 40);
}

TEST_CASE("testing 1.3.4, Evaluation examples, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(* 5 8)");
  CHECK(ret == 40);
}

TEST_CASE("testing 2.2, Whitespace and comments") {
  Scheme scm;
  Cell ret;
  ret = scm.eval(R"(
;;; The FACT procedure computes the factorial
;;; of a non-negative integer.
(define fact
  (lambda (n)
    (if (= n 0)
      1         ;Base case: return 1
      (* n (fact (- n 1))))))
)");
  CHECK(ret == Cell::none());
  ret = scm.eval("(fact 0)");
  CHECK(ret == 1);
  ret = scm.eval("(fact 1)");
  CHECK(ret == 1);
  ret = scm.eval("(fact 2)");
  CHECK(ret == 2);
  ret = scm.eval("(fact 3)");
  CHECK(ret == 6);
}

TEST_CASE("testing 2.2, Whitespace and comments, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval(R"(
;;; The FACT procedure computes the factorial
;;; of a non-negative integer.
(define fact
  (lambda (n)
    (if (= n 0)
      1         ;Base case: return 1
      (* n (fact (- n 1))))))
)");
  CHECK(ret == Cell::none());
  ret = scm.eval("fact");
  REQUIRE(ret.is_proc());
  ret = scm.eval("(fact 0)");
  CHECK(ret == 1);
  ret = scm.eval("(fact 1)");
  CHECK(ret == 1);
  ret = scm.eval("(fact 2)");
  CHECK(ret == 2);
  ret = scm.eval("(fact 3)");
  CHECK(ret == 6);
}

TEST_CASE("testing 4.1.2, Literal expressions") {
  Scheme scm;
  Cell ret;
  ret = scm.eval("(quote a)");
  CHECK(ret == "a"_sym);
  ret = scm.eval("(quote #(a b c))");
  Cell::Vec vec;
  auto a = "a"_sym;
  auto b = "b"_sym;
  auto c = "c"_sym;
  vec.emplace_back(&a);
  vec.emplace_back(&b);
  vec.emplace_back(&c);
  CHECK(ret == vec);
  ret = scm.eval("(quote (+ 1 2))");
  auto plus = "+"_sym;
  auto one = "1"_num;
  auto two = "2"_num;
  CHECK(ret == cons(&plus, cons(&one, cons(&two, nil))));

  ret = scm.eval("'a");
  CHECK(ret == "a"_sym);
  ret = scm.eval("'#(a b c)");
  CHECK(ret == Cell(&vec));
  ret = scm.eval("'()");
  CHECK(ret == nil);
  ret = scm.eval("'(+ 1 2)");
  CHECK(ret == cons(&plus, cons(&one, cons(&two, nil))));
  ret = scm.eval("'(quote a)");
  auto quote = "quote"_sym;
  CHECK(ret == pscm::list(&quote, &a));
  ret = scm.eval("''a");
  CHECK(ret == pscm::list(&quote, &a));

  ret = scm.eval("'\"abc\"");
  pscm::String abc("abc");
  CHECK(ret == Cell(&abc));
  ret = scm.eval("\"abc\"");
  CHECK(ret == Cell(&abc));
  ret = scm.eval("'145932");
  CHECK(ret == 145932);
  ret = scm.eval("145932");
  CHECK(ret == 145932);
  ret = scm.eval("'#t");
  CHECK(ret == Cell::bool_true());
  ret = scm.eval("#t");
  CHECK(ret == Cell::bool_true());
}

TEST_CASE("testing 4.1.2, Literal expressions, with register machine") {
  Scheme scm(true);
  Cell ret;
  ret = scm.eval("(quote a)");
  CHECK(ret == "a"_sym);
  ret = scm.eval("(quote #(a b c))");
  Cell::Vec vec;
  auto a = "a"_sym;
  auto b = "b"_sym;
  auto c = "c"_sym;
  vec.emplace_back(&a);
  vec.emplace_back(&b);
  vec.emplace_back(&c);
  CHECK(ret == vec);
  ret = scm.eval("(quote (+ 1 2))");
  auto plus = "+"_sym;
  auto one = "1"_num;
  auto two = "2"_num;
  CHECK(ret == cons(&plus, cons(&one, cons(&two, nil))));

  ret = scm.eval("'a");
  CHECK(ret == "a"_sym);
  ret = scm.eval("'#(a b c)");
  CHECK(ret == vec);
  ret = scm.eval("'()");
  CHECK(ret == nil);
  ret = scm.eval("'(+ 1 2)");
  CHECK(ret == cons(&plus, cons(&one, cons(&two, nil))));
  ret = scm.eval("'(quote a)");
  auto quote = "quote"_sym;
  CHECK(ret == pscm::list(&quote, &a));
  ret = scm.eval("''a");
  CHECK(ret == pscm::list(&quote, &a));

  ret = scm.eval("'\"abc\"");
  pscm::String abc("abc");
  CHECK(ret == Cell(&abc));
  ret = scm.eval("\"abc\"");
  CHECK(ret == Cell(&abc));
  ret = scm.eval("'145932");
  CHECK(ret == 145932);
  ret = scm.eval("145932");
  CHECK(ret == 145932);
  ret = scm.eval("'#t");
  CHECK(ret == Cell::bool_true());
  ret = scm.eval("#t");
  CHECK(ret == Cell::bool_true());
}

TEST_CASE("testing 4.1.3, Procedure calls") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(+ 3 4)");
    CHECK(ret == 7);
    ret = scm.eval("((if #f + *) 3 4)");
    CHECK(ret == 12);
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

TEST_CASE("testing 4.1.4, Lambda expressions") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(lambda (x) (+ x x))");
    CHECK(ret.is_proc());
    ret = scm.eval("((lambda (x) (+ x x)) 4)");
    CHECK(ret == 8);
    ret = scm.eval(R"(
(define reverse-subtract
  (lambda (x y) (- y x)))
)");
    REQUIRE(ret == Cell::none());
    ret = scm.eval("(reverse-subtract 7 10)");
    CHECK(ret == 3);
    ret = scm.eval(R"(
(define add4
  (let ((x 4))
    (lambda (y) (+ x y))))
)");
    REQUIRE(ret == Cell::none());
    ret = scm.eval("(add4 6)");
    CHECK(ret == 10);

    ret = scm.eval("((lambda x x) 3 4 5 6)");
    CHECK(ret == list(3, 4, 5, 6));
    ret = scm.eval("((lambda (x y . z) z) 3 4 5 6)");
    CHECK(ret == list(5, 6));
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

TEST_CASE("testing 4.1.5, conditionals") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(if (> 3 2) 'yes 'no)");
    CHECK(ret == "yes"_sym);
    ret = scm.eval("(if (> 2 3) 'yes 'no)");
    CHECK(ret == "no"_sym);
    ret = scm.eval(R"(
(if (> 3 2)
    (- 3 2)
    (+ 3 2))
)");
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

TEST_CASE("testing 4.1.6, Assignments") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(define x 2)");
    REQUIRE(ret == Cell::none());
    ret = scm.eval("x");
    CHECK(ret == 2);
    ret = scm.eval("(+ x 1)");
    CHECK(ret == 3);
    ret = scm.eval("(set! x 4)");
    CHECK(ret == Cell::none());
    ret = scm.eval("(+ x 1)");
    CHECK(ret == 5);
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

TEST_CASE("testing 4.2.1, Conditionals, cond") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(cond ((> 3 2) 'greater)
      ((< 3 2) 'less))
)");
    CHECK(ret == "greater"_sym);
    ret = scm.eval(R"(
(cond ((> 3 3) 'greater)
      ((< 3 3) 'less)
      (else 'equal))
)");
    CHECK(ret == "equal"_sym);
    ret = scm.eval(R"(
(cond ((assv 'b '((a 1) (b 2))) => cadr)
      (else #f))
)");
    CHECK(ret == 2);
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

TEST_CASE("testing 4.2.1, Conditionals, case") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(case (* 2 3)
  ((2 3 5 7) 'prime)
  ((1 4 6 8 9) 'composite))
)");
    CHECK(ret == "composite"_sym);
    ret = scm.eval("(define var 'c)");
    ret = scm.eval(R"(
    (cond ((member var '(a)) 'a)
          ((member var '(b)) 'b))
    )");
    CHECK(ret == Cell::none());
    ret = scm.eval(R"(
(case (car '(c d))
  ((a) 'a)
  ((b) 'b))
)");
    CHECK(ret == Cell::none());
    ret = scm.eval(R"(
(case (car '(c d))
  ((a e i o u) 'vowel)
  ((w y) 'semivowel)
  (else 'constant))
)");
    CHECK(ret == "constant"_sym);
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

TEST_CASE("testing 4.2.1, Conditionals, and") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(and (= 2 2) (> 2 1))
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(and (= 2 2) (< 2 1))
)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(and 1 2 'c '(f g))");
    auto f = "f"_sym;
    auto g = "g"_sym;
    CHECK(ret == list(&f, &g));
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

TEST_CASE("testing 4.2.1, Conditionals, or") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(or (= 2 2) (> 2 1))
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(or (= 2 2) (< 2 1))
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(or #f #f #f)");
    CHECK(ret == Cell::bool_false());
    auto b = "b"_sym;
    auto c = "c"_sym;
    ret = scm.eval("(memq 'b '(a b c))");
    CHECK(ret == list(&b, &c));
    ret = scm.eval("(or (memq 'b '(a b c)) (/ 3 0))");
    CHECK(ret == list(&b, &c));
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

TEST_CASE("testing 4.2.2, Binding constructs") {

  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
    (let ((x 2) (y 3))
      (* x y))
    )");
    CHECK(ret == 6);
    ret = scm.eval(R"(
    (let ((x 2) (y 3))
      (* x y)
      8)
    )");
    CHECK(ret == 8);
    ret = scm.eval(R"(
    (let ((x 2) (y 3))
      (let ((x 7)
            (z (+ x y)))
        (* z x)))
    )");
    CHECK(ret == 35);
    ret = scm.eval(R"(
    (let ((x 2) (y 3))
      (let* ((x 7)
             (z (+ x y)))
        (* z x)))
    )");
    CHECK(ret == 70);
    ret = scm.eval(R"(
(letrec ((even?
          (lambda (n)
            (if (zero? n)
                #t
                (odd? (- n 1)))))
         (odd?
          (lambda (n)
            (if (zero? n)
                #f
                (even? (- n 1))))))
  (even? 88))
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
    (letrec ((fib (lambda (n)
                (cond ((zero? n) 1)
                      ((= 1 n) 1)
                      (else  (+ (fib (- n 1))
    			    (fib (- n 2))))))))
        (fib 10))
    )");
    CHECK(ret == 89);
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

TEST_CASE("testing 4.2.3, Sequencing") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(define x 0)");
    CHECK(ret == Cell::none());
    ret = scm.eval(R"(
(begin (set! x 5)
       (+ x 1))
)");
    CHECK(ret == 6);
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

TEST_CASE("testing 4.2.4, Iteration") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(do ((vec (make-vector 5))
     (i 0 (+ i 1)))
    ((= i 5) vec)
  (vector-set! vec i i))
)");
    Cell::Vec vec;
    for (int i = 0; i < 5; i++) {
      vec.push_back(new Number(i));
    }
    CHECK(ret == Cell(&vec));
    ret = scm.eval(R"(
(let ((x '(1 3 5 7 9)))
(do ((x x (cdr x))
     (sum 0 (+ sum (car x))))
    ((null? x) sum)))
)");
    CHECK(ret == 25);
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

TEST_CASE("testing 4.2.4, Iteration, named let") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(let loop ((numbers '(3 -2 1 6 -5))
	   (nonneg '())
	   (neg '()))
(cond ((null? numbers)
       (list nonneg neg))
      ((>= (car numbers) 0)
       (loop (cdr numbers)
	     (cons (car numbers) nonneg)
	     neg))
      ((< (car numbers) 0)
       (loop (cdr numbers) nonneg
	     (cons (car numbers) neg)))))
)");
    CHECK(ret == list(list(6, 1, 3), list(-5, -2)));
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

TEST_CASE("testing 4.2.5, Delay evaluation") {
  auto f = [](Scheme& scm) {
    Cell ret;
    // TODO: add test case
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

TEST_CASE("testing 4.2.6, Quasiquotation") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("`(list ,(+ 1 2) 4)");
    auto l = "list"_sym;
    CHECK(ret == list(&l, 3, 4));

    ret = scm.eval("(let ((name 'a)) `(list ,name ',name))");
    auto a = "a"_sym;
    auto sym_quote = "quote"_sym;
    CHECK(ret == list(&l, &a, list(&sym_quote, &a)));

    ret = scm.eval("`(a ,(+ 1 2) ,@(map abs '(4 -5 6)) b)");
    auto b = "b"_sym;
    CHECK(ret == list(&a, 3, 4, 5, 6, &b));

    auto foo = "foo"_sym;
    auto sym_cons = "cons"_sym;
    ret = scm.eval("`(,(car '(cons)))");
    CHECK(ret == list(&sym_cons));

    ret = scm.eval("`(,@(cdr '(c)))");
    CHECK(ret == nil);

    ret = scm.eval("`((foo ,(- 10 3)) ,@(cdr '(c)) . ,(car '(cons)))");
    // ((foo 7) . cons)
    CHECK(ret == Cell(cons(list(&foo, 7), &sym_cons)));

    ret = scm.eval("(cons 10 (cons 5 (cons (sqrt 4) (append (map sqrt (quote (16 9))) (quote (8))))))");
    CHECK(ret == list(10, 5, 2, 4, 3, 8));

    Cell::Vec vec;
    vec.push_back(new Number(10));
    vec.push_back(new Number(5));
    vec.push_back(new Number(2));
    vec.push_back(new Number(4));
    vec.push_back(new Number(3));
    vec.push_back(new Number(8));
    ret = scm.eval("`#(10 5 ,(sqrt 4) ,@(map sqrt '(16 9)) 8)");
    CHECK(ret == Cell(&vec));
    Cell expected_expr;
#ifndef WASM_PLATFORM
    // FIXME: bus error
    ret = scm.eval("`(a `(b ,(+ 1 2) ,(foo ,(+ 1 3) d) e) f)");
    expected_expr = Parser("(a `(b ,(+ 1 2) ,(foo 4 d) e) f)").parse();
    CHECK(ret == expected_expr);

    ret = scm.eval(R"(
    (let ((name1 'x)
          (name2 'y))
      `(a `(b ,,name1 ,',name2 d) e))
    )");
    expected_expr = Parser("(a `(b ,x ,'y d) e)").parse();
    CHECK(ret == expected_expr);
#endif
    ret = scm.eval("(quasiquote (list (unquote (+ 1 2)) 4))");
    CHECK(ret == list(new Symbol("list"), 3, 4));

    ret = scm.eval("'(quasiquote (list (unquote (+ 1 2)) 4))");
    expected_expr = Parser("`(list ,(+ 1 2) 4)").parse();
    CHECK(ret == expected_expr);
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

TEST_CASE("testing 5.2.1, Top level definitions") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(define add3
  (lambda (x) (+ x 3)))
)");
    CHECK(ret == Cell::none());
    ret = scm.eval("(add3 3)");
    CHECK(ret == 6);
    ret = scm.eval("(define first car)");
    CHECK(ret == Cell::none());
    ret = scm.eval("(first '(1 2))");
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

TEST_CASE("testing 5.2.2, Internal definitions") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(let ((x 5))
  (define foo (lambda (y) (bar x y)))
  (define bar (lambda (a b) (+ (* a b) a)))
  (foo (+ x 3)))
)");
    CHECK(ret == 45);
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

TEST_CASE("testing 6.1, Booleans, constants") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("#t");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("#f");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("'#f");
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

TEST_CASE("testing 6.1, Booleans, not") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(not #t)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(not 3)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(not (list 3))");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(not #f)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(not '())");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(not (list))");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(not 'nil)");
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

TEST_CASE("testing 6.1, Booleans, boolean?") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(boolean? #f)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(boolean? 0)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(boolean? '())");
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

TEST_CASE("testing 6.2, Equivalence predicates, eqv?") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(eqv? 'a 'a)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(eqv? 'a 'b)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(eqv? 2 2)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(eqv? '() '())");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(eqv? 100000000 100000000)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(eqv? (cons 1 2) (cons 1 2))");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval(R"(
(eqv? (lambda () 1)
      (lambda () 2))
)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(eqv? #f 'nil)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval(R"(
(let ((p (lambda (x) x)))
  (eqv? p p))
)");
    CHECK(ret == Cell::bool_true());

    ret = scm.eval(R"(
(eqv? "" "")
)");
    // the following 4 case return unspecified
    // we follow the guile output #t or #f
    // CHECK(ret == Cell::none());
    // guile -> #t
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(eqv? '#() '#())");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval(R"(
(eqv? (lambda (x) x)
      (lambda (x) x))
)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval(R"(
(eqv? (lambda (x) x)
      (lambda (y) y))
)");
    CHECK(ret == Cell::bool_false());

    ret = scm.eval(R"(
(define gen-counter
  (lambda ()
    (let ((n 0))
      (lambda () (set! n (+ n 1)) n))))
)");
    ret = scm.eval(R"(
(let ((g (gen-counter)))
  (eqv? g g))
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(eqv? (gen-counter) (gen-counter))");
    CHECK(ret == Cell::bool_false());

    ret = scm.eval(R"(
(define gen-loser
  (lambda ()
    (let ((n 0))
      (lambda () (set! n (+ n 1)) 27))))
)");
    ret = scm.eval(R"(
(let ((g (gen-loser)))
  (eqv? g g))
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(eqv? (gen-loser) (gen-loser))");
    // r4rs -> unspecified
    // guile -> #f
    CHECK(ret == Cell::bool_false());
    // TODO: letrec
    ret = scm.eval(R"(
(letrec ((f (lambda () (if (eqv? f g) ’both ’f)))
	 (g (lambda () (if (eqv? f g) ’both ’g))))
  (eqv? f g))
)");
    // r4rs -> unspecified
    // guile -> #f
    CHECK(ret == Cell::bool_false());
    ret = scm.eval(R"(
(letrec ((f (lambda () (if (eqv? f g) ’f ’both)))
	 (g (lambda () (if (eqv? f g) ’g ’both))))
  (eqv? f g))
)");
    CHECK(ret == Cell::bool_false());
    // the following three case return unspecified
    // while guile return #f
    ret = scm.eval("(eqv? '(a) '(a))");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval(R"(
(eqv? "a" "a")
)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(eqv? '(b) (cdr '(a b)))");
    CHECK(ret == Cell::bool_false());

    ret = scm.eval(R"(
(let ((x '(a)))
  (eqv? x x))
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

TEST_CASE("testing 6.2, Equivalence predicates, eq?") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(eq? 'a 'a)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(eq? '(a) '(a))");
    // r4rs -> unspecified
    // guile -> #f
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(eq? (list 'a) (list 'a))");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval(R"(
(eq? "a" "a")
)");
    // r4rs -> unspecified
    // guile -> #f
    CHECK(ret == Cell::bool_false());
    ret = scm.eval(R"(
(eq? "" "")
)");
    // r4rs -> unspecified
    // guile -> #t
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(eq? '() '())");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(eq? 2 2)");
    // r4rs -> unspecified
    // guile -> #t
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(eq? #\A #\A)
)");
    // r4rs -> unspecified
    // guile -> #t
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(eq? car car)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(let ((n (+ 2 3)))
  (eq? n n))
)");
    // r4rs -> unspecified
    // guile -> #t
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(let ((x '(a)))
  (eq? x x))
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(let ((x '#()))
  (eq? x x))
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(let ((p (lambda (x) x)))
  (eq? p p))
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

TEST_CASE("testing 6.2, Equivalence predicates, equal?") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(equal? '(a (b) c)
        '(a (b) c))
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(equal? 2 2)
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(equal? (make-vector 5 'a)
        (make-vector 5 'a))
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(equal? (lambda (x) x)
        (lambda (y) y))
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

TEST_CASE("testing 6.3, Pairs and lists") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(define x (list 'a 'b 'c))");
    REQUIRE(ret == Cell::none());
    ret = scm.eval("(define y x)");
    CHECK(ret == Cell::none());
    ret = scm.eval("y");
    auto a = "a"_sym;
    auto b = "b"_sym;
    auto c = "c"_sym;
    CHECK(ret == list(&a, &b, &c));
    ret = scm.eval("(list? y)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(set-cdr! x 4)");
    CHECK(ret == Cell::none());
    ret = scm.eval("x");
    auto num4 = "4"_num;
    CHECK(ret == Cell(cons(&a, &num4)));
    ret = scm.eval("(eqv? x y)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("y");
    CHECK(ret == Cell(cons(&a, &num4)));
    ret = scm.eval("(list? y)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(set-cdr! x x)");
    ret = scm.eval("(list? x)");
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

TEST_CASE("testing 6.3, Pairs and lists, pair?") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(pair? '(a . b))");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(pair? '(a b c))");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(pair? '())");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(pair? #(a b))");
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

TEST_CASE("testing 6.3, Pairs and lists, cons") {
  auto f = [](Scheme& scm) {
    Cell ret;
    auto a = "a"_sym;
    auto b = "b"_sym;
    auto c = "c"_sym;
    auto d = "d"_sym;
    ret = scm.eval("(cons 'a '())");
    CHECK(ret == list(&a));
    ret = scm.eval("(cons '(a) '(b c d))");
    CHECK(ret == list(list(&a), &b, &c, &d));
    ret = scm.eval(R"(
(cons "a" '(b c))
)");
    pscm::String str_a("a");
    CHECK(ret == list(&str_a, &b, &c));
    ret = scm.eval("(cons 'a 3)");
    auto num3 = "3"_num;
    CHECK(ret == Cell(cons(&a, &num3)));
    ret = scm.eval("(cons '(a b) 'c)");
    CHECK(ret == Cell(cons(list(&a, &b), &c)));
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

TEST_CASE("testing 6.3, Pairs and lists, car") {
  auto f = [](Scheme& scm) {
    Cell ret;
    auto a = "a"_sym;
    auto b = "b"_sym;
    auto c = "c"_sym;
    auto d = "d"_sym;
    ret = scm.eval("(car '(a b c))");
    CHECK(ret == Cell(&a));
    ret = scm.eval("(car '((a) b c d))");
    CHECK(ret == list(&a));
    ret = scm.eval("(car '(1 . 2))");
    CHECK(ret == 1);
    // ret = scm.eval("(car '())");
    // CHECK(ret.is_none());
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

TEST_CASE("testing 6.3, Pairs and lists, cdr") {
  auto f = [](Scheme& scm) {
    Cell ret;
    auto a = "a"_sym;
    auto b = "b"_sym;
    auto c = "c"_sym;
    auto d = "d"_sym;
    ret = scm.eval("(cdr '((a) b c d))");
    CHECK(ret == list(&b, &c, &d));
    ret = scm.eval("(cdr '(1 . 2))");
    CHECK(ret == 2);
    // ret = scm.eval("(car '())");
    // CHECK(ret.is_none());
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

TEST_CASE("testing 6.3, Pairs and lists, set-car!") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(define (f) (list 'not-a-constant-list))");
    CHECK(ret == Cell::none());
    ret = scm.eval("(define (g) '(constant-list))");
    CHECK(ret == Cell::none());
    ret = scm.eval("(set-car! (f) 3)");
    CHECK(ret == Cell::none());
    ret = scm.eval("(f)");
    auto sym1 = "not-a-constant-list"_sym;
    CHECK(ret == list(&sym1));
    ret = scm.eval("(set-car! (g) 3)");
    CHECK(ret == Cell::none());
    // r4rs  -> error
    // guile -> (3)
    // why???
    // ret = scm.eval("(g)");
    // CHECK(ret == list(3));
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

TEST_CASE("testing 6.3, Pairs and lists, list?") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(list? '(a b c))");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(list? '())");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(list? '(a . b))");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval(R"(
(let ((x (list 'a)))
  (set-cdr! x x)
  (list? x))
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

TEST_CASE("testing 6.3, Pairs and lists, list") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(list 'a (+ 3 4) 'c)");
    auto a = "a"_sym;
    auto c = "c"_sym;
    CHECK(ret == list(&a, 7, &c));
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

TEST_CASE("testing 6.3, Pairs and lists, length") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(length '(a b c))");
    CHECK(ret == 3);
    ret = scm.eval("(length '(a (b) (c d e)))");
    CHECK(ret == 3);
    ret = scm.eval("(length '())");
    CHECK(ret == 0);
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

TEST_CASE("testing 6.3, Pairs and lists") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(define x (list 'a 'b 'c))");
    REQUIRE(ret == Cell::none());
    ret = scm.eval("(define y x)");
    REQUIRE(ret == Cell::none());
    ret = scm.eval("y");
    auto a = "a"_sym;
    auto b = "b"_sym;
    auto c = "c"_sym;
    CHECK(ret == list(&a, &b, &c));
    ret = scm.eval("(list? y)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(set-cdr! x 4)");
    REQUIRE(ret == Cell::none());
    auto num4 = "4"_num;
    ret = scm.eval("x");
    CHECK(ret == Cell(cons(&a, &num4)));
    ret = scm.eval("(eqv? x y)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("y");
    CHECK(ret == cons(&a, &num4));
    ret = scm.eval("(list? y)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(set-cdr! x x)");
    REQUIRE(ret == Cell::none());
    ret = scm.eval("(list? x)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(assv 5 '((2 3) (5 7) (11 13)))");
    CHECK(ret == list(5, 7));
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

TEST_CASE("testing 6.3, Pairs and lists, append") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(append '(x) '(y))");
    auto x = "x"_sym;
    auto y = "y"_sym;
    CHECK(ret == list(x, y));
    ret = scm.eval("(append '(a) '(b c d))");
    auto a = "a"_sym;
    auto b = "b"_sym;
    auto c = "c"_sym;
    auto d = "d"_sym;
    CHECK(ret == list(&a, &b, &c, &d));
    ret = scm.eval("(append '(a (b)) '((c)))");
    CHECK(ret == list(&a, list(&b), list(&c)));
    ret = scm.eval("(append '(a b) '(c . d))");
    CHECK(ret == Cell(cons(&a, cons(&b, cons(&c, &d)))));
    ret = scm.eval("(append '() 'a)");
    CHECK(ret == a);
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

TEST_CASE("testing 6.3, Pairs and lists, reverse") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(reverse '(a b c))");
    auto a = "a"_sym;
    auto b = "b"_sym;
    auto c = "c"_sym;
    CHECK(ret == list(&c, &b, &a));
    ret = scm.eval("(reverse '(a (b c) d (e (f))))");
    auto d = "d"_sym;
    auto e = "e"_sym;
    auto f = "f"_sym;
    CHECK(ret == list(list(&e, list(&f)), &d, list(&b, &c), &a));
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

TEST_CASE("testing 6.3, Pairs and lists, list-tail") {
  auto f = [](Scheme& scm) {
    Cell ret;
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

TEST_CASE("testing 6.3, Pairs and lists, list-ref") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(list-ref '(a b c d) 2)");
    auto c = "c"_sym;
    CHECK(ret == c);
    ret = scm.eval(R"(
(list-ref '(a b c d)
          (inexact->exact (round 1.8)))
)");
    CHECK(ret == c);
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

TEST_CASE("testing 6.3, Pairs and lists, memq memv member") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(memq 'a '(a b c))");
    auto a = "a"_sym;
    auto b = "b"_sym;
    auto c = "c"_sym;
    CHECK(ret == list(&a, &b, &c));
    ret = scm.eval("(memq 'b '(a b c))");
    CHECK(ret == list(b, c));
    ret = scm.eval("(memq 'a '(b c d))");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(memq (list 'a) '(b (a) c))");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(member (list 'a) '(b (a) c))");
    CHECK(ret == list(list(a), c));
    ret = scm.eval("(memq 101 '(100 101 102))");
    // r4rs -> unspecified
    // guile -> (101 102)
    CHECK(ret == list(101, 102));
    ret = scm.eval("(memv 101 '(100 101 102))");
    CHECK(ret == list(101, 102));
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

TEST_CASE("testing 6.3, Pairs and lists, assq assv assoc") {
  auto f = [](Scheme& scm) {
    Cell ret;
    auto a = "a"_sym;
    auto b = "b"_sym;
    auto c = "c"_sym;
    ret = scm.eval("(define e '((a 1) (b 2) (c 3)))");
    CHECK(ret == Cell::none());
    ret = scm.eval("(assq 'a e)");
    CHECK(ret == list(&a, 1));
    ret = scm.eval("(assq 'b e)");
    CHECK(ret == list(&b, 2));
    ret = scm.eval("(assq 'd e)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(assq (list 'a) '(((a)) ((b)) ((c))))");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(assoc (list 'a) '(((a)) ((b)) ((c))))");
    CHECK(ret == list(list(&a)));
    ret = scm.eval("(assq 5 '((2 3) (5 7) (11 13)))");
    // r4rs -> unspecified
    // guile -> (5, 7)
    CHECK(ret == list(5, 7));
    ret = scm.eval("(assv 5 '((2 3) (5 7) (11 13)))");
    CHECK(ret == list(5, 7));
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

TEST_CASE("testing 6.4, Symbols, symbol?") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(symbol? 'foo)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(symbol? (car '(a b)))");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(symbol? "bar")
)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(symbol? 'nil)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(symbol? '())");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(symbol? #f)");
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

TEST_CASE("testing 6.4, Symbols, symbol->string") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(symbol->string 'flying-fish)");
    CHECK(ret == "flying-fish"_str);
    ret = scm.eval("(symbol->string 'Martin)");
    // r4rs -> marin
    // guile -> Martin
    CHECK(ret == "Martin"_str);
    ret = scm.eval(R"(
(symbol->string
  (string->symbol "Malvina"))
)");
    CHECK(ret == "Malvina"_str);
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

TEST_CASE("testing 6.4, Symbols, string->symbol") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(eq? 'mISSISSIppi 'mississippi)");
    // r4rs -> #t
    // guile -> #f
    CHECK(ret == Cell::bool_false());
    ret = scm.eval(R"(
(string->symbol "mISSISSIppi")
)");
    CHECK(ret == "mISSISSIppi"_sym);
    ret = scm.eval(R"(
(eq? 'bitBlt (string->symbol "bitBlt"))
)");
    // r4rs -> #f
    // guile -> #t
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(eq? 'JollyWog
     (string->symbol
       (symbol->string 'JollyWog)))
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(string=? "K. Harper, M.D."
          (symbol->string
            (string->symbol "K. Harper, M.D.")))
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

TEST_CASE("testing 6.5.5, Numerical operations, number? complex? real? rational? interger?") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(complex? 3+4i)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(complex? 3)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(real? 3)");
    CHECK(ret == Cell::bool_true());

    ret = scm.eval("(real? -2.5+0.0i)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(real? #e1e10)");
    CHECK(ret == Cell::bool_true());

    ret = scm.eval("(rational? 6/10)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(rational? 6/3)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(integer? 3+0i)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(integer? 3.0)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(integer? 8/4)");
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

TEST_CASE("testing 6.5.5, Numerical operations, max") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(max 3 4)");
    CHECK(ret == 4);
    ret = scm.eval("(max 3.9 4)");
    CHECK(ret == 4.0);
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

TEST_CASE("testing 6.5.5, Numerical operations, expt") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(expt 0 0)");
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

TEST_CASE("testing 6.6, Characters") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(char? #\A)
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(char<? #\A #\B)
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(char<? #\a #\b)
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(char<? #\0 #\9)
)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval(R"(
(char-ci=? #\A #\a)
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

TEST_CASE("testing 6.8, Vectors") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(vector-ref '#(1 1 2 3 5 8 13 21) 5)
)");
    CHECK(ret == 8);
    ret = scm.eval(R"(
(vector-ref '#(1 1 2 3 5 8 13 21)
            (inexact->exact
               (round (* 2 (acos -1)))))
)");
    CHECK(ret == 13);
    ret = scm.eval(R"(
(let ((vec (vector 0 '(2 2 2 2) "Anna")))
  (vector-set! vec 1 '("Sue" "Sue"))
  vec)
)");
    Parser parser(R"(
#(0 ("Sue" "Sue") "Anna")
)");
    auto expr = parser.parse();
    CHECK(ret == expr);
    ret = scm.eval(R"(
(vector-set! '#(0 1 2) 1 "doe")
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

TEST_CASE("testing 6.9, Control features, procedure?") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(procedure? car)");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(procedure? 'car)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(procedure? (lambda (x) (* x x)))");
    CHECK(ret == Cell::bool_true());
    ret = scm.eval("(procedure? '(lambda (x) (* x x)))");
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

TEST_CASE("testing 6.9, Control features, procedure? call/cc") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(call-with-current-continuation procedure?)");
    CHECK(ret == Cell::bool_true());
  };
  {
    Scheme scm(true);
    f(scm);
  }
}

TEST_CASE("testing 6.9, Control features, apply") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(apply + (list 3 4))");
    CHECK(ret == 7);
    ret = scm.eval("(apply cadr '((a b)))");
    CHECK(ret == "b"_sym);
    ret = scm.eval("(apply cadr '((b 2)))");
    CHECK(ret == 2);
    ret = scm.eval(R"(
(define compose
  (lambda (f g)
    (lambda args
      (f (apply g args)))))
)");
    CHECK(ret == Cell::none());
    ret = scm.eval("((compose sqrt *) 12 75)");
    CHECK(ret == 30);
    ret = scm.eval("(apply + '(1 2))");
    CHECK(ret == 3);
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

TEST_CASE("testing 6.9, Control features, map") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(map cadr '((a b) (d e) (g h)))");
    auto b = "b"_sym;
    auto e = "e"_sym;
    auto h = "h"_sym;
    CHECK(ret == list(&b, &e, &h));

    ret = scm.eval(R"(
(map (lambda (n) (expt n n))
     '(1 2 3 4 5))
)");
    CHECK(ret == list(1, 4, 27, 256, 3125));

    //    ret = scm.eval(R"(
    //(map + '(1 2 3) '(4 5 6))
    //)");
    //    CHECK(ret == list(5, 7, 9));

    ret = scm.eval(R"(
(let ((count 0))
(map (lambda (ignored)
       (set! count (+ count 1))
       count)
     '(a b c)))
)");
    CHECK(ret == list(1, 2, 3));
    ret = scm.eval(R"(
((lambda (count)
   (map (lambda (ignored)
	  (set! count (+ count 1))
	  count)
	(quote (a b c))))
0)
)");
    CHECK(ret == list(1, 2, 3));
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

TEST_CASE("testing 6.9, Control features, for-each") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(let ((v (make-vector 5)))
  (for-each (lambda (i)
              (vector-set! v i (* i i)))
              '(0 1 2 3 4))
  v)
)");
    auto vec = new Cell::Vec();
    vec->push_back(new Number(0));
    vec->push_back(new Number(1));
    vec->push_back(new Number(4));
    vec->push_back(new Number(9));
    vec->push_back(new Number(16));
    CHECK(ret == Cell(vec));
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

TEST_CASE("testing 6.9, Control features, force") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval("(force (delay (+ 1 2)))");
    CHECK(ret == 3);
    // FIXME:
    // r4rs -> 2
    // guile -> Unbound variable: stream
    //     ret = scm.eval(R"(
    // (define a-stream
    //   (letrec ((next
    //             (lambda (n)
    //               (cons n (delay (next (+ n 1)))))))
    //     (next 0)))
    // )");
    //     REQUIRE(ret == Cell::none());
    //     ret = scm.eval("(define head car)");
    //     REQUIRE(ret == Cell::none());
    //     ret = scm.eval(R"(
    // (define tail
    //   (lambda (sream) (force (cdr stream))))
    // )");
    //     REQUIRE(ret == Cell::none());
    //     ret = scm.eval("(head (tail (tail a-stream)))");
    //     CHECK(ret == 2);

    ret = scm.eval("(define count 0)");
    REQUIRE(ret == Cell::none());
    ret = scm.eval(R"(
(define p
  (delay (begin (set! count (+ count 1))
                (if (> count x)
                    count
                    (force p)))))
)");
    REQUIRE(ret == Cell::none());
    ret = scm.eval("(define x 5)");
    REQUIRE(ret == Cell::none());
    ret = scm.eval("p");
    CHECK(ret.is_promise());
    ret = scm.eval("(force p)");
    CHECK(ret == 6);
    ret = scm.eval("p");
    CHECK(ret.is_promise());
    ret = scm.eval(R"(
(begin (set! x 10)
       (force p))
)");
    CHECK(ret == 6);

    ret = scm.eval("(eqv? (delay 1) 1)");
    CHECK(ret == Cell::bool_false());
    ret = scm.eval("(pair? (delay (cons 1 2)))");
    CHECK(ret == Cell::bool_false());

    // implicit forcing
    // r4rs -> 34
    // guile -> Wrong type argument in position 1: #<promise #<procedure #f ()>>
    // ret = scm.eval("(+ (delay (* 3 7)) 13)");
    // CHECK(ret == 34);
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

TEST_CASE("testing 6.9, Control features, call-with-continuation") {
  auto f = [](Scheme& scm) {
    Cell ret;
    ret = scm.eval(R"(
(call/cc
 (lambda (exit)
   (for-each (lambda (x)
	       (if (negative? x) (exit x)))
	     '(54 0 37 -3 245 19))
   #t))
)");
    CHECK(ret == "-3"_num);
    ret = scm.eval(R"(
(define list-length
  (lambda (obj)
    (call/cc
      (lambda (return)
        (letrec ((r
                  (lambda (obj)
                    (cond ((null? obj) 0)
                          ((pair? obj)
                           (+ (r (cdr obj)) 1))
                          (else (return #f))))))
                (r obj))))))
)");
    REQUIRE(ret == Cell::none());
    ret = scm.eval("(list-length '(1 2 3 4))");
    CHECK(ret == 4);
    ret = scm.eval("(list-length '(a b . c))");
    CHECK(ret == Cell::bool_false());
  };
  {
    Scheme scm(true);
    f(scm);
  }
}

TEST_CASE("testing ") {
}

int main(int argc, char **argv) {
  doctest::Context context;

  int res = context.run(); // run

  if (context.shouldExit()) // important - query flags (and --exit) rely on the user doing this
    return res;             // propagate the result of the tests

  int client_stuff_return_code = 0;
  // your program - if the testing framework is integrated in your production code

  return res + client_stuff_return_code; // the result from doctest is propagated here as well
}