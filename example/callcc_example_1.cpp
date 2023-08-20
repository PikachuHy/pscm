//
// Created by PikachuHy on 2023/3/20.
//
#include <pscm/Scheme.h>
using namespace pscm;

void version_1() {
  Scheme scm(true);
  scm.eval("(define n 0)");
  scm.eval("(define bar (lambda (bar) bar))");
  scm.eval(R"(
(define foo (lambda (foo)
              (display n)
              (newline)
              (set! n (+ n 1))
              foo))
)");
  scm.eval("((call/cc bar) (foo (call/cc bar)))");
}

void version_2() {
  Scheme scm(true);
  scm.eval(R"(
(let* ((n 0)
      (bar (lambda (bar) bar))
      (foo (lambda (foo)
	     (display n)
	     (newline)
	     (set! n (+ n 1))
	     foo))
      (f1 (call/cc bar))
      (f2 (foo (call/cc bar))))
  (f1 f2))
)");
}

int main() {
  version_2();
  return 0;
}