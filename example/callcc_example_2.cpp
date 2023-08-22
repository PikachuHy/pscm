//
// Created by PikachuHy on 2023/3/20.
//
#include <chrono>
#include <pscm/Scheme.h>

using namespace pscm;
using namespace std::chrono_literals;

void version_1() {
  Scheme scm(true);
  scm.eval(R"(
  (define (f1 cc)
    (display #\@)
    cc)
  )");
  scm.eval(R"(
  (define (f2 cc)
    (display #\*)
    cc)
  )");
  scm.eval(R"(
  (define (identity x) x)
  )");
  scm.eval(R"(
  ((f1 (call/cc identity)) (f2 (call/cc identity)))
  )");
}

void version_2() {
  Scheme scm(true);
  scm.eval(R"(
  (define (f1 cc)
    (display #\@)
    cc)
  )");
  scm.eval(R"(
  (define (f2 cc)
    (display #\*)
    cc)
  )");
  scm.eval(R"(
  (define (identity x) x)
  )");
  scm.eval(R"(
(let ((yin (f1 (call/cc identity))))
   (let ((yang (f2 (call/cc identity))))
(yin yang)))
  )");
}

void version_3() {
  Scheme scm(true);
  scm.eval(R"(
  (define (f1 cc)
    (display #\@)
    cc)
  )");
  scm.eval(R"(
  (define (f2 cc)
    (display #\*)
    cc)
  )");
  scm.eval(R"(
  (define (identity x) x)
  )");
  scm.eval(R"(
(let* ((yin (f1 (call/cc identity)))
       (yang (f2 (call/cc identity))))
(yin yang))
  )");
}

void version_4() {
  Scheme scm(true);
  scm.eval(R"(
(let* ((yin ((lambda (cc) (display #\@) cc) (call/cc (lambda (c) c))))
       (yang ((lambda (cc) (display #\*) cc) (call/cc (lambda (c) c)))))
  (yin yang))
  )");
}

int main() {
  version_4();
  return 0;
}