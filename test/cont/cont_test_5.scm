;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; TODO: %pscm_main --test %s | FileCheck %s

(define foo #f)
;; CHECK: 123
(call/cc (lambda (bar) (set! foo bar) 123))
;; CHECK: 456
(foo 456)
;; CHECK: abc
(foo 'abc)
