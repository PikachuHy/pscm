;; RUN: %pscm_cc --test %s | FileCheck %s --check-prefix=LONGJMP
;; RUN: %pscm_cc --test %s | FileCheck %s

;; LONGJMP-NOT: not supported

(define foo #f)
;; CHECK: 123
(call/cc (lambda (bar) (set! foo bar) 123))
;; CHECK: 456
(foo 456)
;; CHECK: abc
(foo 'abc)
