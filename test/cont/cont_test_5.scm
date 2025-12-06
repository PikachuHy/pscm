;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_cc --test %s | FileCheck %s --check-prefix=LONGJMP
;; RUN: %pscm_cc --test %s | FileCheck %s
;; TODO: %pscm_main --test %s | FileCheck %s

;; LONGJMP-NOT: not supported

(define foo #f)
;; CHECK: 123
(call/cc (lambda (bar) (set! foo bar) 123))
;; CHECK: 456
(foo 456)
;; CHECK: abc
(foo 'abc)
