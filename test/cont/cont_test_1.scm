;; RUN: %pscm_cc --test %s | FileCheck %s --check-prefix=LONGJMP
;; RUN: %pscm_cc --test %s | FileCheck %s

;; LONGJMP-NOT: not supported

(define (f return)
    (return 2)
    3)

;; CHECK: 2
(call/cc f)
;; CHECK-NEXT: 2
(call/cc f)
;; CHECK-NEXT: 2
(call/cc f)
;; CHECK-NEXT: 2
(call/cc f)

;; CHECK: 2
(define (f a) 2)
(call/cc f)
