;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; TODO: %pscm_main --test %s | FileCheck %s


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
