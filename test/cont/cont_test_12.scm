;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_cc --test %s | FileCheck %s --check-prefix=LONGJMP
;; RUN: %pscm_cc --test %s | FileCheck %s
;; TODO: %pscm_main --test %s | FileCheck %s

;; LONGJMP-NOT: not supported

(define cc #f)
;; CHECK: 9
(* 3 (call/cc (lambda (k)
                 (set! cc k)
                 (+ 1 2))))
;; CHECK: 9
(+ 100 (cc 3))
;; CHECK: 30 
(+ 100 (cc 10))
