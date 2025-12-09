;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s
;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: -1
(expt -1 -255)
;; CHECK: #t
(number? 3)
;; CHECK: 1/3
(/ 1 3)
;; CHECK: 100/3
(* (/ 1 3) 100)
