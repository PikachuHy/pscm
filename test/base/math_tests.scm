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

;; Test max function
;; CHECK: 5
(max 3 5 2)
;; CHECK: 10
(max 1 2 3 4 5 6 7 8 9 10)
;; CHECK: -1
(max -5 -3 -1 -10)
;; CHECK: 3
(max 3)
;; CHECK: 5.5
(max 3.5 5.5 2.0)
;; CHECK: 5
(max 3 5.0 2)
;; CHECK: 5.0
(max 3.0 5 2.0)

;; Test min function
;; CHECK: 2
(min 3 5 2)
;; CHECK: 1
(min 1 2 3 4 5 6 7 8 9 10)
;; CHECK: -10
(min -5 -3 -1 -10)
;; CHECK: 3
(min 3)
;; CHECK: 2.0
(min 3.5 5.5 2.0)
;; CHECK: 2
(min 3 5.0 2)
;; CHECK: 2.0
(min 3.0 5 2.0)
