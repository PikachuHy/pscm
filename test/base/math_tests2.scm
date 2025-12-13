;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test division with zero arguments
;; CHECK: 1
(/)
;; CHECK: 1
(apply / '())
;; CHECK: 1/2
(/ 2)
;; CHECK: 2
(/ 4 2)
;; CHECK: 1/2
(/ 1 2)
;; CHECK: 1/6
(/ 1 2 3)
