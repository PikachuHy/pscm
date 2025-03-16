;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s

;; CHECK: #:use
:use

;; CHECK: #t
(eqv? :check-mark :check-mark)
