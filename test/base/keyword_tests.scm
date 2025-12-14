;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: #:use
:use

;; CHECK: #t
(eqv? :check-mark :check-mark)
