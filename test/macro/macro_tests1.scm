;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test basic define-macro - simple macro that returns a constant
(define-macro (return-three) 3)
;; CHECK: #<macro! return-three>
return-three
;; CHECK: 3
(return-three)

;; Test define-macro with one argument - returns the argument
(define-macro (identity x) x)
;; CHECK: #<macro! identity>
identity
;; CHECK: 42
(identity 42)

