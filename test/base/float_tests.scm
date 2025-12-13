;; RUN: %pscm_cc --test %s | FileCheck %s

;; Float number tests

;; Test 1: Basic float parsing and printing
;; CHECK: 12.7
12.7
;; CHECK: -3.14
-3.14
;; CHECK: 0.5
0.5
;; CHECK: 1.0
1.0

;; Test 2: Scientific notation
;; CHECK: 1200.0
1.2e3
;; CHECK: 0.0012
1.2e-3
;; CHECK: -500.0
-5.0e2

;; Test 3: Float arithmetic
;; CHECK: 4.0
(+ 1.5 2.5)
;; CHECK: 2.5
(- 5.0 2.5)
;; CHECK: 7.0
(* 2.0 3.5)
;; CHECK: 3.5
(+ 1 2.5)  ; Integer + Float -> Float
;; CHECK: 3.5
(+ 2.5 1)  ; Float + Integer -> Float

;; Test 4: Unary minus
;; CHECK: -3.14
(- 3.14)
;; CHECK: 2.5
(- -2.5)

;; Test 5: Comparisons
;; CHECK: #t
(= 1.0 1)
;; CHECK: #t
(= 2.5 2.5)
;; CHECK: #f
(= 1.5 2.5)
;; CHECK: #t
(< 1.5 2.5)
;; CHECK: #t
(> 3.5 2.5)
;; CHECK: #t
(<= 1.0 1.0)
;; CHECK: #t
(>= 2.0 1.5)

;; Test 6: Mixed operations
;; CHECK: 6.5
(+ 1 2.5 3)
;; CHECK: 6.5
(- 10.0 2 1.5)
;; CHECK: 9.0
(* 2 1.5 3)

