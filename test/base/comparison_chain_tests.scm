;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test comparison operators with multiple arguments (chain comparisons)

;; Test = (equality) with multiple arguments
;; CHECK: #t
(= 1 1)
;; CHECK-NEXT: #t
(= 1 1 1)
;; CHECK-NEXT: #f
(= 1 1 2)
;; CHECK-NEXT: #f
(= 1 2 1)
;; CHECK-NEXT: #f
(= 2 1 1)
;; CHECK-NEXT: #t
(= 5 5 5 5)
;; CHECK-NEXT: #f
(= 5 5 5 6)

;; Test < (less than) with multiple arguments
;; CHECK-NEXT: #t
(< 1 2)
;; CHECK-NEXT: #t
(< 1 2 3)
;; CHECK-NEXT: #t
(< 1 2 3 4)
;; CHECK-NEXT: #f
(< 1 3 2)
;; CHECK-NEXT: #f
(< 3 2 1)
;; CHECK-NEXT: #f
(< 1 1 2)
;; CHECK-NEXT: #f
(< 1 2 2)
;; CHECK-NEXT: #f
(< -1 2 3 4 4 5 6 7)

;; Test > (greater than) with multiple arguments
;; CHECK-NEXT: #t
(> 3 2)
;; CHECK-NEXT: #t
(> 3 2 1)
;; CHECK-NEXT: #t
(> 5 4 3 2)
;; CHECK-NEXT: #f
(> 3 2 3)
;; CHECK-NEXT: #f
(> 1 2 3)
;; CHECK-NEXT: #f
(> 3 3 2)
;; CHECK-NEXT: #f
(> 3 2 2)

;; Test <= (less than or equal) with multiple arguments
;; CHECK-NEXT: #t
(<= 1 2)
;; CHECK-NEXT: #t
(<= 1 1 2)
;; CHECK-NEXT: #t
(<= 1 2 3)
;; CHECK-NEXT: #t
(<= 1 1 1)
;; CHECK-NEXT: #f
(<= 1 2 1)
;; CHECK-NEXT: #f
(<= 2 1 1)

;; Test >= (greater than or equal) with multiple arguments
;; CHECK-NEXT: #t
(>= 3 2)
;; CHECK-NEXT: #t
(>= 3 2 1)
;; CHECK-NEXT: #t
(>= 3 3 2)
;; CHECK-NEXT: #t
(>= 3 3 3)
;; CHECK-NEXT: #f
(>= 3 2 3)
;; CHECK-NEXT: #f
(>= 1 2 3)

;; Test with zero arguments (should return #t)
;; CHECK-NEXT: #t
(=)
;; CHECK-NEXT: #t
(<)
;; CHECK-NEXT: #t
(>)
;; CHECK-NEXT: #t
(<=)
;; CHECK-NEXT: #t
(>=)

;; Test with one argument (should return #t)
;; CHECK-NEXT: #t
(= 5)
;; CHECK-NEXT: #t
(< 5)
;; CHECK-NEXT: #t
(> 5)
;; CHECK-NEXT: #t
(<= 5)
;; CHECK-NEXT: #t
(>= 5)

;; Test with floats only
;; CHECK-NEXT: #t
(= 1.0 1.0 1.0)
;; CHECK-NEXT: #t
(< 1.0 2.0 3.0)
;; CHECK-NEXT: #t
(> 3.0 2.0 1.0)
;; CHECK-NEXT: #f
(= 1.0 1.0 2.0)

;; Test with negative numbers
;; CHECK-NEXT: #t
(< -5 -3 -1 0 1 3 5)
;; CHECK-NEXT: #t
(> 5 3 1 0 -1 -3 -5)
;; CHECK-NEXT: #f
(< -1 0 1 -2)

;; Test edge cases
;; CHECK-NEXT: #t
(= 0 0 0)
;; CHECK-NEXT: #f
(< 0 0 0)
;; CHECK-NEXT: #f
(> 0 0 0)
;; CHECK-NEXT: #t
(<= 0 0 0)
;; CHECK-NEXT: #t
(>= 0 0 0)

