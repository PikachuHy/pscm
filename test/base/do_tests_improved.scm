;; RUN: %pscm_cc --test %s | FileCheck %s
;; Improved tests for do loop implementation

;; Test 1: Basic do loop
;; CHECK: 10
(do ((i 0 (+ i 1)))
  ((= i 10) i))

;; Test 2: Do loop with no body
;; CHECK: 5
(do ((i 0 (+ i 1)))
  ((= i 5) i))

;; Test 3: Multiple variables without dependencies
;; CHECK: 15
(do ((i 0 (+ i 1))
     (j 0 (+ j 2)))
  ((= i 5) (+ i j)))

;; Test 4: Variables with dependencies (parallel binding - critical test)
;; CHECK: 25
(let ((x '(1 3 5 7 9)))
  (do ((x x (cdr x))
       (sum 0 (+ sum (car x))))
    ((null? x) sum)))

;; Test 5: Variable without update step
;; CHECK: 42
(do ((x 42))
  ((= x 42) x))

;; Test 6: Do loop with body
;; CHECK: 10
(do ((i 0 (+ i 1)))
  ((= i 10) i)
  #t)

;; Test 7: Multiple return expressions (should return last)
;; CHECK: 20
(do ((i 0 (+ i 1)))
  ((= i 10) i (* i 2)))

;; Test 8: Empty variable list
;; CHECK: done
(do ()
  (#t "done"))

;; Test 9: List traversal with accumulation
;; CHECK: 15
(do ((lst '(1 2 3 4 5) (cdr lst))
     (sum 0 (+ sum (car lst))))
  ((null? lst) sum))

;; Test 10: Vector operations
;; CHECK: #(0 1 4 9 16)
(do ((vec (make-vector 5))
     (i 0 (+ i 1)))
  ((= i 5) vec)
  (vector-set! vec i (* i i)))

