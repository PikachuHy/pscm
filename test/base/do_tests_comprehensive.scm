;; RUN: %pscm_cc --test %s | FileCheck %s
;; Comprehensive tests for do loop implementation

;; Test 1: Basic do loop with single variable
;; CHECK: 10
(do ((i 0 (+ i 1)))
  ((= i 10) i))

;; Test 2: Do loop with no body
;; CHECK: 5
(do ((i 0 (+ i 1)))
  ((= i 5) i))

;; Test 3: Do loop with multiple variables, no dependencies
;; CHECK: 15
(do ((i 0 (+ i 1))
     (j 0 (+ j 2)))
  ((= i 5) (+ i j)))

;; Test 4: Do loop with variables that depend on each other (parallel binding)
;; CHECK: 25
(let ((x '(1 3 5 7 9)))
  (do ((x x (cdr x))
       (sum 0 (+ sum (car x))))
    ((null? x) sum)))

;; Test 5: Do loop with no update step (variable stays constant)
;; CHECK: 42
(do ((x 42))
  ((= x 42) x))

;; Test 6: Do loop with mixed update steps
;; CHECK: 45
(do ((i 0 (+ i 1))
     (sum 0 (+ sum i))
     (count 0))
  ((= i 10) sum))

;; Test 7: Do loop with body that modifies variables
;; CHECK: 100
(do ((i 0))
  ((= i 100) i)
  (set! i (+ i 1)))

;; Test 8: Do loop with multiple return expressions (should return last)
;; CHECK: 20
(do ((i 0 (+ i 1)))
  ((= i 10) i (* i 2) (+ i 10)))

;; Test 9: Do loop with empty variable list
;; CHECK: done
(do ()
  (#t "done"))

;; Test 10: Do loop with complex dependencies
;; CHECK: 55
(do ((i 1 (+ i 1))
     (sum 0 (+ sum i)))
  ((> i 10) sum))

;; Test 11: Do loop with vector operations
;; CHECK: #(0 1 4 9 16)
(do ((vec (make-vector 5))
     (i 0 (+ i 1)))
  ((= i 5) vec)
  (vector-set! vec i (* i i)))

;; Test 12: Do loop with string operations
;; CHECK: hello
(do ((str (make-string 5))
     (i 0 (+ i 1))
     (chars '(#\h #\e #\l #\l #\o)))
  ((= i 5) str)
  (string-set! str i (car chars))
  (set! chars (cdr chars)))

;; Test 13: Nested do loops
;; CHECK: 45
(do ((i 0 (+ i 1))
     (sum 0))
  ((= i 10) sum)
  (do ((j 0 (+ j 1)))
    ((= j i))
    (set! sum (+ sum 1))))

;; Test 14: Do loop that never executes (test is true initially)
;; CHECK: 0
(do ((x 0))
  (#t x))

;; Test 15: Do loop with variable shadowing
;; CHECK: 10
(let ((x 5))
  (do ((x 0 (+ x 1)))
    ((= x 10) x)))

;; Test 16: Do loop with multiple variables, all with update steps
;; CHECK: 30
(do ((a 0 (+ a 1))
     (b 0 (+ b 2))
     (c 0 (+ c 3)))
  ((= a 5) (+ a b c)))

;; Test 17: Do loop with circular dependency (should work with parallel binding)
;; TODO: dead loop here
;; (do ((x 1 y)
;;      (y 2 x))
;;   ((= x 2) (+ x y))
;;   (set! x 3)
;;   (set! y 3))

;; Test 18: Do loop with list traversal and accumulation
;; CHECK: 15
(do ((lst '(1 2 3 4 5) (cdr lst))
     (sum 0 (+ sum (car lst))))
  ((null? lst) sum))

;; Test 19: Do loop with early exit via set!
;; TOOD: dead loop here
;; (do ((i 0 (+ i 1)))
;;   ((= i 10) i)
;;   (if (= i 5)
;;       (set! i 10)))

;; Test 20: Do loop with no return expressions
(do ((i 0 (+ i 1)))
  ((= i 3)))

