;; RUN: %pscm_cc --test %s | FileCheck %s
;; Additional tests for let, let*, letrec, and named let
;; Focus on variable shadowing and edge cases

;; Test 1: Named let with variable shadowing (the key fix)
;; This was the bug: (let ((f -)) (let f ((n (f 1))) n)) should return -1
;; CHECK: -1
(let ((f -)) (let f ((n (f 1))) n))

;; Test 2: Named let without shadowing
;; CHECK: 15
(let loop ((n 5) (acc 0))
  (if (zero? n)
      acc
      (loop (- n 1) (+ acc n))))

;; Test 3: Named let with empty bindings
;; CHECK: 42
(let loop () 42)

;; Test 4: Named let with multiple parameters
;; CHECK: 10
(let sum ((a 3) (b 7))
  (+ a b))

;; Test 5: Named let that calls itself recursively
;; CHECK: 120
(let fact ((n 5) (acc 1))
  (if (<= n 1)
      acc
      (fact (- n 1) (* acc n))))

;; Test 6: Named let with outer variable reference (no shadowing)
;; CHECK: 8
(let ((x 2))
  (let mul ((n 4))
    (* x n)))

;; Test 7: Named let with shadowing in initialization expression
;; CHECK: -5
(let ((f -))
  (let f ((a (f 3)) (b (f 2)))
    (+ a b)))

;; Test 8: Nested named let
;; CHECK: 30
(let outer ((x 5))
  (let inner ((y 6))
    (* x y)))

;; Test 9: Named let with same name as outer binding (shadowing)
;; CHECK: 100
(let ((counter 0))
  (let counter ((n 10))
    (if (zero? n)
        100
        (counter (- n 1)))))

;; Test 10: let with empty bindings
;; CHECK: 7
(let () 7)

;; Test 11: let* with empty bindings
;; CHECK: 8
(let* () 8)

;; Test 12: letrec with empty bindings
;; CHECK: 9
(letrec () 9)

;; Test 13: let with parallel bindings (should use outer values)
;; CHECK: 3
(let ((x 1) (y 2))
  (let ((x 3) (z (+ x y)))
    z))

;; Test 14: let* with sequential bindings
;; CHECK: 5
(let ((x 1) (y 2))
  (let* ((x 3) (z (+ x y)))
    z))

;; Test 15: letrec with mutual recursion
;; CHECK: #t
(letrec ((even?
          (lambda (n) (if (zero? n) #t (odd? (- n 1)))))
         (odd?
          (lambda (n) (if (zero? n) #f (even? (- n 1))))))
  (even? 4))

;; Test 16: letrec with mutual recursion (odd case)
;; CHECK: #f
(letrec ((even?
          (lambda (n) (if (zero? n) #t (odd? (- n 1)))))
         (odd?
          (lambda (n) (if (zero? n) #f (even? (- n 1))))))
  (even? 5))

;; Test 17: Named let with complex initialization
;; CHECK: 6
(let ((add +))
  (let compute ((a (+ 1 2)) (b (+ 2 1)))
    (+ a b)))

;; Test 18: Named let that doesn't recurse
;; CHECK: 10
(let once ((x 5) (y 5))
  (+ x y))

;; Test 19: Multiple nested lets with shadowing
;; CHECK: 24
(let ((x 1))
  (let ((x 2))
    (let ((x 3))
      (let ((x 4))
        (* x 6)))))

;; Test 20: let* with dependencies
;; CHECK: 15
(let* ((a 1)
       (b (+ a 2))
       (c (+ b 3))
       (d (+ c 4))
       (e (+ d 5)))
  e)

;; Test 21: letrec with self-reference
;; CHECK: 55
(letrec ((sum (lambda (n)
                (if (zero? n)
                    0
                    (+ n (sum (- n 1)))))))
  (sum 10))

;; Test 22: Named let with outer variable in body
;; CHECK: 50
(let ((multiplier 5))
  (let compute ((x 10))
    (* multiplier x)))

;; Test 23: Named let with shadowing in nested context
;; CHECK: -3
(let ((op -))
  (let ((f op))
    (let f ((n (f 3)))
      n)))

;; Test 24: let with all bindings using outer scope
;; CHECK: 6
(let ((a 1) (b 2) (c 3))
  (let ((x a) (y b) (z c))
    (+ x y z)))

;; Test 25: let* with forward references
;; CHECK: 10
(let* ((a 1)
       (b (+ a 1))
       (c (+ b 1))
       (d (+ c 1))
       (e (+ d 1))
       (f (+ e 1))
       (g (+ f 1))
       (h (+ g 1))
       (i (+ h 1))
       (j (+ i 1)))
  j)

;; Test 26: Named let with list processing
;; CHECK: 15
(let sum-list ((lst '(1 2 3 4 5)) (acc 0))
  (if (null? lst)
      acc
      (sum-list (cdr lst) (+ acc (car lst)))))

;; Test 27: Named let with early return
;; CHECK: 3
(let find ((lst '(1 2 3 4 5)) (target 3))
  (cond ((null? lst) #f)
        ((= (car lst) target) target)
        (else (find (cdr lst) target))))

;; Test 28: let with function binding shadowing
;; CHECK: 20
(let ((f (lambda (x) (* x 2))))
  (let ((f (lambda (x) (* x 4))))
    (f 5)))

;; Test 29: letrec with multiple self-referential bindings
;; CHECK: 8
(letrec ((a (lambda () (+ 1 (b))))
         (b (lambda () (+ 2 (c))))
         (c (lambda () 5)))
  (a))

;; Test 30: Named let with continuation-like behavior
;; CHECK: 100
(let loop ((n 0) (acc 0))
  (if (>= n 10)
      acc
      (loop (+ n 1) (+ acc 10))))

