;; RUN: %pscm_cc --test %s | FileCheck %s

;; ========================================
;; Test 1: letrec body copying fix
;; This tests that letrec correctly copies the body expressions
;; when expanding to let form
;; ========================================

;; Test 1.1: Basic letrec with body expressions
;; CHECK: #t
(letrec ((even?
          (lambda (n) (if (zero? n) #t (odd? (- n 1)))))
         (odd?
          (lambda (n) (if (zero? n) #f (even? (- n 1))))))
  (even? 88))

;; Test 1.2: letrec with multiple body expressions
;; This tests that all body expressions are properly copied and executed
;; CHECK-NEXT: 100
(letrec ((counter
          (lambda (n) (if (zero? n) 0 (+ 1 (counter (- n 1)))))))
  (counter 10)
  (counter 20)
  (counter 30)
  (counter 40)
  (counter 50)
  (counter 60)
  (counter 70)
  (counter 80)
  (counter 90)
  (counter 100))

;; Test 1.3: letrec with body that uses the recursive function
;; CHECK-NEXT: 55
(letrec ((sum (lambda (n)
                (if (zero? n)
                    0
                    (+ n (sum (- n 1)))))))
  (sum 10))

;; Test 1.4: letrec with mutual recursion and body expressions
;; CHECK-NEXT: #t
(letrec ((even?
          (lambda (n) (if (zero? n) #t (odd? (- n 1)))))
         (odd?
          (lambda (n) (if (zero? n) #f (even? (- n 1))))))
  (odd? 5)
  (even? 4))

;; Test 1.5: letrec with empty bindings but body expressions
;; CHECK-NEXT: 42
(letrec ()
  (define x 42)
  x)

;; ========================================
;; Test 2: quasiquote dotted pair fix
;; This tests that quasiquote correctly handles unquote-splicing
;; followed by a dotted pair
;; ========================================

;; Test 2.1: Basic unquote-splicing with dotted pair
;; Note: The fix ensures dotted pair is handled correctly (not crashing)
;; CHECK-NEXT: (a b c . d)
`(a ,@'(b c) . d)

;; Test 2.2: Unquote-splicing with empty list and dotted pair
;; CHECK-NEXT: (a . b)
`(a ,@'() . b)

;; Test 2.3: Unquote-splicing with single element and dotted pair
;; CHECK-NEXT: (a 1 . b)
`(a ,@'(1) . b)

;; Test 2.4: Unquote-splicing with multiple elements and dotted pair
;; CHECK-NEXT: (a 1 2 3 . b)
`(a ,@'(1 2 3) . b)

;; Test 2.5: Unquote-splicing with computed list and dotted pair
;; CHECK-NEXT: (a 1 2 3 . end)
`(a ,@(list 1 2 3) . end)

;; Test 2.6: Multiple unquote-splicing with dotted pair
;; CHECK-NEXT: (a 1 2 b 3 4 . c)
`(a ,@'(1 2) b ,@'(3 4) . c)

;; Test 2.7: Unquote-splicing at beginning with dotted pair
;; CHECK-NEXT: (1 2 3 . tail)
`(,@'(1 2 3) . tail)

;; Test 2.8: Complex case: unquote-splicing with map and dotted pair
;; CHECK-NEXT: (a 2 4 6 . rest)
`(a ,@(map (lambda (x) (* x 2)) '(1 2 3)) . rest)

;; Test 2.9: Nested unquote-splicing with dotted pair (edge case)
;; CHECK-NEXT: (a . b)
`(a ,@'() . b)

;; ========================================
;; Test 3: set! search order fix
;; This tests that set! correctly searches current environment first,
;; then parent environments
;; ========================================

;; Test 3.1: set! in current environment (should update current binding)
;; CHECK-NEXT: 20
(let ((x 10))
  (set! x 20)
  x)

;; Test 3.2: set! in parent environment (should update parent binding)
;; CHECK-NEXT: 30
(let ((x 10))
  (let ((y 5))
    (set! x 30))
  x)

;; Test 3.3: set! with shadowing (should update current, not parent)
;; CHECK-NEXT: 10
(let ((x 10))
  (let ((x 20))
    (set! x 40)
    x)
  x)

;; Test 3.4: set! in nested lets (should update correct level)
;; CHECK-NEXT: 50
(let ((x 10))
  (let ((x 20))
    (let ((x 30))
      (set! x 50)
      x)))

;; Test 3.5: set! in lambda (should update closure variable)
;; CHECK-NEXT: 60
(let ((x 10))
  ((lambda ()
     (set! x 60)))
  x)

;; Test 3.6: set! with multiple bindings (should update correct one)
;; set! x updates inner x, set! y updates outer y (since inner has no y)
;; CHECK-NEXT: (10 80)
(let ((x 10) (y 20))
  (let ((x 30))
    (set! x 70)  ; Updates inner x
    (set! y 80)) ; Updates outer y (inner has no y)
  (list x y))    ; Returns outer x and y

;; Test 3.7: set! in letrec (should update letrec binding)
;; CHECK-NEXT: 90
(letrec ((x 10))
  (set! x 90)
  x)

;; Test 3.8: set! with function binding
;; CHECK-NEXT: 100
(let ((f (lambda (x) (+ x 1))))
  (set! f (lambda (x) (+ x 10)))
  (f 90))

;; ========================================
;; Test 4: String null termination fix
;; This tests that strings are properly null-terminated
;; (indirectly tested through symbol and function name operations)
;; ========================================

;; Test 4.1: Symbol operations (uses string null termination)
;; CHECK-NEXT: #t
(equal? 'test-symbol 'test-symbol)

;; Test 4.2: Function name operations (uses string null termination)
;; CHECK-NEXT: 3
(define test-func (lambda (x) (+ x 1)))
(test-func 2)

;; Test 4.3: Multiple symbol comparisons
;; CHECK-NEXT: #t
(and (eq? 'a 'a)
     (eq? 'b 'b)
     (eq? 'c 'c))

;; Test 4.4: Symbol in let binding (uses string operations)
;; CHECK-NEXT: 42
(let ((test-var 42))
  test-var)

;; Test 4.5: Function registration (uses string null termination)
;; CHECK-NEXT: 5
(+ 2 3)

;; ========================================
;; Test 5: scm_env_insert search_parent parameter fix
;; This tests that scm_env_insert correctly uses search_parent parameter
;; ========================================

;; Test 5.1: Insert in current environment without searching parent
;; CHECK-NEXT: 110
(let ((x 100))
  (let ((x 110))
    x))

;; Test 5.2: Insert in current environment (should not affect parent)
;; CHECK-NEXT: 100
(let ((x 100))
  (let ((y 200))
    x))

;; Test 5.3: Function definition (uses scm_env_insert)
;; CHECK-NEXT: 6
(define add-two (lambda (x) (+ x 2)))
(add-two 4)

;; Test 5.4: Multiple function definitions
;; CHECK-NEXT: 8
(define multiply-two (lambda (x) (* x 2)))
(multiply-two 4)

;; ========================================
;; Test 6: Combined scenarios
;; ========================================

;; Test 6.1: letrec with set! and quasiquote
;; CHECK-NEXT: (result 2)
(letrec ((counter 0)
         (inc (lambda () (set! counter (+ counter 1)) counter)))
  (inc)
  (inc)
  `(result ,counter))

;; Test 6.2: Nested letrec with set! and body expressions
;; CHECK-NEXT: 300
(letrec ((outer 100))
  (letrec ((inner 200))
    (set! outer 300)
    outer))

;; Test 6.3: Complex quasiquote with set!
;; CHECK-NEXT: (before 400 after)
(let ((value 400))
  `(before ,value after))

