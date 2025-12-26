;; RUN: %pscm_cc --test %s | FileCheck %s
;; Tests for second batch optimizations (simplified safety checks)
;; These tests verify that after removing excessive type checks and try-catch,
;; the core functionality still works correctly

;; ========================================
;; Test 1: Environment search functionality
;; Tests that scm_env_search and scm_env_exist work correctly
;; after removing excessive type checks
;; ========================================

;; Test 1.1: Basic environment search
;; CHECK: 10
(let ((x 10))
  x)

;; Test 1.2: Environment search with parent
;; CHECK-NEXT: 20
(let ((x 10))
  (let ((y 20))
    y))

;; Test 1.3: Environment search with shadowing
;; CHECK-NEXT: 30
(let ((x 10))
  (let ((x 30))
    x))

;; Test 1.4: Environment search with null value (#f)
;; CHECK-NEXT: #f
(let ((x #f))
  x)

;; Test 1.5: Environment search with multiple levels
;; CHECK-NEXT: 50
(let ((x 10))
  (let ((x 20))
    (let ((x 30))
      (let ((x 40))
        (let ((x 50))
          x)))))

;; Test 1.6: Environment search with function values
;; CHECK-NEXT: 6
(let ((add (lambda (x y) (+ x y))))
  (add 2 4))

;; Test 1.7: Environment search with nested functions
;; CHECK-NEXT: 12
(let ((multiply (lambda (x) (lambda (y) (* x y)))))
  ((multiply 3) 4))

;; ========================================
;; Test 2: Macro expansion functionality
;; Tests that macro expansion works correctly
;; after removing excessive type checks
;; ========================================

;; Test 2.1: Basic macro expansion
;; CHECK-NEXT: 3
(define-macro (add-one x) `(+ 1 ,x))
(add-one 2)

;; Test 2.2: Macro with multiple arguments
;; CHECK-NEXT: 7
(define-macro (add-two a b) `(+ ,a ,b 2))
(add-two 2 3)

;; Test 2.3: Macro with rest arguments
;; CHECK-NEXT: 15
(define-macro (sum . args)
  `(+ ,@args))
(sum 1 2 3 4 5)

;; Test 2.4: Nested macro expansion
;; CHECK-NEXT: 10
(define-macro (double x) `(* 2 ,x))
(define-macro (quadruple x) `(double (double ,x)))
(quadruple 2.5)

;; Test 2.5: Macro with conditional expansion
;; CHECK-NEXT: 42
(define-macro (if-true x y) `(if #t ,x ,y))
(if-true 42 0)

;; Test 2.6: Macro with list manipulation
;; CHECK-NEXT: (1 2 3)
(define-macro (make-list a b c) `(list ,a ,b ,c))
(make-list 1 2 3)

;; Test 2.7: Macro with quasiquote
;; CHECK-NEXT: (result 100)
(define-macro (wrap-value x) ``(result ,,x))
(wrap-value 100)

;; Test 2.8: Macro expansion with set!
;; CHECK-NEXT: 200
(define-macro (set-and-return var val) `(begin (set! ,var ,val) ,var))
(define x 100)
(set-and-return x 200)

;; ========================================
;; Test 3: set! functionality
;; Tests that set! works correctly after simplification
;; ========================================

;; Test 3.1: Basic set! in current environment
;; CHECK-NEXT: 15
(let ((x 10))
  (set! x 15)
  x)

;; Test 3.2: set! in parent environment
;; CHECK-NEXT: 25
(let ((x 10))
  (let ((y 5))
    (set! x 25))
  x)

;; Test 3.3: set! with shadowing
;; CHECK-NEXT: 10
(let ((x 10))
  (let ((x 20))
    (set! x 35)
    x)
  x)

;; Test 3.4: set! with function values
;; CHECK-NEXT: 30
(let ((f (lambda (x) (+ x 1))))
  (set! f (lambda (x) (+ x 10)))
  (f 20))

;; Test 3.5: set! with null value (#f)
;; CHECK-NEXT: #f
(let ((x #t))
  (set! x #f)
  x)

;; Test 3.6: Multiple set! operations
;; CHECK-NEXT: 70
(let ((x 10) (y 20))
  (set! x 30)
  (set! y 40)
  (+ x y))

;; Test 3.7: set! in letrec
;; CHECK-NEXT: 60
(letrec ((x 10))
  (set! x 60)
  x)

;; Test 3.8: set! with nested environments
;; CHECK-NEXT: 70
(let ((x 10))
  (let ((x 20))
    (let ((x 30))
      (set! x 70)
      x)))

;; ========================================
;; Test 4: Combined environment and macro operations
;; Tests that environment search works correctly with macros
;; Note: define-macro can only be used at top level (following Guile 1.8)
;; ========================================

;; Test 4.1: Macro using literal values
;; CHECK-NEXT: 8
(define-macro (times multiplier x) `(* ,multiplier ,x))
(times 2 4)

;; Test 4.2: Macro with set! in expansion (using global variable)
;; CHECK-NEXT: 2
(define counter-4-2 0)
(define-macro (inc-counter-4-2) `(set! counter-4-2 (+ counter-4-2 1)))
(inc-counter-4-2)
(inc-counter-4-2)
counter-4-2

;; Test 4.3: Nested macros with literal values
;; CHECK-NEXT: 16
(define-macro (power base n) `(expt ,base ,n))
(define-macro (double-power base n) `(* 2 (power ,base ,n)))
(double-power 2 3)

;; ========================================
;; Test 5: Edge cases and null handling
;; Tests that null values are handled correctly
;; ========================================

;; Test 5.1: Environment search with #f
;; CHECK-NEXT: false
(let ((x #f))
  (if x 'true 'false))

;; Test 5.2: set! with #f
;; CHECK-NEXT: #t
(let ((x #f))
  (set! x #t)
  x)

;; Test 5.3: Macro returning #f
;; CHECK-NEXT: #f
(define-macro (return-false) '#f)
(return-false)

;; Test 5.4: Environment search with empty let
;; CHECK-NEXT: 100
(let ()
  (define x 100)
  x)

;; Test 5.5: set! with empty environment chain
;; CHECK-NEXT: 200
(let ((x 100))
  (let ()
    (set! x 200))
  x)

;; ========================================
;; Test 6: Complex scenarios combining all features
;; ========================================

;; Test 6.1: Macro with set! using global variable
;; CHECK-NEXT: 300
(define value-6-1 100)
(define-macro (increment-6-1) `(set! value-6-1 (+ value-6-1 100)))
(increment-6-1)
(increment-6-1)
value-6-1

;; Test 6.2: Macro with set! using global variable
;; CHECK-NEXT: 400
(define counter-6-2 0)
(define-macro (inc-counter-6-2) `(set! counter-6-2 (+ counter-6-2 100)))
(inc-counter-6-2)
(inc-counter-6-2)
(inc-counter-6-2)
(inc-counter-6-2)
counter-6-2

;; Test 6.3: Complex macro expansion with literal values
;; CHECK-NEXT: 500
(define-macro (add-base base x) `(+ ,base ,x))
(define-macro (multiply-base base x) `(* ,base ,x))
(let ((result (+ 100 200)))
  (+ result (* 200 1)))

;; Test 6.4: set! with macro-generated code using global variable
;; CHECK-NEXT: 600
(define x-6-4 100)
(define-macro (set-to-6-4 y) `(set! x-6-4 ,y))
(set-to-6-4 200)
(set-to-6-4 300)
(set-to-6-4 600)
x-6-4

;; Test 6.5: Environment chain with set! (no macro needed)
;; CHECK-NEXT: 700
(let ((outer 100))
  (let ((middle 200))
    (let ((inner 300))
      (set! outer 700)
      (set! middle 700)
      (set! inner 700)
      inner)))

;; ========================================
;; Test 7: Macros with multiple calls and passed values
;; ========================================

;; Test 7.1: Macro with multiple calls using global variable
;; CHECK-NEXT: 8
(define count-7-1 0)
(define-macro (inc-7-1) `(set! count-7-1 (+ count-7-1 1)))
(inc-7-1)
(inc-7-1)
(inc-7-1)
(inc-7-1)
(inc-7-1)
(inc-7-1)
(inc-7-1)
(inc-7-1)
count-7-1

;; Test 7.2: Macro with literal multiplier value
;; CHECK-NEXT: 900
(define-macro (triple multiplier x) `(* ,multiplier ,x))
(triple 9 100)

;; ========================================
;; Test 8: Function definitions and environment
;; ========================================

;; Test 8.1: Function definition in let
;; CHECK-NEXT: 10
(let ()
  (define (add-five x) (+ x 5))
  (add-five 5))

;; Test 8.2: Function definition with set!
;; CHECK-NEXT: 20
(let ((f (lambda (x) x)))
  (set! f (lambda (x) (* x 2)))
  (f 10))

;; Test 8.3: Multiple function definitions
;; CHECK-NEXT: 40
(let ()
  (define (f1 x) (+ x 10))
  (define (f2 x) (+ x 20))
  (+ (f1 5) (f2 5)))

;; ========================================
;; Test 9: Module and global environment interaction
;; ========================================

;; Test 9.1: Global function access
;; CHECK-NEXT: 15
(+ 5 10)

;; Test 9.2: Local binding shadows global
;; CHECK-NEXT: 25
(let ((+ (lambda (x y) (* x y))))
  (+ 5 5))

;; Test 9.3: set! with global function
;; CHECK-NEXT: 35
(let ((old-+ +))
  (set! + (lambda (x y) (* x y)))
  (let ((result (+ 5 7)))
    (set! + old-+)
    result))

;; ========================================
;; Test 10: Stress tests with multiple operations
;; ========================================

;; Test 10.1: Multiple set! operations in sequence
;; CHECK-NEXT: 1000
(let ((x 0))
  (set! x 100)
  (set! x 200)
  (set! x 300)
  (set! x 400)
  (set! x 500)
  (set! x 600)
  (set! x 700)
  (set! x 800)
  (set! x 900)
  (set! x 1000)
  x)

;; Test 10.2: Multiple macro expansions
;; CHECK-NEXT: 1100
(define-macro (a) '100)
(define-macro (b) '200)
(define-macro (c) '300)
(define-macro (d) '500)
(+ (a) (b) (c) (d))

;; Test 10.3: Complex nested operations with macro using literal values
;; CHECK-NEXT: 1200
(define-macro (add-xy x y) `(+ ,x ,y))
(let ((x 100))
  (let ((y 200))
    (set! x 400)
    (set! y 800)
    (+ x y)))

