;; RUN: %pscm_cc --test %s | FileCheck %s
;; Test internal catch functions behavior (indirectly through catch/throw)

;; Note: scm_internal_catch and scm_internal_lazy_catch are C API functions.
;; We test their behavior indirectly through the Scheme catch function,
;; which uses scm_internal_catch internally.

;; Test 1: Normal catch (uses scm_internal_catch) - basic functionality
;; CHECK: caught
(define tag1 (string->symbol "tag1"))
(catch tag1
       (lambda () (throw tag1 "error"))
       (lambda (key . args) 'caught))

;; Test 2: Normal catch returns value when no exception
;; CHECK-NEXT: 123
(catch tag1
       (lambda () 123)
       (lambda (key . args) 'error))

;; Test 3: Nested catches (both use scm_internal_catch)
;; CHECK-NEXT: inner-caught
(define outer-tag (string->symbol "outer"))
(define inner-tag (string->symbol "inner"))
(catch outer-tag
       (lambda ()
         (catch inner-tag
                (lambda () (throw inner-tag "inner error"))
                (lambda (key . args) 'inner-caught)))
       (lambda (key . args) 'outer-caught))

;; Test 4: Outer catch handles exception not caught by inner
;; CHECK-NEXT: outer-caught
(catch outer-tag
       (lambda ()
         (catch inner-tag
                (lambda () (throw outer-tag "outer error"))
                (lambda (key . args) 'inner-caught)))
       (lambda (key . args) 'outer-caught))

;; Test 5: Catch #t catches all exceptions
;; CHECK-NEXT: caught-all
(define any-tag (string->symbol "any-tag"))
(catch #t
       (lambda () (throw any-tag "any error"))
       (lambda (key . args) 'caught-all))

;; Test 6: Handler receives correct arguments
;; CHECK-NEXT: test-arg
(define arg-tag (string->symbol "arg-tag"))
(catch arg-tag
       (lambda () (throw arg-tag "test-arg"))
       (lambda (key . args) 
         (if (and (pair? args) (string? (car args)))
             (car args)
             'wrong-args)))

;; Test 7: Multiple throws in sequence (all caught)
;; CHECK-NEXT: first
(define seq-tag (string->symbol "seq"))
(define result7 (catch seq-tag
       (lambda () 
         (catch seq-tag
                (lambda () (throw seq-tag "first"))
                (lambda (key . args) (display (car args)) (newline) 'first)))
       (lambda (key . args) 'outer)))
;; CHECK-NEXT: second
(define result7b (catch seq-tag
       (lambda () (throw seq-tag "second"))
       (lambda (key . args) (display (car args)) (newline) 'second)))

;; Test 8: Catch with dynamic-wind interaction
;; CHECK-NEXT: in-guard-called
;; CHECK-NEXT: caught
;; CHECK-NEXT: out-guard-called
(define wind-tag (string->symbol "wind"))
(define guard-called #f)
(define result (dynamic-wind
 (lambda () (set! guard-called #t) (display "in-guard-called") (newline))
 (lambda ()
   (catch wind-tag
          (lambda () (throw wind-tag "error"))
          (lambda (key . args) (display "caught") (newline) 'caught)))
 (lambda () (display "out-guard-called") (newline))))

;; Test 9: Catch preserves return value from body
;; CHECK-NEXT: 42
(catch tag1
       (lambda () 42)
       (lambda (key . args) 'error))

;; Test 10: Handler return value is used
;; CHECK-NEXT: handler-value
(catch tag1
       (lambda () (throw tag1 "error"))
       (lambda (key . args) 'handler-value))

