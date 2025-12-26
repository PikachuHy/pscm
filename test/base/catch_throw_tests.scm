;; RUN: %pscm_cc --test %s | FileCheck %s
;; Test catch/throw exception handling

;; Test 1: Basic catch/throw with symbol created via string->symbol
;; CHECK: ok
(define test-tag (string->symbol "test"))
(catch test-tag
       (lambda () (throw test-tag "hello"))
       (lambda (key . args) 'ok))

;; Test 2: catch #t catches all exceptions
;; CHECK-NEXT: caught-any
(define any-key-tag (string->symbol "any-key"))
(catch #t
       (lambda () (throw any-key-tag "any error"))
       (lambda (key . args) 'caught-any))

;; Test 3: catch returns normal value when no exception
;; CHECK-NEXT: 42
(define test-tag2 (string->symbol "test"))
(catch test-tag2
       (lambda () 42)
       (lambda (key . args) 'error))

;; Test 4: Handler receives key and args
;; CHECK-NEXT: test-key
(define test-key-tag (string->symbol "test-key"))
(catch test-key-tag
       (lambda () (throw test-key-tag "arg1" "arg2"))
       (lambda (key . args) key))

;; Test 5: Handler can return a value
;; CHECK-NEXT: recovered
(define error-tag (string->symbol "error"))
(catch error-tag
       (lambda () (throw error-tag "something went wrong"))
       (lambda (key . args) 'recovered))

;; Test 6: Error handling - eval_error now throws instead of exit
;; CHECK-NEXT: error-handled
(define error-tag2 (string->symbol "error"))
(catch error-tag2
       (lambda () 
         ;; Trigger a type error (car expects a pair)
         (car 42))
       (lambda (key . args) 'error-handled))
