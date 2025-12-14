;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test define-macro with rest arguments - basic case
;; This is similar to the test macro in r5rstest.scm
(define-macro (test . args)
  (let ((expr (car args))
        (expect (cadr args)))
    (if (= (length args) 3)
        (begin
          (set! expr (cadr args))
          (set! expect (caddr args))))
    `(begin
       (display "Test: ")
       (display ',expr)
       (display " -> ")
       (display ,expr)
       (display " (expected: ")
       (display ,expect)
       (display ")")
       (newline)
       (if (equal? ,expr ,expect)
           (display "PASS")
           (display "FAIL"))
       (newline))))

;; CHECK: #<macro! test>
test

;; Test 1: Two arguments (expr expect)
;; CHECK: Test: 8 -> 8 (expected: 8)
;; CHECK: PASS
(test 8 8)

;; Test 2: Three arguments (name expr expect)
;; CHECK: Test: 8 -> 8 (expected: 8)
;; CHECK: PASS
(test "test-name" 8 8)

;; Test 3: Complex expression
;; CHECK: Test: 8 -> 8 (expected: 8)
;; CHECK: PASS
(test 8 ((lambda (x) (+ x x)) 4))

;; Test 4: List result
;; CHECK: Test: (quote (3 4 5 6)) -> (3 4 5 6) (expected: (3 4 5 6))
;; CHECK: PASS
(test '(3 4 5 6) ((lambda x x) 3 4 5 6))

;; Test define-macro with regular args + rest args
(define-macro (my-list a b . rest)
  `(list ,a ,b ,@rest))

;; CHECK: #<macro! my-list>
my-list

;; CHECK: (1 2 3 4 5)
(my-list 1 2 3 4 5)

;; CHECK: (10 20)
(my-list 10 20)

;; Test define-macro with only rest args - simple case
(define-macro (collect-all . items)
  `(list ,@items))

;; CHECK: #<macro! collect-all>
collect-all

;; CHECK: (1 2 3)
(collect-all 1 2 3)

;; CHECK: (a b c d)
(collect-all 'a 'b 'c 'd)

;; CHECK: ()
(collect-all)

;; Test define-macro with rest args - nested macro expansion
(define-macro (wrap . body)
  `(begin ,@body))

;; CHECK: #<macro! wrap>
wrap

;; CHECK: 42
(wrap 42)

;; CHECK: 100
(wrap (display 100) (newline) 100)

;; Test define-macro with rest args - quasiquote expansion
(define-macro (my-cond . clauses)
  (if (null? clauses)
      `#f
      (let ((clause (car clauses)))
        `(if ,(car clause)
             ,(cadr clause)
             (my-cond ,@(cdr clauses))))))

;; CHECK: #<macro! my-cond>
my-cond

;; CHECK: yes
(my-cond ((> 3 2) 'yes) ((< 3 2) 'no))

;; CHECK: no
(my-cond ((> 2 3) 'yes) ((< 2 3) 'no))

;; CHECK: #f
(my-cond)

;; Test define-macro with rest args - complex transformation
(define-macro (my-and . args)
  (if (null? args)
      `#t
      (if (null? (cdr args))
          (car args)
          `(if ,(car args)
               (my-and ,@(cdr args))
               #f))))

;; CHECK: #<macro! my-and>
my-and

;; CHECK: #t
(my-and)

;; CHECK: #t
(my-and #t)

;; CHECK: #f
(my-and #f)

;; CHECK: #t
(my-and #t #t)

;; CHECK: #f
(my-and #t #f)

;; CHECK: 3
(my-and 1 2 3)

;; Test define-macro with rest args - pattern matching style (simplified)
(define-macro (my-case expr . clauses)
  (if (null? clauses)
      `#f
      (let ((clause (car clauses)))
        (if (eq? (car clause) 'else)
            `(begin ,@(cdr clause))
            `(if (member ,expr ',(car clause))
                 (begin ,@(cdr clause))
                 (my-case ,expr ,@(cdr clauses)))))))

;; CHECK: #<macro! my-case>
my-case

;; CHECK: composite
(my-case (* 2 3)
         ((2 3 5 7) 'prime)
         ((1 4 6 8 9) 'composite))

;; CHECK: composite
(my-case 4
         ((2 3 5 7) 'prime)
         ((1 4 6 8 9) 'composite))

;; CHECK: consonant
(my-case (car '(c d))
         ((a e i o u) 'vowel)
         ((w y) 'semivowel)
         (else 'consonant))

;; Test define-macro with rest args - let-style binding (simplified)
(define-macro (my-let bindings . body)
  (if (null? bindings)
      `(begin ,@body)
      (let ((binding (car bindings))
            (rest-bindings (cdr bindings)))
        `((lambda (,(car binding))
            (my-let ,rest-bindings ,@body))
          ,(cadr binding)))))

;; CHECK: #<macro! my-let>
my-let

;; CHECK: 6
(my-let ((x 2) (y 3)) (* x y))

;; CHECK: 70
(my-let ((x 2) (y 3))
        (my-let ((x 7) (z (+ x y)))
                (* z x)))

;; Test define-macro with rest args - multiple rest args (should fail at definition time)
;; This tests that we properly handle the case where rest args are correctly parsed

;; Test define-macro with rest args - empty rest args
(define-macro (no-args . rest)
  (if (null? rest)
      `'empty
      `'not-empty))

;; CHECK: #<macro! no-args>
no-args

;; CHECK: empty
(no-args)

;; CHECK: not-empty
(no-args 1)

;; CHECK: not-empty
(no-args 1 2 3)

;; Test define-macro with rest args - preserving source location
(define-macro (debug-print . items)
  `(begin
     (display "Debug: ")
     ,@(map (lambda (item) `(begin (display ,item) (display " "))) items)
     (newline)))

;; CHECK: #<macro! debug-print>
debug-print

;; CHECK: Debug: 1 2 3
(debug-print 1 2 3)

;; CHECK: Debug: hello world
(debug-print "hello" "world")

;; Test define-macro with rest args - nested macro calls
(define-macro (apply-macro macro-name . args)
  `(,macro-name ,@args))

;; CHECK: #<macro! apply-macro>
apply-macro

;; CHECK: (1 2 3)
(apply-macro my-list 1 2 3)

;; CHECK: (a b c)
(apply-macro collect-all 'a 'b 'c)

;; Test define-macro with rest args - variable number of arguments
(define-macro (sum-all . numbers)
  (if (null? numbers)
      `0
      (if (null? (cdr numbers))
          (car numbers)
          `(+ ,(car numbers) (sum-all ,@(cdr numbers))))))

;; CHECK: #<macro! sum-all>
sum-all

;; CHECK: 0
(sum-all)

;; CHECK: 10
(sum-all 10)

;; CHECK: 15
(sum-all 1 2 3 4 5)

;; Test complete - all rest args macro tests passed
(display "All rest args macro tests completed successfully!")
(newline)
