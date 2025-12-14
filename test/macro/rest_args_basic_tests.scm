;; RUN: %pscm_cc --test %s | FileCheck %s

;; Basic test: define-macro with only rest arguments
(define-macro (collect . items)
  `(list ,@items))

;; CHECK: #<macro! collect>
collect

;; CHECK: ()
(collect)

;; CHECK: (1)
(collect 1)

;; CHECK: (1 2 3)
(collect 1 2 3)

;; CHECK: (a b c)
(collect 'a 'b 'c)

;; Basic test: define-macro with regular args + rest args
(define-macro (prepend x y . rest)
  `(list ,x ,y ,@rest))

;; CHECK: #<macro! prepend>
prepend

;; CHECK: (1 2)
(prepend 1 2)

;; CHECK: (1 2 3)
(prepend 1 2 3)

;; CHECK: (1 2 3 4 5)
(prepend 1 2 3 4 5)

;; Test: rest args with quasiquote
(define-macro (make-list . elems)
  `(quote ,elems))

;; CHECK: #<macro! make-list>
make-list

;; CHECK: (1 2 3)
(make-list 1 2 3)

;; Test: rest args with unquote-splicing
(define-macro (append-all . lists)
  `(append ,@lists))

;; CHECK: #<macro! append-all>
append-all

;; CHECK: (1 2 3 4 5)
(append-all '(1 2) '(3 4 5))

;; Test: empty rest args
(define-macro (empty-rest . args)
  (if (null? args)
      `'empty
      `'not-empty))

;; CHECK: #<macro! empty-rest>
empty-rest

;; CHECK: empty
(empty-rest)

;; CHECK: not-empty
(empty-rest 1)

;; CHECK: not-empty
(empty-rest 1 2 3)

;; Test: rest args in nested macro
(define-macro (wrapper . body)
  `(begin ,@body))

;; CHECK: #<macro! wrapper>
wrapper

;; CHECK: 42
(wrapper 42)

;; CHECK: 100
(wrapper (display 100) (newline) 100)

;; Test: rest args with complex expressions
(define-macro (test-expr . args)
  (let ((expr (car args))
        (expected (cadr args)))
    `(if (equal? ,expr ,expected)
         'pass
         'fail)))

;; CHECK: #<macro! test-expr>
test-expr

;; CHECK: pass
(test-expr 8 8)

;; CHECK: pass
(test-expr (+ 3 4) 7)

;; CHECK: fail
(test-expr 1 2)

;; Test complete
(display "All basic rest args macro tests passed!")
(newline)
