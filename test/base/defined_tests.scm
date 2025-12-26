;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test the defined? function
;; defined? checks if a symbol is defined in the current environment or module

;; Test defined? with built-in functions (should return #t)
;; CHECK: #t
(defined? 'car)

;; CHECK: #t
(defined? 'cdr)

;; CHECK: #t
(defined? '+)

;; CHECK: #t
(defined? 'cons)

;; CHECK: #t
(defined? 'list)

;; Note: define and lambda are special forms, may not be in environment
;; CHECK: #f
(defined? 'define)

;; CHECK: #f
(defined? 'lambda)

;; Test defined? with undefined symbols (should return #f)
;; CHECK: #f
(defined? 'nonexistent-symbol)

;; CHECK: #f
(defined? 'undefined-var)

;; Test defined? with user-defined variables
;; CHECK: #f
(defined? 'my-var)

;; CHECK: #t
(define my-var 42)
(defined? 'my-var)

;; Test defined? with user-defined procedures
;; CHECK: #f
(defined? 'my-proc)

;; CHECK: #t
(define (my-proc x) (* x 2))
(defined? 'my-proc)

;; Test defined? with variables defined in let
;; Note: defined? checks global environment and module, not local let bindings
;; CHECK: #f
(defined? 'local-var)

;; CHECK: #f
(let ((local-var 10))
  (defined? 'local-var))

;; Test defined? with symbols extracted from pairs
;; CHECK: #f
(define test-pair '(my-symbol other))
(defined? (car test-pair))

;; CHECK: #t
(define my-symbol 123)
(defined? (car test-pair))

;; Test defined? with nested symbols
;; CHECK: #f
(define nested-pair '((inner-symbol)))
(defined? (car (car nested-pair)))

;; CHECK: #t
(define inner-symbol 456)
(defined? (car (car nested-pair)))

;; Test defined? with symbols that are already defined
;; CHECK: #t
(define existing-var 789)
(defined? 'existing-var)

;; Test defined? with multiple definitions
;; CHECK: #t
(define var1 1)
(defined? 'var1)

;; CHECK: #t
(define var2 2)
(defined? 'var2)

;; CHECK: #t
(define var3 3)
(defined? 'var3)

;; Test defined? with redefined variables
;; CHECK: #t
(define redef-var 100)
(defined? 'redef-var)

;; CHECK: #t
(define redef-var 200)
(defined? 'redef-var)

;; Test defined? with special forms (these are not symbols, but test error handling)
;; Note: defined? should only accept symbols, but we test that it works correctly

;; Test defined? with quoted symbols vs unquoted
;; CHECK: #f
(defined? 'quoted-var)

;; CHECK: #t
(define quoted-var 999)
(defined? 'quoted-var)

;; Test defined? in different scopes
;; CHECK: #t
(define global-var 111)
(defined? 'global-var)

;; Test that define in let creates a local binding (not visible to defined?)
;; Note: In this implementation, define in let creates a local binding
;; CHECK: #f
(let ((local-scope 222))
  (define local-var-in-let 333)
  (defined? 'local-var-in-let))

;; Test defined? with module-level definitions
;; CHECK: #t
(define module-var 444)
(defined? 'module-var)

