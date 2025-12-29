;; RUN: %pscm_cc --test %s | FileCheck %s
;; Test variable operations (compatible with Guile 1.8)

;; Test 1: Create a variable with initial value
(define var1 (make-variable 42))
;; CHECK: #t
(variable? var1)

;; Test 2: Get value from variable
;; CHECK-NEXT: 42
(variable-ref var1)

;; Test 3: Check if variable is bound
;; CHECK-NEXT: #t
(variable-bound? var1)

;; Test 4: Create an undefined variable
(define unbound-var (make-undefined-variable))
;; CHECK-NEXT: #t
(variable? unbound-var)

;; Test 5: Check if undefined variable is bound
;; CHECK-NEXT: #f
(variable-bound? unbound-var)

;; Test 6: Set value of variable
(variable-set! var1 100)
;; CHECK-NEXT: 100
(variable-ref var1)

;; Test 7: Set value of undefined variable (make it bound)
(variable-set! unbound-var 200)
;; CHECK-NEXT: #t
(variable-bound? unbound-var)

;; CHECK-NEXT: 200
(variable-ref unbound-var)

;; Test 8: variable? returns #f for non-variables
;; CHECK-NEXT: #f
(variable? 42)

;; CHECK-NEXT: #f
(variable? "hello")

;; CHECK-NEXT: #f
(variable? 'symbol)

;; CHECK-NEXT: #f
(variable? #t)

;; Test 9: Multiple variables with different values
(define var2 (make-variable "test"))
(define var3 (make-variable 'symbol))
(define var4 (make-variable #t))
;; CHECK-NEXT: "test"
(variable-ref var2)

;; CHECK-NEXT: symbol
(variable-ref var3)

;; CHECK-NEXT: #t
(variable-ref var4)

;; Test 10: Variable can hold any type
(define var5 (make-variable (list 1 2 3)))
;; CHECK-NEXT: (1 2 3)
(variable-ref var5)

;; Test 11: Variable can be set to different types
(variable-set! var1 "changed to string")
;; CHECK-NEXT: "changed to string"
(variable-ref var1)

(variable-set! var1 (list 'a 'b 'c))
;; CHECK-NEXT: (a b c)
(variable-ref var1)

;; Test 12: Variable can hold another variable
(define var6 (make-variable var1))
;; CHECK-NEXT: #t
(variable? (variable-ref var6))

;; CHECK-NEXT: (a b c)
(variable-ref (variable-ref var6))

;; Test 13: Variable can hold nil
(define var7 (make-variable '()))
;; CHECK-NEXT: ()
(variable-ref var7)

;; Test 14: Variable can hold false
(define var8 (make-variable #f))
;; CHECK-NEXT: #f
(variable-ref var8)

;; Test 15: Variable can hold numbers
(define var9 (make-variable 123))
(define var10 (make-variable -456))
(define var11 (make-variable 3.14))
;; CHECK-NEXT: 123
(variable-ref var9)

;; CHECK-NEXT: -456
(variable-ref var10)

;; CHECK-NEXT: 3.14
(variable-ref var11)

;; Test 16: Variable can hold procedures
(define var12 (make-variable +))
;; CHECK-NEXT: #t
(procedure? (variable-ref var12))

;; Test 17: Variable can hold lists
(define var13 (make-variable (cons 1 2)))
;; CHECK-NEXT: (1 . 2)
(variable-ref var13)

