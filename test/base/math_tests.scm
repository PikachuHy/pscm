;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: -1
(expt -1 -255)
;; CHECK: #t
(number? 3)
;; CHECK: 1/3
(/ 1 3)
;; CHECK: 100/3
(* (/ 1 3) 100)

;; Test quotient function
;; CHECK: 5
(quotient 35 7)
;; CHECK: -5
(quotient -35 7)
;; CHECK: -5
(quotient 35 -7)
;; CHECK: 5
(quotient -35 -7)
;; CHECK: 3
(quotient 10 3)
;; CHECK: -3
(quotient -10 3)
;; CHECK: -3
(quotient 10 -3)
;; CHECK: 3
(quotient -10 -3)

;; Test modulo function
;; CHECK: 1
(modulo 13 4)
;; CHECK: 3
(modulo -13 4)
;; CHECK: -3
(modulo 13 -4)
;; CHECK: -1
(modulo -13 -4)
;; CHECK: 0
(modulo 10 5)
;; CHECK: 2
(modulo -8 5)

;; Test remainder function
;; CHECK: 1
(remainder 13 4)
;; CHECK: -1
(remainder -13 4)
;; CHECK: 1
(remainder 13 -4)
;; CHECK: -1
(remainder -13 -4)

;; Test gcd function
;; CHECK: 4
(gcd 32 -36)
;; CHECK: 1
(gcd 17 19)
;; CHECK: 6
(gcd 12 18)
;; CHECK: 0
(gcd)
;; CHECK: 5
(gcd 5)
;; CHECK: 1
(gcd 5 3 7)

;; Test lcm function
;; CHECK: 288
(lcm 32 -36)
;; CHECK: 323
(lcm 17 19)
;; CHECK: 36
(lcm 12 18)
;; CHECK: 1
(lcm)
;; CHECK: 5
(lcm 5)
;; CHECK: 105
(lcm 5 3 7)
;; CHECK: 0
(remainder 10 5)
;; CHECK: -3
(remainder -8 5)

;; Test max function
;; CHECK: 5
(max 3 5 2)
;; CHECK: 10
(max 1 2 3 4 5 6 7 8 9 10)
;; CHECK: -1
(max -5 -3 -1 -10)
;; CHECK: 3
(max 3)
;; CHECK: 5.5
(max 3.5 5.5 2.0)
;; CHECK: 5
(max 3 5.0 2)
;; CHECK: 5.0
(max 3.0 5 2.0)

;; Test min function
;; CHECK: 2
(min 3 5 2)
;; CHECK: 1
(min 1 2 3 4 5 6 7 8 9 10)
;; CHECK: -10
(min -5 -3 -1 -10)
;; CHECK: 3
(min 3)
;; CHECK: 2.0
(min 3.5 5.5 2.0)
;; CHECK: 2
(min 3 5.0 2)
;; CHECK: 2.0
(min 3.0 5 2.0)

;; Test sqrt function
;; CHECK: 2.0
(sqrt 4)
;; CHECK: 3.0
(sqrt 9)
;; CHECK: 4.0
(sqrt 16)
;; CHECK: 5.0
(sqrt 25)
;; CHECK: 1.0
(sqrt 1)
;; CHECK: 0.0
(sqrt 0)
;; CHECK: 1.4142135623731
(sqrt 2)
;; CHECK: 1.7320508075689
(sqrt 3)
;; CHECK: 10.0
(sqrt 100)
;; CHECK: 30.0
(sqrt 900)

;; Test sqrt with compose (sqrt as function value)
;; CHECK: 30.0
(define (compose f g)
  (lambda args (f (apply g args))))
((compose sqrt *) 12 75)

;; Test sqrt in map
;; CHECK: (2.0 3.0 4.0)
(map sqrt '(4 9 16))
;; CHECK: (1.0 1.4142135623731 1.7320508075689)
(map sqrt '(1 2 3))
