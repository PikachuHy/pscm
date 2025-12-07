;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s
;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: (1 . 2)
(cons 1 2)

;; CHECK: (1 2)
(cons 1 (cons 2 '()))

(define (square x) (* x x))
;; CHECK: #<procedure square (x)>
square

;; CHECK: 1
1

;; CHECK: (0 1 2)
'(0 1 2)

;; CHECK: (45)
'(45)

;; CHECK: 5
(+ (+ (+ 3) 2 ))

;; CHECK: -1
(- 1)

;; CHECK: -3
-3
;; CHECK: 123
123
;; CHECK: 12.7
12.7
;; CHECK: -3
-3

;; CHECK: (a b c d)
'(a b . (c d))
