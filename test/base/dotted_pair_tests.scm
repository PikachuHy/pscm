;; Test dotted pair parsing and printing
;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test simple dotted pair
'(1 . 2)
;; CHECK: (1 . 2)

;; Test proper list
'(1 2)
;; CHECK: (1 2)

;; Test longer dotted pair
'(1 2 3 . 4)
;; CHECK: (1 2 3 . 4)

;; Test proper list with 3 elements
'(1 2 3)
;; CHECK: (1 2 3)

;; Test cons creating dotted pair
(cons 1 2)
;; CHECK: (1 . 2)

;; Test cons creating proper list
(cons 1 '(2))
;; CHECK: (1 2)

;; Test nested dotted pair
'((1 . 2) (3 . 4))
;; CHECK: ((1 . 2) (3 . 4))

