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

;; Test cdr with dotted pair (should return the cdr value directly)
;; CHECK: 2
(cdr '(1 . 2))

;; Test cdr with proper list (should return the rest of the list)
;; CHECK: (2 3)
(cdr '(1 2 3))

;; Test cdr with single element list (should return empty list)
;; CHECK: ()
(cdr '(1))

;; Test car and cdr with dotted pair
;; CHECK: 1
(car '(1 . 2))
;; CHECK: 2
(cdr '(1 . 2))

;; Test nested dotted pair cdr
;; CHECK: (2 . 3)
(cdr '(1 2 . 3))

