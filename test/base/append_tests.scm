;; RUN: %pscm_cc --test %s | FileCheck %s

(define a '(1 2 3))
(append a 4)
;; CHECK: (1 2 3)
a

;; CHECK: (a b c . d)
(append '(a b) '(c . d))

;; CHECK: ()
(apply append '())

;; CHECK: ()
(apply append '(()))

;; CHECK: (1 2 3 4 5 6)
(append '(1 2) '(3 4) '(5 6))

;; CHECK: (1 2 3 4 5 6)
(apply append '((1 2) (3 4) (5 6)))

;; Test append with empty list and non-list (should return the non-list)
;; CHECK: a
(append '() 'a)

;; Test append with empty list and list
;; CHECK: (1 2 3)
(append '() '(1 2 3))

;; Test append with multiple empty lists
;; CHECK: (1 2 3)
(append '() '() '(1 2 3))
