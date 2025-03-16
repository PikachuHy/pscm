;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s

;; CHECK: ()
(append)
;; CHECK: ()
(list)

(define a '(a b c d))
;; CHECK: (a b c d)
a

(define b (list-head a 2))
;; CHECK: (a b)
b
(set-cdr! b 2)
;; CHECK: (a . 2)
b
;; CHECK: (a b c d)
a

(define a '(a b c d))
(define b (list-tail a 2))
;; CHECK: (c d)
b
(set-cdr! b 2)
;; CHECK: (c . 2)
b

(define a '(a b c d))
(define b (list-tail a 3))
;; CHECK: (b c d)
b

;; CHECK: (d)
(last-pair '(a b c d))
;; CHECK: ()
(last-pair '())