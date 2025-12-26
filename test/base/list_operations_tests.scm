;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test list-head function
;; CHECK: (1 2)
(define a '(1 2 3 4))
(list-head a 2)

;; Test list-tail function
;; CHECK: (3 4)
(list-tail a 2)

;; Test list-ref function
;; CHECK: 3
(list-ref a 2)

;; CHECK: 1
(list-ref a 0)

;; Test list operations with different indices
;; CHECK: (a b)
(list-head '(a b c d) 2)

;; CHECK: (c d)
(list-tail '(a b c d) 2)

;; CHECK: c
(list-ref '(a b c d) 2)

;; Test list operations with empty list
;; CHECK: ()
(list-head '() 0)

;; CHECK: ()
(list-tail '() 0)

