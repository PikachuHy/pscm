;; RUN: %pscm_cc --test %s | FileCheck %s

(define (make-counter)
  (let ((count 0))
    (lambda () (set! count (+ count 1)) count)))
(define c (make-counter))

;; CHECK: 1
(c)

;; CHECK: ()
(gc)

;; CHECK: 2
(c)

;; CHECK: 3
(c)
