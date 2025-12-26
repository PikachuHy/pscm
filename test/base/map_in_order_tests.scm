;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test map-in-order basic functionality
;; CHECK: (2 4 6)
(map-in-order (lambda (x) (* x 2)) '(1 2 3))

;; CHECK: (4 6)
(map-in-order (lambda (x y) (+ x y))
              '(1 2)
              '(3 4))

;; CHECK: (4 5 6)
(map-in-order abs '(4 -5 6))

;; Test map-in-order as a function value
;; CHECK: (2 4 6)
(let ((f map-in-order))
  (f * '(1 2 3) '(2 2 2)))

;; CHECK: (1 4 9)
(let ((f map-in-order))
  (f (lambda (x) (* x x)) '(1 2 3)))

;; Test map-in-order with multiple lists
;; CHECK: (5 7 9)
(let ((f map-in-order))
  (f + '(1 2 3) '(4 5 6)))

;; Test map-in-order preserves order
;; CHECK: (1 2 3 4 5)
(let ((count 0))
  (map-in-order (lambda (ignored)
                  (set! count (+ count 1))
                  count)
                '(a b c d e)))

