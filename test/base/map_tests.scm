;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: (4 6)
(map (lambda (x y) (+ x y))
          '(1 2)
          '(3 4))

;; CHECK: (4 5 6)
(map abs '(4 -5 6))

;; CHECK: (1 2 3)
(let ((count 0))
     (map (lambda (ignored)
            (set! count (+ count 1))
            count)
          '(a b c)))

;; Test map as a function value (stored in variable and called)
;; Note: These tests verify that map can be used as a function value
;; The key test is that map works when passed as a value, not just as a special form
;; CHECK: (2 4 6)
(let ((f map))
  (f * '(1 2 3) '(2 2 2)))

;; CHECK: (1 4 9)
(let ((f map))
  (f (lambda (x) (* x x)) '(1 2 3)))

;; CHECK: (b e h)
(let ((f map))
  (f cadr '((a b) (d e) (g h))))

;; Test map with compose (map as function value)
(define (compose f g)
  (lambda args (f (apply g args))))
;; CHECK: (2.0 3.0 4.0)
(let ((f map))
  (f (compose sqrt *) '(4 9 16) '(1 1 1)))

;; Test map with multiple lists
;; CHECK: (5 7 9)
(let ((f map))
  (f + '(1 2 3) '(4 5 6)))
