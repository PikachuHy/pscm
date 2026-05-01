;; RUN: %pscm_cc --test %s | FileCheck %s

(define x (list 'a 'b 'c))

(do ((i 0 (+ i 1))) ((= i 50))
  (gc))

;; CHECK: a
(car x)
;; CHECK: b
(cadr x)
;; CHECK: c
(caddr x)
