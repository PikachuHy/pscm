;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s
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
