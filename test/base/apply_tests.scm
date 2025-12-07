;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s
;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: 7
(apply + '(3 4))

(define (test expect fun . args)
  ((lambda (res)
           (cond ((not (equal? expect res)) #f)
                  (else #t)))
    (apply fun args)))

;; CHECK: #t
(test 7 apply + '(3 4))

;; CHECK: 17
(apply + 10 (list 3 4))

;; CHECK: 6
(apply (lambda (x) (+ x x)) '(3))

;; CHECK: 0
(apply + '())

