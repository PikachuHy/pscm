;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s

(define x '(1 3 5 7 9))

(define sum 0)

(do ((x x (cdr x)))
    ((null? x))
    (set! sum (+ sum (car x))))

;; CHECK: 25
sum

;; CHECK: 25
(let ((x '(1 3 5 7 9))
                   (sum 0))
               (do ((x x (cdr x)))
                   ((null? x))
                 (set! sum (+ sum (car x))))
               sum)

;; CHECK: 9
(letrec () (define x 9) x)  
