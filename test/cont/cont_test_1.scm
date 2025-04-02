;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; TODO: %pscm_main --test %s | FileCheck %s


(define (f return)
    (return 2)
    3)

;; CHECK: 2
(call/cc f)
;; CHECK-NEXT: 2
(call/cc f)
;; CHECK-NEXT: 2
(call/cc f)
;; CHECK-NEXT: 2
(call/cc f)

;; CHECK: 2
(define (f a) 2)
(call/cc f)

;; CHECK: -3
(call/cc
 (lambda (exit)
   (for-each (lambda (x)
	       (if (negative? x) (exit x)))
	     '(54 0 37 -3 245 19))
   #t))
