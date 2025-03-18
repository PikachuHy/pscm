;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; TODO: %pscm_main --test %s | FileCheck %s

(define c #f)
;; CHECK: talk1
(call/cc
  (lambda (c0)
          (set! c c0)
          'talk1))

;; CHECK: talk2
(c 'talk2)

;; CHECK: talk2
(c 'talk2)

;; CHECK: talk2
(c 'talk2)

;; CHECK: talk2
(c 'talk2)

(define cc #f)
(* 3 (call/cc (lambda (k)
                 (set! cc k)
                 (+ 1 2))))
;; CHECK: 9
(+ 100 (cc 3))
;; CHECK: 30 
(+ 100 (cc 10))
