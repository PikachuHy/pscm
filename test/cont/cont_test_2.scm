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
