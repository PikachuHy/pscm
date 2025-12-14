;; RUN: %pscm_cc --test %s | FileCheck %s --check-prefix=LONGJMP
;; RUN: %pscm_cc --test %s | FileCheck %s

;; LONGJMP-NOT: not supported

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
