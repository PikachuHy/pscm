;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; TODO: %pscm_main --test %s | FileCheck %s

(define (generate-one-element-at-a-time lst)
  (define (control-state return)
    (for-each
      (lambda (element)
        (set! return (call/cc
          (lambda (resume-here)
            (set! control-state resume-here)
            (return element)))))
      lst)
    (return 'you-fell-off-the-end))
  (define (generator)
    (call/cc control-state))
 generator)

;; CHECK: #<procedure generator ()>
(generate-one-element-at-a-time '(0 1 2))

(define generate-digit (generate-one-element-at-a-time '(0 1 2)))

;; CHECK-NEXT: 0
(generate-digit)

;; CHECK-NEXT: 1
(generate-digit)

;; CHECK-NEXT: 2
(generate-digit)

;; CHECK-NEXT: you-fell-off-the-end
(generate-digit)

;; CHECK-NEXT: you-fell-off-the-end
(generate-digit)

;; CHECK-NEXT: you-fell-off-the-end
(generate-digit)

;; CHECK-NEXT: you-fell-off-the-end
(generate-digit)
