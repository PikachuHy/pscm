;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s --check-prefix=DIRECT

;; CHECK: #t
(cond ((eq? 1 2))
  ((eq? 1 1))
  (else 'a))

;; CHECK: greater
(cond ((> 3 2) 'greater)
  ((< 3 2) 'less))

;; CHECK: equal
(cond ((> 3 3) 'greater)
  ((< 3 3) 'less)
  (else 'equal))

;; CHECK: 2
(cond ((assv 'b '((a 1) (b 2))) => cadr)
  (else #f))

;; CHECK: 2
(cond ((> 3 2) (+ 1) (+ 2)))

;; CHECK: ok
(let ((=> 1)) (cond (#t => 'ok)))

;; DIRECT: 3
;; REGISTER_MACHINE not supported the feature now
(cond
  ((= 1 2) '())
  (else 1 2 3))
