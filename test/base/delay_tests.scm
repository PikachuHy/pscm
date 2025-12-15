;; RUN: %pscm_cc --test %s | FileCheck %s

;; 1. basic delay/force
;; CHECK: 3
(force (delay (+ 1 2)))

;; 2. force same promise multiple times should only evaluate once
;;    use a side effect counter to observe evaluation count
;;    NOTE: outer let binds count so the delay thunk captures it correctly
;; CHECK: (42 42 1)
(let ((count 0))
  (let ((p (delay (begin
                    (set! count (+ count 1))
                    42))))
    (list (force p) (force p) count)))

;; 3. simple lazy stream via promises
;;    a-stream = 0,1,2,3,...
;;    (head (tail (tail a-stream))) => 2
;; CHECK: 2
(letrec ((a-stream
           (letrec ((next (lambda (n)
                            (cons n (delay (next (+ n 1)))))))
             (next 0)))
         (head car)
         (tail (lambda (stream) (force (cdr stream)))))
  (head (tail (tail a-stream))))

;; 4. re-entrant force example from R4RS/R5RS tests
;;    should evaluate to 3 (not 4) when force handles re-entrancy correctly
;; CHECK: 3
(letrec ((p (delay (if c
                    3
                    (begin
                      (set! c #t)
                      (+ (force p) 1)))))
         (c #f))
  (force p))
