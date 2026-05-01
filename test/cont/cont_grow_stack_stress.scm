;; RUN: %pscm_cc --test %s | FileCheck %s
;;
;; Stress test for grow_stack: captures a continuation from deep recursion
;; (200 levels, forcing a large stack save), then invokes it from shallow
;; and intermediate depths to force repeated grow_stack expansions.
;; Each grow_stack call must allocate enough stack space for the saved
;; continuation; if the compiler optimizes away the growth array or the
;; stack size calculation is wrong, this test will segfault.

(define captured #f)

(define (deep-recurse n)
  (if (= n 0)
      (call/cc (lambda (k)
                 (set! captured k)
                 'at-bottom-200))
      (deep-recurse (- n 1))))

;; Capture continuation at depth 200.
;; CHECK: at-bottom-200
(deep-recurse 200)

;; Invoke from top level — maximum grow_stack stress.
;; CHECK: from-top
(captured 'from-top)

;; Multi-shot from top level.
;; CHECK: multi-1
(captured 'multi-1)

;; CHECK: multi-2
(captured 'multi-2)

;; Invoke from intermediate recursion depth.
(define (mid-depth n)
  (if (= n 0)
      (captured 'from-depth-50)
      (mid-depth (- n 1))))

;; CHECK: from-depth-50
(mid-depth 50)

;; Invoke from shallow recursion depth.
(define (shallow n)
  (if (= n 0)
      (captured 'from-depth-10)
      (shallow (- n 1))))

;; CHECK: from-depth-10
(shallow 10)
