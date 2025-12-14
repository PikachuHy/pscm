;; RUN: %pscm_cc --test %s | FileCheck %s --check-prefix=LONGJMP
;; RUN: %pscm_cc --test %s | FileCheck %s

;; LONGJMP-NOT: not supported
;; CHECK: -3
(call/cc
 (lambda (exit)
   (for-each (lambda (x)
	       (if (negative? x) (exit x)))
	     '(54 0 37 -3 245 19))
   #t))
