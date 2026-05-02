;; Test: nested module resolution -- modules that use other modules
(set! %load-path (cons "." %load-path))
(load "nested_a.scm")
(load "nested_b.scm")

;; Verify module A's exports are visible
(display (module-ref (resolve-module '(test nested a)) 'value-a)) (newline)

;; Verify module B can see module A's exports
(display (module-ref (resolve-module '(test nested b)) 'value-b)) (newline)

;; Verify isolation: module B's internal symbol should NOT be visible
(display (module-bound? (resolve-module '(test nested b)) 'internal-b)) (newline)

(display "NESTED-PASS") (newline)
