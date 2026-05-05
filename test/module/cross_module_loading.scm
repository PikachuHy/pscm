;; Test: cross-file module loading via scm_resolve_module
;; Uses resolve-module which triggers scm_resolve_module -> file load path
(set! %load-path (cons "." %load-path))

;; Load via resolve-module (triggers scm_resolve_module pre-creation + file load)
(define mod-a (resolve-module '(cross module a)))
(define mod-b (resolve-module '(cross module b)))

;; Verify module a exports
(define val-a (module-ref mod-a 'value-a))
(if (= val-a 42)
    (begin (display "PASS: cross_module_a value-a is 42") (newline))
    (begin (display "FAIL: cross_module_a value-a is ") (display val-a) (newline) (exit 1)))

;; Verify module b sees module a's exports via uses chain
(define val-b (module-ref mod-b 'value-b))
(if (= val-b 84)
    (begin (display "PASS: cross_module_b value-b is 84") (newline))
    (begin (display "FAIL: cross_module_b value-b is ") (display val-b) (newline) (exit 1)))

;; Verify function cross-module access
(define fb (module-ref mod-b 'b-func))
(define result (fb 5))
(if (= result 11)
    (begin (display "PASS: cross_module_b b-func(5) is 11") (newline))
    (begin (display "FAIL: cross_module_b b-func(5) is ") (display result) (newline) (exit 1)))

(display "CROSS-MODULE-PASS") (newline)
