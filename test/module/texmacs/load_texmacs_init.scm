;; Incremental loading test for TeXmacs init sequence.
;; This harness loads init-texmacs.scm and reports success or the first error.

;; Set up load path to find TeXmacs progs
(set! %load-path (cons "/Users/pikachu/pr/texmacs/TeXmacs/progs" %load-path))
(set! %load-path (cons "/Users/pikachu/pr/texmacs/TeXmacs/fonts" %load-path))

;; TeXmacs init needs C++-side functions. Define stubs:
(define (cpp-get-preference key default) default)
(define (os-mingw?) #f)
(define (os-macos?) #t)
(define (texmacs-time) 0)
(define (tm-interactive) #f)
(define (scheme-dialect) "guile-a")
(define (supports-email?) #f)
(define (symbol-property sym key) #f)
(define (set-symbol-property! sym key val) (noop))
(define (source-property form key)
  (if (eq? key 'line) 0
      (if (eq? key 'column) 0
          (if (eq? key 'filename) "unknown" #f))))
(define (debug-set! key val) (noop))
(define (debug-enable . args) (noop))
(define module-export! export)
(define (with-fluids fluids thunk) (thunk))

;; Try loading the init file
(catch #t
  (lambda ()
    (load "/Users/pikachu/pr/texmacs/TeXmacs/progs/init-texmacs.scm")
    (display "INIT-LOADED") (newline))
  (lambda (key . args)
    (display "ERROR-KEY: ") (display key) (newline)
    (display "ERROR-ARGS: ") (display args) (newline)))
