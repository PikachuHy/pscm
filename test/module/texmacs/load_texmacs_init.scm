;; Incremental loading test for TeXmacs init sequence.
;; This harness loads init-texmacs.scm and reports success or the first error.

;; Set up load path to find TeXmacs progs
(set! %load-path (cons "/Users/pikachu/pr/texmacs/TeXmacs/progs" %load-path))
(set! %load-path (cons "/Users/pikachu/pr/texmacs/TeXmacs/fonts" %load-path))

;; TeXmacs init needs C++-side functions. Define stubs:
(define (cpp-get-preference key default) default)
(define (os-mingw?) #f)
(define (os-macos?) #t)
(define (os-win32?) #f)
(define (os-linux?) #f)
(define (os-gnu?) #f)
(define (texmacs-time) 0)
(define (tm-interactive) #f)
(define (scheme-dialect) "guile-b")
(define (supports-email?) #f)
(define (symbol-property sym key) #f)
(define (set-symbol-property! sym key val) (noop))
(define (source-property form key)
  (if (eq? key 'line) 0
      (if (eq? key 'column) 0
          (if (eq? key 'filename) "unknown" #f))))
(define (debug-set! key val) (noop))
(define (debug-enable . args) (noop))
(define (url-concretize path)
  ;; Stub: returns the path with $TEXMACS_PATH expanded
  "/Users/pikachu/pr/texmacs/TeXmacs/progs/kernel/boot/boot.scm")
(define (gui-version) "qt5")
(define (module-export! . args) (noop))
(define (with-fluids fluids thunk) (thunk))
(define (selection-active-any?) #f)
(define (window-visible?) #f)
(define (get-output-tree . args) "")
(define (cpp-string->object s) s)
(define (object->string obj) (if (string? obj) obj ""))
(define (get-boolean-preference key) #f)
(define (get-string-preference key) "")
(define (cpp-get-default-library) "")
(define (get-document-language) "english")
(define (get-user-language) "english")
(define (tree->stree t) t)
;; Missing Guile 1.8 built-ins (from audit)
(define (reverse! lst) (reverse lst))
(define (caar x) (car (car x)))
(define (cdar x) (cdr (car x)))
(define (caaar x) (car (caar x)))
(define (caadr x) (car (cadr x)))
(define (cadar x) (car (cdar x)))
(define (cdaar x) (cdr (caar x)))
(define (cdadr x) (cdr (cadr x)))
(define (cddar x) (cdr (cdar x)))
(define (cdddr x) (cdr (cddr x)))
(define (ca*r x) (if (pair? x) (ca*r (car x)) x))
(define (ca*adr x) (ca*r (cadr x)))
(define (make-ahash-table) (make-hash-table))
(define (ahash-ref t k) (hash-ref t k))
(define (ahash-set! t k v) (hash-set! t k v))
(define (ahash-remove! t k) (hash-remove! t k))
;; tm-define module internals (not visible during macro expansion in pscm)
(define cur-props '())
(define cur-props-table (make-ahash-table))
(define cur-conds '())
(define tm-defined-table (make-ahash-table))
(define tm-defined-name (make-ahash-table))
(define tm-defined-module (make-ahash-table))
(define define-option-table (make-hash-table 100))
;; TeXmacs standard macros (C++ functions)
(define (tm? x) #f)
(define (tree-atomic? x) #f)
(define (tree-compound? x) #f)
(define (tree-in? t u) #f)
(define (tree-label t) "")
(define (tree-children t) '())
(define (tree->object t) "")
(define (string-replace s old new)
  ;; Stub: simple string replacement
  (if (string=? s old) new s))
(define (compile-interface-spec spec) spec)
(define (process-use-modules specs) (for-each (lambda (s) (eval `(use-modules ,s))) specs))
(define (string-append-suffix str suffix) (string-append str suffix))
(define (string-append-prefix prefix str) (string-append prefix str))
(define (select . args) (noop))
(define (tree-search-upwards t p) #f)
(define (== a b) (equal? a b))
(define (!= a b) (not (equal? a b)))
(define (tm-output str)
  ;; Must not call display/write (boot.scm redefines them to call tm-output).
  ;; For test purposes, noop is sufficient - we just need init to not crash.
  (noop))
(define (lambda* head body)
  ;; Stub: TeXmacs lambda* is a recursive lambda builder
  (if (pair? head) (lambda* (car head) `((lambda ,(cdr head) ,@body))) (car body)))
;; More tm-define.scm internal functions
(define (property-rewrite l)
  `(property-set! ,@l (list ,@cur-conds)))
(define (nnull? x) (not (null? x)))
(define (module-name mod)
  ;; Stub: return a module's name as a list of symbols, Guile 1.8 style
  (if (module? mod) '() '(not-a-module)))
(define (filter-conds l)
  ;; Stub: tm-define internal — filter condition list
  (if (null? l) '() (cons (car l) (filter-conds (cdr l)))))
(define (unlambda pred?)
  ;; Stub: tm-define internal — check if a symbol is unlambda
  (and (symbol? pred?) (not (null? (string->list (symbol->string pred?))))))
(define lazy-define-table (make-ahash-table))
(define (and-apply l args)
  ;; Stub: tm-define internal — apply all predicates
  (or (null? l) (and (apply (car l) (or args '())) (and-apply (cdr l) args))))
(define (listify args)
  ;; Stub: tm-define internal — convert anything to list
  (if (pair? args) (cons (car args) (listify (cdr args))) (if (null? args) '() (list args))))
(define (apply* fun head)
  ;; Stub: tm-define internal
  (if (pair? head) (apply fun head) (fun head)))
(define (and* conds)
  ;; Stub: tm-define internal — and for conditions
  (if (null? conds) #t (and (car conds) (and* (cdr conds)))))
(define (begin* conds)
  ;; Stub: tm-define internal — begin for conditions
  (if (null? conds) #t (car conds)))
(define (list-2? x) (and (pair? x) (pair? (cdr x)) (null? (cddr x))))
(define (list-3? x) (and (pair? x) (pair? (cdr x)) (pair? (cddr x)) (null? (cdddr x))))
(define (display-to-string obj)
  ;; Stub: convert any object to string for TeXmacs display
  (if (string? obj) obj (object->string obj)))
(define (keyword? x)
  ;; Stub: pscm parses #:keyword syntax but doesn't have keyword? built-in
  (and (symbol? x)
       (let ((s (symbol->string x)))
         (and (> (string-length s) 0)
              (char=? (string-ref s 0) #\:)))))

;; Save original display before boot.scm redefines it
(define pscm-display display)
(define pscm-write write)
(define pscm-newline newline)

;; Try loading the init file
(catch #t
  (lambda ()
    (load "/Users/pikachu/pr/texmacs/TeXmacs/progs/init-texmacs.scm")
    ;; Use saved originals because boot.scm redefines display/write
    (pscm-display "INIT-LOADED") (pscm-newline))
  (lambda (key . args)
    (pscm-display "ERROR-KEY: ") (pscm-display key) (pscm-newline)
    (pscm-display "ERROR-ARGS: ") (pscm-display args) (pscm-newline)))
