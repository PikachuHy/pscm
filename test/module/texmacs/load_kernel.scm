;; Kernel-wide loading driver for TeXmacs.
;; Loads init-texmacs.scm first, then remaining kernel files not already
;; pulled in by the init chain (via inherit-modules etc.).
;; This is the foundation for the iterative fix cycle (Tasks 6-N).

;; Set up load path to find TeXmacs progs
(set! %load-path (cons "/Users/pikachu/pr/texmacs/TeXmacs/progs" %load-path))
(set! %load-path (cons "/Users/pikachu/pr/texmacs/TeXmacs/fonts" %load-path))

;; ----- stubs (from load_texmacs_init.scm) -----
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
(define-macro (debug-set! key val) '(noop))
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

;; Missing Guile 1.8 built-ins
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

;; tm-define module internals — global stubs take priority over
;; module-scoped values (which may have been modified during loading).
(define cur-props '())
(define cur-props-table (make-ahash-table))
(define cur-conds '())
(define tm-defined-table (make-ahash-table))
(define tm-defined-name (make-ahash-table))
(define tm-defined-module (make-ahash-table))
(define define-option-table (make-hash-table 100))

;; TeXmacs standard macros
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

;; More tm-define internal functions (registered in define-option-table)
(define (ctx-add-condition! kind opt) (noop))
(define (define-option-mode opt decl) decl)
(define (define-option-match opt decl) decl)
(define (define-option-require opt decl) decl)
(define (define-option-applicable opt decl) decl)
(define (compute-arguments decl) '())
(define (define-option-argument opt decl) decl)
(define (define-option-default opt decl) decl)
(define (define-option-proposals opt decl) decl)
(define (property-set! var prop what conds*) (noop))
(define (property var prop) #f)

;; More TeXmacs C++ functions needed by tm-preferences and beyond
(define (notify-preferences-booted) (noop))
(define (list-break lst pred)
  ;; Stub: split list into (take-while, drop-while) based on pred
  (if (null? lst) (values '() '())
      (let loop ((in lst) (out '()))
        (if (or (null? in) (not (pred (car in))))
            (values (reverse out) in)
            (loop (cdr in) (cons (car in) out))))))
(define (not-define-option? x) (not (keyword? x)))
(define (texmacs-error msg . args) (pscm-display "TEXMACS-ERROR: ") (pscm-display msg) (pscm-newline))

;; receive macro (SRFI-8)
(define-macro (receive vars expr . body)
  `(call-with-values (lambda () ,expr) (lambda ,vars ,@body)))
(define-macro (lazy-define module . names) '(noop))
(define-macro (texmacs-modes . l) ''texmacs-modes-override)

;; Plugin-related C++ stubs
(define (plugin-supports-math-input-ref) #f)
(define (plugin-supports-latex?) #f)
(define (plugin-supports-html?) #f)
(define (plugin-supports-pdf?) #f)
(define (plugin-supports-image?) #f)
(define (plugin-supports-xml?) #f)
(define (plugin-supports-bibtex?) #f)
(define (plugin-supports-search?) #f)
(define (plugin-supports-replace?) #f)
(define (plugin-name) "pscm")
(define (plugin-format) "")
(define (plugin-input) "")
(define (plugin-output) "")
(define (test-preference? key) #f)
(define (get-preference key . default) (if (null? default) "" (car default)))
(define (set-preference key val) (noop))
(define (cpp-get-custom-private-style) "")
(define (cpp-get-default-modes) "")
(define (cpp-get-style) "")
(define (cpp-get-user-init-file) "")
(define (cpp-get-default-documentation) "")
(define (cpp-get-user-preference-file) "")
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

;; ----- Guile built-in stubs needed by remaining kernel files -----
(define (cadddr x) (car (cdddr x)))

;; Macros from abbrevs.scm — needed by tm-define.scm and other kernel modules
(define == equal?)
(define (!= x y) (not (equal? x y)))
(define-macro (when cond? . body)
  `(if ,cond? (begin ,@body)))
(define-macro (unless cond? . body)
  `(if (not ,cond?) (begin ,@body)))
(define-macro (with var val . body)
  (if (or (pair? var) (null? var))
      `(apply (lambda ,var ,@body) ,val)
      `(let ((,var ,val)) ,@body)))
(define-macro (with-global var val . body)
  `(let ((,var ,val)) ,@body))
(define-macro (and-with var val . body)
  `(and ,val (let ((,var ,val)) ,@body)))
(define-macro (with-result result . body)
  `(let ((,result '())) ,@body (reverse ,result)))
(define-macro (repeat n . body)
  `(do ((i 0 (+ i 1))) ((= i ,n) (noop)) ,@body))
(define-macro (twice . body)
  `(begin ,@body ,@body))
(define-macro (for what . body)
  (let ((n (length what)))
    (cond ((= n 2)
           `(for-each (lambda (,(car what)) ,@body)
                      ,(cadr what)))
          ((= n 3)
           `(do ((,(car what) ,(cadr what) (+ ,(car what) 1)))
                ((>= ,(car what) ,(caddr what)) (noop))
              ,@body))
          ((= n 4)
           `(if (> ,(cadddr what) 0)
                (do ((,(car what) ,(cadr what) (+ ,(car what) ,(cadddr what))))
                    ((>= ,(car what) ,(caddr what)) (noop))
                  ,@body)
                (do ((,(car what) ,(cadr what) (+ ,(car what) ,(cadddr what))))
                    ((<= ,(car what) ,(caddr what)) (noop))
                  ,@body)))
          (else '(noop)))))

;; ----- Save original display/write before boot.scm redefines them -----
(define pscm-display display)
(define pscm-write write)
(define pscm-newline newline)

;; ----- try-load helper -----
(define (try-load path)
  (display "Loading ") (display path) (newline)
  (catch #t
    (lambda () (load path) (display "  OK") (newline))
    (lambda (key . args)
      (display "  ERROR: ") (display key)
      (display " ") (display args) (newline))))

;; ----- Step 1: Load init-texmacs.scm (bootstrap) -----
;; Save current module; init chain creates modules and switches context.
;; We must restore afterward so kernel-base and stubs are visible to try-load.
(define load-kernel-module (current-module))
(pscm-display "=== Loading init-texmacs.scm (bootstrap) ===") (pscm-newline)
(catch #t
  (lambda ()
    (load "/Users/pikachu/pr/texmacs/TeXmacs/progs/init-texmacs.scm")
    (pscm-display "INIT-LOADED") (pscm-newline))
  (lambda (key . args)
    (pscm-display "INIT-ERROR: ") (pscm-display key) (pscm-newline)
    (pscm-display "INIT-ARGS: ") (pscm-display args) (pscm-newline)))
(set-current-module load-kernel-module)

;; ----- Step 2: Load remaining kernel files -----
(pscm-display "") (pscm-newline)
(pscm-display "=== Loading remaining kernel files ===") (pscm-newline)

(define kernel-base "/Users/pikachu/pr/texmacs/TeXmacs/progs/")
(define kernel-files
  (list
    "kernel/library/tree.scm"
    "kernel/library/patch.scm"
    "kernel/library/iterator.scm"
    "kernel/library/content.scm"
    "kernel/boot/ahash-table.scm"
    "kernel/boot/prologue.scm"
    "kernel/regexp/regexp-test.scm"
    "kernel/old-gui/old-gui-factory.scm"
    "kernel/old-gui/old-gui-test.scm"
    "kernel/old-gui/old-gui-form.scm"
    "kernel/old-gui/old-gui-widget.scm"
    "kernel/logic/logic-test.scm"
    "kernel/gui/menu-convert.scm"
    "kernel/gui/menu-test.scm"
    "kernel/gui/menu-widget.scm"
    "kernel/gui/menu-define.scm"
    "kernel/gui/kbd-define.scm"
    "kernel/gui/kbd-handlers.scm"
    "kernel/gui/speech-define.scm"
    "kernel/gui/gui-markup.scm"))

(for-each (lambda (f) (try-load (string-append kernel-base f))) kernel-files)

;; ----- Done -----
(pscm-display "") (pscm-newline)
(pscm-display "KERNEL-LOADED") (pscm-newline)
