;; Utils-wide loading driver for TeXmacs.

(set! %load-path (cons "/Users/pikachu/pr/texmacs/TeXmacs/progs" %load-path))
(set! %load-path (cons "/Users/pikachu/pr/texmacs/TeXmacs/fonts" %load-path))

;; Define bp, try-load and utils-files BEFORE loading kernel, so these
;; bindings live in load-kernel-module and survive module changes.
(define bp "/Users/pikachu/pr/texmacs/TeXmacs/progs/")

(define (try-load path)
  (display "Loading ") (display path) (newline)
  (catch #t
    (lambda () (load path))
    (lambda (key . args)
      (display "  ERROR: ") (display key)
      (display " ") (display args) (newline))))

;; Collected from: find progs/utils -name '*.scm' | sort
(define utils-files
  (list
    "utils/automate/auto-build.scm"
    "utils/automate/auto-edit.scm"
    "utils/automate/auto-kbd.scm"
    "utils/automate/auto-menu.scm"
    "utils/automate/auto-tmfs.scm"
    "utils/base/environment.scm"
    "utils/cas/cas-out.scm"
    "utils/cas/cas-rewrite.scm"
    "utils/cite/cite-sort-test.scm"
    "utils/cite/cite-sort.scm"
    "utils/edit/auto-close.scm"
    "utils/edit/selections.scm"
    "utils/edit/variants.scm"
    "utils/email/email-tmfs.scm"
    "utils/handwriting/handwriting.scm"
    "utils/library/cpp-wrap.scm"
    "utils/library/cursor.scm"
    "utils/library/length.scm"
    "utils/library/ptrees.scm"
    "utils/library/smart-table.scm"
    "utils/library/tree.scm"
    "utils/literate/lp-build.scm"
    "utils/literate/lp-edit.scm"
    "utils/literate/lp-menu.scm"
    "utils/misc/ai.scm"
    "utils/misc/artwork.scm"
    "utils/misc/doxygen.scm"
    "utils/misc/extern-demo.scm"
    "utils/misc/gui-keyboard.scm"
    "utils/misc/gui-utils.scm"
    "utils/misc/markup-funcs.scm"
    "utils/misc/tiles.scm"
    "utils/misc/tm-keywords.scm"
    "utils/misc/tooltip.scm"
    "utils/misc/translation-list.scm"
    "utils/misc/updater.scm"
    "utils/plugins/plugin-cmd.scm"
    "utils/plugins/plugin-convert.scm"
    "utils/plugins/plugin-eval.scm"
    "utils/relate/live-connection.scm"
    "utils/relate/live-document.scm"
    "utils/relate/live-menu.scm"
    "utils/relate/live-view.scm"
    "utils/relate/relate-menu.scm"
    "utils/test/test-convert.scm"
    "utils/test/test-latex-export.scm"))

;; Load kernel stubs and init chain (restores module on exit)
(load "/Users/pikachu/pr/pscm/test/module/texmacs/load_kernel.scm")

;; Load all utils files
(for-each (lambda (f) (try-load (string-append bp f))) utils-files)

(display "UTILS-LOADED") (newline)
