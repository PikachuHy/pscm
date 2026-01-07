;; RUN: %pscm_cc --test %s | FileCheck %s
;; Test debug-set! functionality

;; Test 1: Get current debug options (default)
(debug-options-interface)
;; CHECK: (show-file-name #t stack 20000 depth 20 maxdepth 1000 frames 3 indent 10 width 79 procnames cheap)

;; Test 2: Set stack option to a different value
(debug-set! stack 200000)
;; CHECK-NEXT: (show-file-name #t stack 200000 depth 20 maxdepth 1000 frames 3 indent 10 width 79 procnames cheap)

;; Test 3: Set frames option
(debug-set! frames 5)
;; CHECK-NEXT: (show-file-name #t stack 200000 depth 20 maxdepth 1000 frames 5 indent 10 width 79 procnames cheap)

;; Test 4: Set width option
(debug-set! width 100)
;; CHECK-NEXT: (show-file-name #t stack 200000 depth 20 maxdepth 1000 frames 5 indent 10 width 100 procnames cheap)

;; Test 5: Enable backtrace option using debug-enable
(debug-enable 'backtrace)
;; CHECK-NEXT: (show-file-name #t stack 200000 backtrace depth 20 maxdepth 1000 frames 5 indent 10 width 100 procnames cheap)

;; Test 6: Enable trace option
(debug-enable 'trace)
;; CHECK-NEXT: (show-file-name #t stack 200000 backtrace depth 20 maxdepth 1000 frames 5 indent 10 width 100 procnames trace cheap)

;; Test 7: Disable backtrace option using debug-disable
(debug-disable 'backtrace)
;; CHECK-NEXT: (show-file-name #t stack 200000 depth 20 maxdepth 1000 frames 5 indent 10 width 100 procnames trace cheap)

;; Test 8: Disable trace option
(debug-disable 'trace)
;; CHECK-NEXT: (show-file-name #t stack 200000 depth 20 maxdepth 1000 frames 5 indent 10 width 100 procnames cheap)

;; Test 9: Set show-file-name to #f
(debug-set! show-file-name #f)
;; CHECK-NEXT: (show-file-name #f stack 200000 depth 20 maxdepth 1000 frames 5 indent 10 width 100 procnames cheap)

;; Test 10: Set show-file-name back to #t
(debug-set! show-file-name #t)
;; CHECK-NEXT: (show-file-name #t stack 200000 depth 20 maxdepth 1000 frames 5 indent 10 width 100 procnames cheap)

;; Test 11: debug-enable with multiple options
(debug-enable 'backtrace 'debug)
;; CHECK-NEXT: (show-file-name #t stack 200000 debug backtrace depth 20 maxdepth 1000 frames 5 indent 10 width 100 procnames cheap)

;; Test 12: debug-disable with multiple options
(debug-disable 'backtrace 'debug)
;; CHECK-NEXT: (show-file-name #t stack 200000 depth 20 maxdepth 1000 frames 5 indent 10 width 100 procnames cheap)

;; Test 13: Set depth option
(debug-set! depth 30)
;; CHECK-NEXT: (show-file-name #t stack 200000 depth 30 maxdepth 1000 frames 5 indent 10 width 100 procnames cheap)

;; Test 14: Set maxdepth option
(debug-set! maxdepth 2000)
;; CHECK-NEXT: (show-file-name #t stack 200000 depth 30 maxdepth 2000 frames 5 indent 10 width 100 procnames cheap)

;; Test 15: Set indent option
(debug-set! indent 15)
;; CHECK-NEXT: (show-file-name #t stack 200000 depth 30 maxdepth 2000 frames 5 indent 15 width 100 procnames cheap)

;; Test 16: Verify that non-boolean options preserve their values when using debug-enable/disable
(debug-set! stack 50000)
;; CHECK-NEXT: (show-file-name #t stack 50000 depth 30 maxdepth 2000 frames 5 indent 15 width 100 procnames cheap)
(debug-enable 'backtrace)
;; CHECK-NEXT: (show-file-name #t stack 50000 backtrace depth 30 maxdepth 2000 frames 5 indent 15 width 100 procnames cheap)
(debug-disable 'backtrace)
;; CHECK-NEXT: (show-file-name #t stack 50000 depth 30 maxdepth 2000 frames 5 indent 15 width 100 procnames cheap)

