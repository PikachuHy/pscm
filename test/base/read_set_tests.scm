;; RUN: %pscm_cc --test %s | FileCheck %s
;; Test read-set! functionality

;; Test 1: Get current read options (default should match Guile 1.8)
;; Guile 1.8 default: (keywords #f positions)
(read-options-interface)
;; CHECK: (keywords #f positions)
;; Test 2: Set keywords option to prefix (using macro, no quote needed for keywords)
(read-set! keywords 'prefix)
;; CHECK-NEXT: (keywords prefix positions)

;; Test 3: Set keywords option to postfix
(read-set! keywords 'postfix)
;; CHECK-NEXT: (keywords postfix positions)

;; Test 4: Disable keywords option (set to #f)
(read-set! keywords #f)
;; CHECK-NEXT: (keywords #f positions)

;; Test 5: Enable copy option
;; Note: When copy is enabled, positions is automatically enabled (Guile 1.8 behavior)
;; Note: For boolean options, use read-enable instead of read-set! with #t
;; Note: In Guile, read-enable is a procedure and requires quotes, but our macro doesn't
;; For compatibility with Guile, we use quoted symbols here
(read-enable 'copy)
;; CHECK-NEXT: (keywords #f positions copy)

;; Test 6: Disable positions option using read-disable
;; Note: read-set! with #f doesn't disable boolean options in Guile 1.8
;; Use read-disable instead
(read-disable 'positions)
;; CHECK-NEXT: (keywords #f positions copy)

;; Test 7: Enable case-insensitive option
(read-enable 'case-insensitive)
;; CHECK-NEXT: (keywords #f case-insensitive positions copy)

;; Test 8: Reset to default (disable all boolean options, set keywords to #f)
(read-disable 'copy)
;; CHECK-NEXT: (keywords #f case-insensitive positions)
(read-disable 'case-insensitive)
;; CHECK-NEXT: (keywords #f positions)
(read-enable 'positions)
;; CHECK-NEXT: (keywords #f positions)
(read-set! keywords #f)
;; CHECK-NEXT: (keywords #f positions)

;; Test 9: read-enable (enable multiple options)
(read-enable 'copy 'positions)
;; CHECK-NEXT: (keywords #f positions copy)

;; Test 10: read-disable (disable options)
(read-disable 'copy)
;; CHECK-NEXT: (keywords #f positions)

;; Test 11: Verify keywords value is preserved when using read-enable/disable
(read-set! keywords 'prefix)
;; CHECK-NEXT: (keywords prefix positions)
(read-enable 'copy)
;; CHECK-NEXT: (keywords prefix positions copy)

(read-disable 'copy)
;; CHECK-NEXT: (keywords prefix positions)
