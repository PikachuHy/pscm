;; RUN: %pscm_cc --test %s | FileCheck %s
;; Test locale string and symbol conversion functions
;; Note: These are C API functions, so we test their behavior indirectly
;; through Scheme functions that use them

;; Test 1: symbol->string (tests scm_symbol_to_string)
(define sym1 'hello)
;; CHECK: hello
(symbol->string sym1)

;; Test 2: string->symbol (tests scm_from_locale_symbol indirectly)
(define str1 "world")
(define sym2 (string->symbol str1))
;; CHECK-NEXT: "world"
(symbol->string sym2)

;; Test 3: symbol->string with special characters
(define sym3 'test-symbol)
;; CHECK-NEXT: "test-symbol"
(symbol->string sym3)

;; Test 4: string->symbol with special characters
(define str2 "test-string")
(define sym4 (string->symbol str2))
;; CHECK-NEXT: "test-string"
(symbol->string sym4)

;; Test 5: Round-trip conversion
(define original-sym 'round-trip)
(define converted-str (symbol->string original-sym))
(define converted-sym (string->symbol converted-str))
;; CHECK-NEXT: #t
(eq? original-sym converted-sym)

;; Test 6: Multiple conversions
(define sym5 (string->symbol "multiple"))
(define str3 (symbol->string sym5))
;; CHECK-NEXT: "multiple"
str3

;; Test 7: Empty string
(define empty-sym (string->symbol ""))
;; CHECK-NEXT: ""
(symbol->string empty-sym)

;; Test 8: String with numbers
(define num-sym (string->symbol "test123"))
;; CHECK-NEXT: "test123"
(symbol->string num-sym)

;; Test 9: String with spaces (should work)
(define space-str "test with spaces")
(define space-sym (string->symbol space-str))
;; CHECK-NEXT: "test with spaces"
(symbol->string space-sym)

;; Test 10: Case sensitivity
(define upper-sym (string->symbol "UPPER"))
(define lower-sym (string->symbol "lower"))
;; CHECK-NEXT: #f
(eq? upper-sym lower-sym)

;; Test 11: Symbol to string preserves case
;; CHECK-NEXT: "UPPER"
(symbol->string upper-sym)
;; CHECK-NEXT: "lower"
(symbol->string lower-sym)

