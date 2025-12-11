;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test parsing of symbols containing dots (like ...)
;; These should be parsed as single symbols, not as dotted pairs

;; CHECK: ...
'...

;; CHECK: ..
'..

;; CHECK: !..
'!..

;; CHECK: $.+
'$.+

;; CHECK: %-
'%-

;; CHECK: &.!
'&.!

;; CHECK: *.:
'*.:

;; CHECK: /:.
'/:. 

;; CHECK: #:+.
':+.

;; CHECK: <-
'<-

;; CHECK: =.
'=.

;; CHECK: >.
'>.

;; CHECK: ?.
'?.

;; CHECK: ~.
'~.

;; CHECK: _.
'_.

;; CHECK: ^.
'^.

;; Test that symbols with dots can be in lists
;; CHECK: (+ - ... !..)
'(+ - ... !..)

;; CHECK: (+ - ... !.. $.+ %.- &.! *.: /:. #:+. <-. =. >. ?. ~. _. ^.)
'(+ - ... !.. $.+ %.- &.! *.: /:. :+. <-. =. >. ?. ~. _. ^.)

;; Test that dotted pairs still work correctly
;; CHECK: (a . b)
'(a . b)

;; CHECK: (a b . c)
'(a b . c)

;; CHECK: (a b c d)
'(a b . (c d))

;; Test that symbols with dots can be used in expressions
(define ellipsis '...)
;; CHECK: ...
ellipsis

(define dot-dot '..)
;; CHECK: ..
dot-dot

(define bang-dot-dot '!..)
;; CHECK: !..
bang-dot-dot

;; Test that symbols with dots can be compared
;; CHECK: #t
(eqv? '... '...)

;; CHECK: #f
(eqv? '... '..)

;; CHECK: #t
(eqv? '!.. '!..)

;; CHECK: #t
(eqv? ':+. ':+.)

;; Test that symbols with dots work in lists with other elements
(define test-list '(+ - ... !..))
;; CHECK: (+ - ... !..)
test-list

;; CHECK: ...
(caddr test-list)

;; CHECK: !..
(car (cdr (cdr (cdr test-list))))

;; Test accessing fourth element using cdr chain
(define fourth (car (cdr (cdr (cdr test-list)))))
;; CHECK: !..
fourth

;; Test edge cases: symbol starting with dot
;; CHECK: .a
'.a

;; CHECK: .+
'.+

;; CHECK: .-
'.-

;; Test that multiple dots in a row work
;; CHECK: ....
'....

;; CHECK: ..+
'..+

;; CHECK: +..
'+..

;; CHECK: +..+
'+..+

;; Test that dots with other special characters work
;; CHECK: !.+
'!.+

;; CHECK: $..
'$..

;; CHECK: %..%
'%..%

;; Test that these symbols can be used as identifiers
(define ... 'ellipsis)
;; CHECK: ellipsis
...

(define !.. 'bang-dot-dot)
;; CHECK: bang-dot-dot
!..

;; Test that dotted pair syntax is not confused with symbols containing dots
;; CHECK: (a . b)
'(a . b)

;; CHECK: (a ... b)
'(a ... b)

;; CHECK: (a . ...)
'(a . ...)

;; CHECK: (... . b)
'(... . b)

;; CHECK: (... . ...)
'(... . ...)
