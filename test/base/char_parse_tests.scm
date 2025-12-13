;; RUN: %pscm_cc --test %s | FileCheck %s

;; Test character literal parsing

;; Basic character literals
;; CHECK: #\a
#\a
;; CHECK: #\A
#\A
;; CHECK: #\1
#\1
;; CHECK: #\.
#\.
;; CHECK: #\!
#\!

;; Test named characters (case-insensitive)
;; CHECK: #\space
#\space
;; CHECK: #\space
#\Space
;; CHECK: #\space
#\SPACE
;; CHECK: #\newline
#\newline
;; CHECK: #\newline
#\Newline
;; CHECK: #\newline
#\NEWLINE
;; CHECK: #\tab
#\tab
;; CHECK: #\tab
#\Tab
;; CHECK: #\tab
#\TAB

;; Test quoted character literals
;; CHECK: #\a
'#\a
;; CHECK: #\A
'#\A
;; CHECK: #\space
'#\space
;; CHECK: #\space
'#\Space
;; CHECK: #\newline
'#\newline

;; Test quoted space character literal (#\ followed by space)
;; CHECK: #\space
'#\ 
;; CHECK: #\space
'#\ 

;; Test eqv? with quoted and unquoted character literals
;; CHECK: #t
(eqv? '#\ #\Space)
;; CHECK: #t
(eqv? #\space '#\Space)
;; CHECK: #t
(eqv? '#\space #\Space)
;; CHECK: #t
(eqv? '#\a #\a)
;; CHECK: #t
(eqv? '#\A #\A)

;; Test char? predicate with quoted characters
;; CHECK: #t
(char? '#\a)
;; CHECK: #t
(char? '#\space)
;; CHECK: #t
(char? '#\newline)
;; CHECK: #t
(char? '#\ )

;; Test character comparison with eqv?
;; CHECK: #t
(eqv? #\space #\Space)
;; CHECK: #t
(eqv? '#\space '#\Space)
;; CHECK: #t
(eqv? '#\ #\space)
;; CHECK: #t
(eqv? '#\ #\Space)

;; Test character in lists
;; CHECK: (#\a #\b #\c)
(list '#\a '#\b '#\c)
;; CHECK: (#\space #\newline #\tab)
(list '#\space '#\newline '#\tab)
;; CHECK: (#\space #\space)
(list '#\ #\Space)

;; Test character with special characters
;; CHECK: #\(
#\(
;; CHECK: #\)
#\)
;; CHECK: #\+
#\+
;; CHECK: #\-
#\-
;; CHECK: #\*
#\*
;; CHECK: #\/
#\/

;; Test quoted special characters
;; CHECK: #\(
'#\(
;; CHECK: #\)
'#\)
;; CHECK: #\+
'#\+
;; CHECK: #\-
'#\-

;; Test character equality with quoted characters
;; CHECK: #t
(eqv? '#\( #\()
;; CHECK: #t
(eqv? '#\) #\))
;; CHECK: #t
(eqv? '#\+ #\+)
;; CHECK: #t
(eqv? '#\- #\-)

