;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: ((3 . "pay gas bill"))
(acons 3 "pay gas bill" '())

;; CHECK: ((3 . "tidy bedroom") (3 . "pay gas bill"))
(define task-list '())
(set! task-list (acons 3 "pay gas bill" '()))
(acons 3 "tidy bedroom" task-list)

(define address-list (acons "mary" "34 Elm Road" (acons "james" "16 Bow Street" '())))
;; CHECK: (("mary" . "34 Elm Road") ("james" . "16 Bow Street"))
address-list
;; CHECK: (("mary" . "34 Elm Road") ("james" . "1a London Road"))
(assoc-set! address-list "james" "1a London Road")


(define address-list (acons "mary" "34 Elm Road" (acons "james" "1a London Road" '())))
;; CHECK: (("mary" . "34 Elm Road") ("james" . "1a London Road"))
address-list
;; CHECK: (("bob" . "11 Newington Avenue") ("mary" . "34 Elm Road") ("james" . "1a London Road"))
(assoc-set! address-list "bob" "11 Newington Avenue")
;; CHECK: (("mary" . "34 Elm Road") ("james" . "1a London Road"))
address-list
(set! address-list (assoc-set! address-list "bob" "11 Newington Avenue"))
;; CHECK: (("bob" . "11 Newington Avenue") ("mary" . "34 Elm Road") ("james" . "1a London Road"))
address-list

(set! address-list (assoc-remove! address-list "mary"))
;; CHECK: (("bob" . "11 Newington Avenue") ("james" . "1a London Road"))
address-list

(define address-list '())
(set! address-list (assq-set! address-list "mary" "11 Elm Street"))
(set! address-list (assq-set! address-list "mary" "57 Pine Drive"))
;; CHECK: (("mary" . "57 Pine Drive") ("mary" . "11 Elm Street"))
address-list
(set! address-list (assoc-remove! address-list "mary"))
;; CHECK: (("mary" . "11 Elm Street"))
address-list

(define capitals '(("New York" . "Albany")
                   ("Oregon"   . "Salem")
                   ("Florida"  . "Miami")))

;; CHECK: ("Oregon" . "Salem")
(assoc "Oregon" capitals)
;; CHECK: Salem
(assoc-ref capitals "Oregon")

(set! capitals (assoc-set! capitals "South Dakota" "Pierre"))
;; CHECK: (("South Dakota" . "Pierre") ("New York" . "Albany") ("Oregon" . "Salem") ("Florida" . "Miami"))
capitals

(set! capitals (assoc-set! capitals "Florida" "Tallahassee"))
;; CHECK: (("South Dakota" . "Pierre") ("New York" . "Albany") ("Oregon" . "Salem") ("Florida" . "Tallahassee"))
capitals

(set! capitals (assoc-remove! capitals "Oregon"))
;; CHECK: (("South Dakota" . "Pierre") ("New York" . "Albany") ("Florida" . "Tallahassee"))
capitals

;; Test assq function (uses eq? for comparison)
(define e '((a 1) (b 2) (c 3)))
;; CHECK: (a 1)
(assq 'a e)
;; CHECK: (b 2)
(assq 'b e)
;; CHECK: #f
(assq 'd e)

;; Test assq with symbol keys (eq? comparison)
(define sym-alist '((x 10) (y 20) (z 30)))
;; CHECK: (x 10)
(assq 'x sym-alist)
;; CHECK: (y 20)
(assq 'y sym-alist)
;; CHECK: #f
(assq 'w sym-alist)

;; Test assq vs assoc: assq uses eq? (pointer comparison), assoc uses equal? (deep comparison)
;; assq should return #f for list keys (because (list 'a) and '(a) are not eq?)
;; CHECK: #f
(assq (list 'a) '(((a)) ((b)) ((c))))
;; assoc should find it (because (list 'a) and '(a) are equal?)
;; CHECK: ((a))
(assoc (list 'a) '(((a)) ((b)) ((c))))

;; Test assq with numeric keys
(define num-alist '((1 "one") (2 "two") (3 "three")))
;; CHECK: (1 "one")
(assq 1 num-alist)
;; CHECK: (2 "two")
(assq 2 num-alist)
;; CHECK: #f
(assq 4 num-alist)
