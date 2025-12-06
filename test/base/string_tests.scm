;; RUN: %pscm_main -m REGISTER_MACHINE --test %s | FileCheck %s
;; RUN: %pscm_cc --test %s | FileCheck %s
;; RUN: %pscm_main --test %s | FileCheck %s

;; CHECK: "Hello World"
"Hello World"

'(#t => 'ok)

#t