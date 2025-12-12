;; RUN: %pscm_cc --test %s | FileCheck %s


;; CHECK: 1
(let foo () 1)