;; RUN: %pscm_cc --test %s | FileCheck %s

;; CHECK: ()
(list->vector '())