---
layout: home

title: pscm
titleTemplate: a scheme language implementation

hero:
  name: PikachuHy's Scheme
  text: a scheme language implementation
  tagline: A lightweight Scheme implementation in C++ (~14k LOC), designed to drive TeXmacs
  actions:
    - theme: brand
      text: Get Started
      link: /cn/pscm_cc
    - theme: alt
      text: View on GitHub
      link: https://github.com/PikachuHy/pscm
features:
  - title: Complete Type System
    details: Unified SCM type supporting 19 data types including NONE, NIL, LIST, NUM, FLOAT, RATIO, CHAR, STR, SYM, BOOL, PROC, FUNC, CONT, MACRO, HASH_TABLE, VECTOR, PORT, PROMISE, MODULE
  - title: Continuation Support
    details: Full continuation support implemented with setjmp/longjmp, including call/cc and dynamic-wind
  - title: Tail Recursion Optimization
    details: Efficient tail call optimization using goto to reduce stack depth
  - title: Rich Built-in Functions
    details: Comprehensive library including list operations, numeric functions (including gcd/lcm), string/char manipulation, hash tables, vectors, ports, lazy evaluation primitives (delay/force), and macro debugging utilities (macroexpand-1, macroexpand)
  - title: Modular Architecture
    details: Clean modular design with separate files for each special form and built-in function category
  - title: Source Location Tracking
    details: Complete source position tracking (file, line, column) with enhanced error reporting and call stack tracing

---