//
// Created by PikachuHy on 2023/2/23.
//

#pragma once
#include <ostream>
#include <vector>

namespace pscm {

struct SourceLocation {
  constexpr SourceLocation(const char *filename = __builtin_FILE(), const char *funcname = __builtin_FUNCTION(),
                           unsigned int linenum = __builtin_LINE())
      : filename(filename)
      , funcname(funcname)
      , linenum(linenum) {
  }

  const char *filename;
  const char *funcname;
  unsigned int linenum;

  std::string to_string() const;
};

class Scheme;
class Number;
class Char;
class String;
class Symbol;
class SymbolTable;
class Pair;
class Procedure;
class Function;
class Macro;
class Promise;
class Continuation;
class Port;
enum class Label {
  DONE,
  EVAL,
  APPLY,
  EVAL_ARGS,
  AFTER_EVAL_FIRST_ARG,
  AFTER_EVAL_OTHER_ARG,
  AFTER_EVAL_ARGS,
  AFTER_EVAL_OP,
  AFTER_APPLY,
  APPLY_FUNC,
  APPLY_PROC,
  APPLY_MACRO,
  APPLY_CONT,
  AFTER_APPLY_FUNC,
  AFTER_APPLY_PROC,
  AFTER_APPLY_MACRO,
  AFTER_APPLY_CONT,
  AFTER_EVAL_FIRST_EXPR,
  AFTER_EVAL_OTHER_EXPR,
  APPLY_APPLY, // call apply from scheme
  APPLY_DEFINE,
  APPLY_COND,
  APPLY_IF,
  APPLY_AND,
  APPLY_OR,
  APPLY_SET,
  APPLY_DELAY,
  APPLY_LAMBDA,
  APPLY_QUOTE,
  APPLY_QUASIQUOTE,
  APPLY_FOR_EACH,
  APPLY_MAP,
  APPLY_FORCE,
  APPLY_BEGIN,
  AFTER_EVAL_FOR_EACH_FIRST_EXPR,
  AFTER_EVAL_MAP_FIRST_EXPR,
  AFTER_EVAL_MAP_OTHER_EXPR,
  AFTER_EVAL_PROMISE,
  AFTER_EVAL_DEFINE_ARG,
  AFTER_EVAL_SET_ARG,
  AFTER_EVAL_COND_TEST,
  AFTER_EVAL_IF_PRED,
  AFTER_EVAL_AND_EXPR,
  AFTER_EVAL_OR_EXPR,
  AFTER_EVAL_CALL_WITH_VALUES_PRODUCER,
};
std::string to_string(Label label);
std::ostream& operator<<(std::ostream& out, const Label& pos);

class Cell {
public:
  typedef Cell (*ScmFunc)(Cell);
  typedef Cell (*ScmMacro)(Scheme&, SymbolTable *, Cell);
  using Vec = std::vector<Cell>;

  Cell() {
    ref_count_++;
  };

  Cell(Number *num);
  Cell(Char *ch);
  Cell(String *str);
  Cell(Symbol *sym);
  Cell(Pair *pair);
  Cell(Vec *pair);
  Cell(Function *f);
  Cell(Macro *f);
  Cell(const Procedure *proc);
  Cell(Promise *p);
  Cell(Continuation *cont);
  Cell(Port *port);
  explicit Cell(bool val);

  ~Cell() {
    ref_count_--;
  };
  enum class Tag {
    NONE,        // default value
    EXCEPTION,   // exception
    NIL,         // empty value
    BOOL,        // bool
    STRING,      // string
    CHAR,        // char
    NUMBER,      // number: int float complex
    SYMBOL,      // symbol
    PAIR,        // pair,
    VECTOR,      // vector, #(a b c)
    FUNCTION,    // c++ function
    PROCEDURE,   // scheme procedure
    MACRO,       //
    PROMISE,     // promise
    PORT,        // port
    CONTINUATION // continuation
  };
  friend std::ostream& operator<<(std::ostream& out, const Cell& cell);
  std::string to_string() const;

  static Cell nil() {
    static Cell ret{ Tag::NIL, nullptr };
    return ret;
  }

  static Cell none() {
    static Cell ret{ Tag::NONE, nullptr };
    return ret;
  }

  static Cell ex(const char *msg);

  static Cell bool_true() {
    static Cell ret{ Tag::BOOL, nullptr };
    ret.data_ = (void *)&ret;
    return ret;
  }

  static Cell bool_false() {
    static Cell ret{ Tag::BOOL, nullptr };
    return ret;
  }

  bool is_none() const {
    return tag_ == Tag::NONE;
  }

  bool is_nil() const {
    return tag_ == Tag::NIL;
  }

  bool is_pair() const {
    return tag_ == Tag::PAIR;
  }

  Pair *to_pair(SourceLocation loc = {}) const;

  bool is_vec() const {
    return tag_ == Tag::VECTOR;
  }

  Vec *to_vec(SourceLocation loc = {}) const;

  bool is_sym() const {
    return tag_ == Tag::SYMBOL;
  }

  Symbol *to_symbol(SourceLocation loc = {}) const;

  bool is_char() const {
    return tag_ == Tag::CHAR;
  }

  Char *to_char(SourceLocation loc = {}) const;

  bool is_str() const {
    return tag_ == Tag::STRING;
  }

  String *to_str(SourceLocation loc = {}) const;

  bool is_num() const {
    return tag_ == Tag::NUMBER;
  }

  Number *to_number(SourceLocation loc = {}) const;

  bool is_bool() const {
    return tag_ == Tag::BOOL;
  }

  bool to_bool(SourceLocation loc = {}) const;

  bool is_func() const {
    return tag_ == Tag::FUNCTION;
  }

  Function *to_func(SourceLocation loc = {}) const;

  bool is_macro() const {
    return tag_ == Tag::MACRO;
  }

  Macro *to_macro(SourceLocation loc = {}) const;

  bool is_proc() const {
    return tag_ == Tag::PROCEDURE;
  }

  Procedure *to_proc(SourceLocation loc = {}) const;

  bool is_promise() const {
    return tag_ == Tag::PROMISE;
  }

  Promise *to_promise(SourceLocation loc = {}) const;

  bool is_cont() const {
    return tag_ == Tag::CONTINUATION;
  }

  Continuation *to_cont(SourceLocation loc = {}) const;

  bool is_port() const {
    return tag_ == Tag::PORT;
  }

  Port *to_port(SourceLocation loc = {}) const;

  bool is_self_evaluated() const;

  [[nodiscard]] Cell is_eqv(const Cell& rhs) const;
  [[nodiscard]] Cell is_eq(const Cell& rhs) const;

private:
  Cell(Tag tag, void *data)
      : tag_(tag)
      , data_(data) {
  }

private:
  friend bool operator==(const Cell& lhs, const Cell& rhs);
  friend bool operator==(const Cell& lhs, const Number& rhs);
  friend bool operator==(const Cell& lhs, const Symbol& rhs);
  friend bool operator==(const Cell& lhs, const Vec& rhs);
  friend bool operator==(const Cell& lhs, const String& rhs);
  int ref_count_ = 0;
  Tag tag_ = Tag::NONE;
  void *data_ = nullptr;
  friend class Scheme;
};

extern Cell nil;
extern Cell lambda;
extern Cell quote;
extern Cell unquote;
extern Cell quasiquote;
extern Cell unquote_splicing;
extern Cell begin;
extern Cell builtin_for_each;
extern Cell builtin_map;
extern Cell builtin_force;
extern Cell apply;
} // namespace pscm
