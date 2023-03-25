//
// Created by PikachuHy on 2023/2/23.
//

#pragma once
#include <ostream>

namespace pscm {
class Scheme;
class Number;
class Char;
class String;
class Symbol;
class Pair;
class Procedure;
class Function;
class Macro;
class Continuation;
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
  APPLY_LET,
  APPLY_LET_STAR,
  APPLY_LAMBDA,
  APPLY_QUOTE,
  APPLY_FOR_EACH,
  AFTER_EVAL_FOR_EACH_FIRST_EXPR,
  AFTER_EVAL_DEFINE_ARG,
  AFTER_EVAL_SET_ARG,
  AFTER_EVAL_COND_TEST,
  AFTER_EVAL_IF_PRED,
  AFTER_EVAL_AND_EXPR,
  AFTER_EVAL_OR_EXPR,
  AFTER_EVAL_CALL_WITH_VALUES_PRODUCER,
};
std::ostream& operator<<(std::ostream& out, const Label& pos);

class Cell {
public:
  typedef Cell (*ScmFunc)(Cell);
  typedef Cell (*ScmMacro)(Scheme&, Cell);

  Cell() {
    ref_count_++;
  };

  Cell(Number *num);
  Cell(Char *ch);
  Cell(String *str);
  Cell(Symbol *sym);
  Cell(Pair *pair);
  Cell(Function *f);
  Cell(Macro *f);
  Cell(const Procedure *proc);
  Cell(Continuation *cont);

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
    FUNCTION,    // c++ function
    PROCEDURE,   // scheme procedure
    MACRO,       //
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

  Pair *to_pair() const {
    return (Pair *)(data_);
  }

  bool is_sym() const {
    return tag_ == Tag::SYMBOL;
  }

  Symbol *to_symbol() const {
    return (Symbol *)(data_);
  }

  bool is_char() const {
    return tag_ == Tag::CHAR;
  }

  Char *to_char() const {
    return (Char *)(data_);
  }

  bool is_str() const {
    return tag_ == Tag::STRING;
  }

  String *to_str() const {
    return (String *)(data_);
  }

  bool is_num() const {
    return tag_ == Tag::NUMBER;
  }

  Number *to_number() const {
    return (Number *)(data_);
  }

  bool is_bool() const {
    return tag_ == Tag::BOOL;
  }

  bool to_bool() const {
    return data_ != nullptr;
  }

  bool is_func() const {
    return tag_ == Tag::FUNCTION;
  }

  Function *to_func() const {
    return (Function *)(data_);
  }

  bool is_macro() const {
    return tag_ == Tag::MACRO;
  }

  Macro *to_macro() const {
    return (Macro *)(data_);
  }

  bool is_proc() const {
    return tag_ == Tag::PROCEDURE;
  }

  Procedure *to_proc() const {
    return (Procedure *)data_;
  }

  bool is_cont() const {
    return tag_ == Tag::CONTINUATION;
  }

  Continuation *to_cont() const {
    return (Continuation *)data_;
  }

private:
  Cell(Tag tag, void *data)
      : tag_(tag)
      , data_(data) {
  }

private:
  friend bool operator==(const Cell& lhs, const Cell& rhs);
  friend bool operator==(const Cell& lhs, const Number& rhs);
  friend bool operator==(const Cell& lhs, const Symbol& rhs);
  int ref_count_ = 0;
  Tag tag_ = Tag::NONE;
  void *data_ = nullptr;
  friend class Scheme;
};

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

extern Cell nil;
extern Cell lambda;
extern Cell quote;
extern Cell for_each;
extern Cell apply;
} // namespace pscm
