//
// Created by PikachuHy on 2023/2/23.
//

#pragma once
#include "compat.h"
#include "pscm/misc/SourceLocation.h"
#include <cstdint>
#include <ostream>
#include <vector>

namespace pscm {
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
class Module;
class HashTable;
class Keyword;
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
  AFTER_APPLY_USER_DEFINED_MACRO,
  AFTER_APPLY_CONT,
  AFTER_EVAL_FIRST_EXPR,
  AFTER_EVAL_OTHER_EXPR,
  APPLY_APPLY, // call apply from scheme
  APPLY_EVAL,  // call eval from scheme
  APPLY_DEFINE,
  APPLY_DEFINE_MACRO,
  APPLY_DEFINE_MODULE,
  APPLY_IS_DEFINED,
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
  APPLY_LOAD,
  APPLY_CURRENT_MODULE,
  APPLY_USE_MODULES,
  APPLY_RESOLVE_MODULE,
  APPLY_EXPORT,
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
  AFTER_APPLY_EVAL,
  TODO
};
const UString to_string(Label label);
std::ostream& operator<<(std::ostream& out, const Label& pos);

class SmallObject {
public:
  SmallObject(long tag, void *data)
      : tag(tag)
      , data(data) {
  }

  UString to_string() const;
  long tag;
  void *data;
};

#define PSCM_CELL_TYPE(Type, type, tag)                                                                                \
  Cell(Type *t, SourceLocation loc = {});                                                                              \
  bool is_##type() const {                                                                                             \
    return tag_ == Tag::tag;                                                                                           \
  }                                                                                                                    \
  Type *to_##type(SourceLocation loc = {}) const
class SchemeProxy;
using HashCodeType = std::uint32_t;

class Cell {
public:
  typedef Cell (*ScmFunc)(Cell);
  typedef Cell (*ScmMacro)(Scheme&, SymbolTable *, Cell);
  typedef Cell (*ScmMacro2)(SchemeProxy, SymbolTable *, Cell);
  typedef bool (*ScmCmp)(Cell, Cell);
  using Vec = std::vector<Cell>;

  Cell() {
    ref_count_++;
  };

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
    SMOB,        // guile small object
    MODULE,      // module
    HASH_TABLE,  // hash table
    KEYWORD,     // keyword
    CONTINUATION // continuation
  };
  UString to_string() const;
  UString pretty_string() const;
  void display(Port& port);

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

  bool is_ex() const {
    return tag_ == Tag::EXCEPTION;
  }

  bool is_bool() const {
    return tag_ == Tag::BOOL;
  }

  bool to_bool(SourceLocation loc = {}) const;

  PSCM_CELL_TYPE(Keyword, keyword, KEYWORD);
  PSCM_CELL_TYPE(Pair, pair, PAIR);
  PSCM_CELL_TYPE(String, str, STRING);
  PSCM_CELL_TYPE(Symbol, sym, SYMBOL);
  PSCM_CELL_TYPE(Char, char, CHAR);
  PSCM_CELL_TYPE(Vec, vec, VECTOR);
  PSCM_CELL_TYPE(Number, num, NUMBER);
  PSCM_CELL_TYPE(Macro, macro, MACRO);
  PSCM_CELL_TYPE(Procedure, proc, PROCEDURE);
  PSCM_CELL_TYPE(Function, func, FUNCTION);
  PSCM_CELL_TYPE(Promise, promise, PROMISE);
  PSCM_CELL_TYPE(Continuation, cont, CONTINUATION);
  PSCM_CELL_TYPE(Port, port, PORT);
  PSCM_CELL_TYPE(Module, module, MODULE);
  PSCM_CELL_TYPE(SmallObject, smob, SMOB);
  PSCM_CELL_TYPE(HashTable, hash_table, HASH_TABLE);

  bool is_self_evaluated() const;

  [[nodiscard]] Cell is_eqv(const Cell& rhs) const;
  [[nodiscard]] Cell is_eq(const Cell& rhs) const;

  HashCodeType hash_code() const;
  static bool is_eq(Cell lhs, Cell rhs);
  static bool is_eqv(Cell lhs, Cell rhs);
  static bool is_equal(Cell lhs, Cell rhs);

  UString source_location() const {
    return loc_.to_string();
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
  friend bool operator==(const Cell& lhs, const Vec& rhs);
  friend bool operator==(const Cell& lhs, const String& rhs);
  int ref_count_ = 0;
  Tag tag_ = Tag::NONE;
  void *data_ = nullptr;
  SourceLocation loc_;
  friend class Scheme;
};

// hack: force load code in AList.cpp
// do not use the class
class AList {
public:
  AList();
};

extern Cell nil;
extern Cell quote;
extern Cell lambda;
extern Cell apply;
} // namespace pscm

namespace std {
template <>
struct hash<pscm::Cell> {
  using result_type = std::size_t;
  using argument_type = pscm::Cell;
  std::size_t operator()(const pscm::Cell& rhs) const;
};

} // namespace std
