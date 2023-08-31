//
// Created by PikachuHy on 2023/2/23.
//
#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/Cell.h"
#include "pscm/ApiManager.h"
#include "pscm/Char.h"
#include "pscm/Continuation.h"
#include "pscm/Function.h"
#include "pscm/HashTable.h"
#include "pscm/Keyword.h"
#include "pscm/Macro.h"
#include "pscm/Module.h"
#include "pscm/Number.h"
#include "pscm/Pair.h"
#include "pscm/Port.h"
#include "pscm/Procedure.h"
#include "pscm/Promise.h"
#include "pscm/Str.h"
#include "pscm/Symbol.h"
#include "pscm/common_def.h"
#include "pscm/logger/Logger.hpp"
#include "pscm/misc/ICUCompat.h"
#include "unicode/ustream.h"
#include "unicode/schriter.h"
#include <cassert>
#include <cstring>
#include <spdlog/fmt/fmt.h>
#include <sstream>
#include <unordered_set>
#endif
namespace pscm {
Cell nil = Cell::nil();
PSCM_INLINE_LOG_DECLARE("pscm.core.Cell");

UString SmallObject::to_string() const{
  UString out;
  out += '<';
  out += "smob";
  out += ' ';
  out += pscm::to_string(tag);
  out += ' ';
  out += pscm::to_string(data);
  out += '>';
  return out;
}

Cell::Cell(bool val) {
  tag_ = Tag::BOOL;
  if (val) {
    data_ = this;
  }
  else {
    data_ = nullptr;
  }
}

UString Cell::to_string() const {
  if (tag_ == Cell::Tag::NONE) {
    return "NONE";
  }
  if (tag_ == Cell::Tag::EXCEPTION) {
    UString res("EXCEPTION: ");
    res += (const char *)data_;
    return res;
  }
  if (tag_ == Cell::Tag::NIL) {
    return "()";
  }
  if (tag_ == Cell::Tag::SYMBOL) {
    return to_sym()->to_string();
  }
  if (tag_ == Cell::Tag::FUNCTION) {
    return to_func()->to_string();
  }
  if (tag_ == Cell::Tag::MACRO) {
    return to_macro()->to_string();
  }
  if (tag_ == Cell::Tag::NUMBER) {
    return to_num()->to_string();
  }
  if (tag_ == Cell::Tag::CHAR) {
    return to_char()->to_string();
  }
  if (tag_ == Cell::Tag::STRING) {
    return to_str()->str();
  }
  if (tag_ == Cell::Tag::BOOL) {
    if (to_bool()) {
      return "#t";
    }
    else {
      return "#f";
    }
  }
  if (tag_ == Cell::Tag::PAIR) {
    UString out;
    out += "(";
    // TODO:
    std::unordered_set<Pair *> p_set;
    int num = 0;
    auto p = to_pair();
    p_set.insert(p);
    while (p) {
      out += p->first.to_string();
      if (p->second.is_pair()) {
        out += " ";
        p = p->second.to_pair();
        if (p_set.find(p) != p_set.end()) {
          out += ". ";
          out += "#" + pscm::to_string(num) + "#";
          break;
        }
      }
      else if (p->second.is_nil()) {
        break;
      }
      else {
        out += " . ";
        if (p->second.is_pair()) {
          auto p2 = p->second.to_pair();
          if (p_set.find(p2) != p_set.end()) {
            out += ". ";
            out += "#" + pscm::to_string(num) + "#";
            break;
          }
        }
        out += p->second.to_string();
        break;
      }
    }
    out += ")";
    return out;
  }
  if (tag_ == Cell::Tag::VECTOR) {
    UString out;
    out += "#(";
    auto vec = to_vec();
    if (!vec->empty()) {
      for (int i = 0; i < vec->size() - 1; ++i) {
        out += vec->at(i).to_string() + " ";
      }
      out += vec->back().to_string();
    }
    out += ")";
    return out;
  }
  if (tag_ == Cell::Tag::PROCEDURE) {
    return to_proc()->to_string();
  }
  if (tag_ == Cell::Tag::PROMISE) {
    return to_promise()->to_string();
  }
  if (tag_ == Cell::Tag::CONTINUATION) {
    return to_cont()->to_string();
  }
  if (tag_ == Cell::Tag::PORT) {
    return to_port()->to_string();
  }
  if (tag_ == Cell::Tag::MODULE) {
    return to_module()->to_string();
  }
  if (tag_ == Cell::Tag::HASH_TABLE) {
    return to_hash_table()->to_string();
  }
  if (tag_ == Cell::Tag::KEYWORD) {
    return to_keyword()->to_string();
  }
  if (tag_ == Cell::Tag::SMOB) {
    return to_smob()->to_string();
  }
  PSCM_ERROR("TODO: {0}", int(tag_));
  //  PSCM_THROW_EXCEPTION("TODO: cell tag ");
  return "TODO";
}

UString Cell::pretty_string() const {
  if (is_none()) {
    return "";
  }
  if (is_nil()) {
    return "()";
  }
  if (is_bool()) {
    if (to_bool()) {
      return "#t";
    }
    else {
      return "#f";
    }
  }
  if (is_sym()) {
    auto sym = to_sym();
    if (sym->name().indexOf(' '_u) != -1) {
      UString ss;
      ss += "\e[;34m";
      ss += to_string();
      ss += "\e[0m";
      return ss;
    }
    else {
      UString ss;
      ss += to_string();
      return ss;
    }
  }
  if (is_macro()) {
    auto macro = to_macro();
    auto name = macro->name();
    UString ss;
    ss += "\e[;35m";
    ss += name;
    ss += "\e[0m";
    return ss;
  }
  if (is_func()) {
    auto func = to_func();
    return func->name();
  }
  if (is_pair()) {
    UString ss;
    if (car(*this).is_macro()) {
      auto macro = car(*this).to_macro();
      if (macro->name() == "quote") {
        ss += "'";
        ss += cadr(*this).pretty_string();
        return ss;
      }
      else if (macro->name() == "quasiquote") {
        ss += "`";
        ss += cadr(*this).pretty_string();
        return ss;
      }
      else if (macro->name() == "unquote") {
        ss += ",";
        ss += cadr(*this).pretty_string();
        return ss;
      }
    }
    else if (car(*this).is_sym()) {
      auto sym = car(*this).to_sym();
      if (sym->name() == "quasiquote") {
        ss += "`";
        ss += cadr(*this).pretty_string();
        return ss;
      }
      else if (sym->name() == "unquote") {
        ss += ",";
        ss += cadr(*this).pretty_string();
        return ss;
      }
      else if (sym->name() == "unquote-splicing") {
        ss += ",@";
        ss += cadr(*this).pretty_string();
        return ss;
      }
    }

    ss += '(';
    auto it = *this;
    ss += car(it).pretty_string();
    it = cdr(it);
    while (it.is_pair()) {
      ss += " ";
      if (car(it).is_sym() && car(it).to_sym()->name() == "unquote") {
        if (cdr(it).is_pair() && cddr(it).is_nil()) {
          ss += ".";
          ss += " ";
          ss += ",";
          ss += cadr(it).pretty_string();
          it = Cell::nil();
          break;
        }
      }
      ss += car(it).pretty_string();
      it = cdr(it);
    }
    if (!it.is_nil()) {
      ss += " ";
      ss += ".";
      ss += " ";
      ss += it.pretty_string();
    }
    ss += ')';
    return ss;
  }
  if (is_num()) {
    return to_num()->to_string();
  }
  if (is_str()) {
    return to_str()->str();
  }
  return to_string();
}

bool Cell::to_bool(SourceLocation loc PSCM_CXX20_MODULES_DEFAULT_ARG_COMPAT) const {
  PSCM_ASSERT_WITH_LOC(is_bool(), loc);
  return data_ != nullptr;
}

#define PSCM_DEFINE_CELL_TYPE(Type, type, tag)                                                                         \
  Cell::Cell(Type *t, SourceLocation loc PSCM_CXX20_MODULES_DEFAULT_ARG_COMPAT)                                        \
      : loc_(loc) {                                                                                                    \
    ref_count_++;                                                                                                      \
    tag_ = Tag::tag;                                                                                                   \
    data_ = (void *)t;                                                                                                 \
  }                                                                                                                    \
  Type *Cell::to_##type(SourceLocation loc PSCM_CXX20_MODULES_DEFAULT_ARG_COMPAT) const {                              \
    PSCM_ASSERT_WITH_LOC(is_##type(), loc);                                                                            \
    return (Type *)(data_);                                                                                            \
  }

PSCM_DEFINE_CELL_TYPE(Keyword, keyword, KEYWORD)
PSCM_DEFINE_CELL_TYPE(Pair, pair, PAIR)
PSCM_DEFINE_CELL_TYPE(String, str, STRING)
PSCM_DEFINE_CELL_TYPE(Symbol, sym, SYMBOL)
PSCM_DEFINE_CELL_TYPE(Char, char, CHAR)
PSCM_DEFINE_CELL_TYPE(Cell::Vec, vec, VECTOR)
PSCM_DEFINE_CELL_TYPE(Number, num, NUMBER)
PSCM_DEFINE_CELL_TYPE(Macro, macro, MACRO)
PSCM_DEFINE_CELL_TYPE(Procedure, proc, PROCEDURE)
PSCM_DEFINE_CELL_TYPE(Function, func, FUNCTION)
PSCM_DEFINE_CELL_TYPE(Promise, promise, PROMISE)
PSCM_DEFINE_CELL_TYPE(Continuation, cont, CONTINUATION)
PSCM_DEFINE_CELL_TYPE(Port, port, PORT)
PSCM_DEFINE_CELL_TYPE(Module, module, MODULE)
PSCM_DEFINE_CELL_TYPE(SmallObject, smob, SMOB)
PSCM_DEFINE_CELL_TYPE(HashTable, hash_table, HASH_TABLE)

Cell Cell::ex(const char *msg) {
  static Cell ret{ Tag::EXCEPTION, nullptr };
  auto len = strlen(msg);
  char *s = new char[len + 1];
  std::memcpy(s, msg, len);
  ret.data_ = (void *)s;
  return ret;
}

void Cell::display(Port& port) {
  if (is_char()) {
    to_char()->display(port);
    return;
  }
  if (is_str()) {
    to_str()->display(port);
    return;
  }
  auto s = to_string();
  auto iter = UIterator(s);
  auto curchr = iter.next32PostInc();
  while (curchr != UIterator::DONE) {
    port.write_char(curchr);
    curchr = iter.next32PostInc();
  }
}

bool operator==(const Cell& lhs, const Cell& rhs) {
  if (lhs.tag_ != rhs.tag_) {
    return false;
  }
  switch (lhs.tag_) {
  case Cell::Tag::NONE:
  case Cell::Tag::NIL: {
    return true;
  }
  case Cell::Tag::NUMBER: {
    return *lhs.to_num() == *rhs.to_num();
  }
  case Cell::Tag::PAIR: {
    return *lhs.to_pair() == *rhs.to_pair();
  }
  case Cell::Tag::STRING: {
    return *lhs.to_str() == *rhs.to_str();
  }
  case Cell::Tag::CHAR: {
    return *lhs.to_char() == *rhs.to_char();
  }
  case Cell::Tag::BOOL: {
    return lhs.to_bool() == rhs.to_bool();
  }
  case Cell::Tag::SYMBOL: {
    return *lhs.to_sym() == *rhs.to_sym();
  }
  case Cell::Tag::KEYWORD: {
    return *lhs.to_keyword() == *rhs.to_keyword();
  }
  case Cell::Tag::VECTOR: {
    return *lhs.to_vec() == *rhs.to_vec();
  }
  default: {
    return lhs.data_ == rhs.data_;
  }
  }
}

bool operator==(const Cell& lhs, const Number& rhs) {
  if (lhs.tag_ != Cell::Tag::NUMBER) {
    return false;
  }
  auto val = static_cast<Number *>(lhs.data_);
  PSCM_ASSERT(val);
  return *val == rhs;
}

bool operator==(const Cell& lhs, const Symbol& rhs) {
  if (lhs.tag_ != Cell::Tag::SYMBOL) {
    return false;
  }
  auto val = static_cast<Symbol *>(lhs.data_);
  PSCM_ASSERT(val);
  return *val == rhs;
}

bool operator==(const Cell& lhs, const Cell::Vec& rhs) {
  if (lhs.tag_ != Cell::Tag::VECTOR) {
    return false;
  }
  auto val = lhs.to_vec();
  PSCM_ASSERT(val);
  if (val->size() != rhs.size()) {
    return false;
  }
  for (int i = 0; i < val->size(); ++i) {
    auto l = val->at(i);
    auto r = rhs.at(i);
    if (!(l == r)) {
      return false;
    }
  }
  return true;
  //  return *val == rhs;
}

bool operator==(const Cell& lhs, const String& rhs) {
  if (lhs.tag_ != Cell::Tag::STRING) {
    return false;
  }
  auto val = static_cast<String *>(lhs.data_);
  PSCM_ASSERT(val);
  return *val == rhs;
}

std::ostream& operator<<(std::ostream& out, const Label& pos) {
  return out << to_string(pos);
}

const UString to_string(Label label) {
  switch (label) {
  case Label::EVAL: {
    return "EVAL";
  }
  case Label::DONE: {
    return "DONE";
  }
  case Label::APPLY: {
    return "APPLY";
  }
  case Label::EVAL_ARGS: {
    return "EVAL_ARGS";
  }
  case Label::AFTER_EVAL_FIRST_ARG: {
    return "AFTER_EVAL_FIRST_ARG";
  }
  case Label::AFTER_EVAL_OTHER_ARG: {
    return "AFTER_EVAL_OTHER_ARG";
  }
  case Label::AFTER_EVAL_ARGS: {
    return "AFTER_EVAL_ARGS";
  }
  case Label::AFTER_EVAL_OP: {
    return "AFTER_EVAL_OP";
  }
  case Label::AFTER_APPLY: {
    return "AFTER_APPLY";
  }
  case Label::APPLY_FUNC: {
    return "APPLY_FUNC";
  }
  case Label::APPLY_PROC: {
    return "APPLY_PROC";
  }
  case Label::APPLY_MACRO: {
    return "APPLY_MACRO";
  }
  case Label::AFTER_APPLY_USER_DEFINED_MACRO: {
    return "AFTER_APPLY_USER_DEFINED_MACRO";
  }
  case Label::APPLY_CONT: {
    return "APPLY_CONT";
  }
  case Label::AFTER_APPLY_FUNC: {
    return "AFTER_APPLY_FUNC";
  }
  case Label::AFTER_APPLY_PROC: {
    return "AFTER_APPLY_PROC";
  }
  case Label::AFTER_APPLY_MACRO: {
    return "AFTER_APPLY_MACRO";
  }
  case Label::AFTER_APPLY_CONT: {
    return "AFTER_APPLY_CONT";
  }
  case Label::APPLY_APPLY: {
    return "APPLY_APPLY";
  }
  case Label::APPLY_DEFINE: {
    return "APPLY_DEFINE";
  }
  case Label::APPLY_DEFINE_MACRO: {
    return "APPLY_DEFINE_MACRO";
  }
  case Label::APPLY_IS_DEFINED: {
    return "APPLY_IS_DEFINED";
  }
  case Label::APPLY_COND: {
    return "APPLY_COND";
  }
  case Label::APPLY_IF: {
    return "APPLY_IF";
  }
  case Label::APPLY_AND: {
    return "APPLY_AND";
  }
  case Label::APPLY_OR: {
    return "APPLY_OR";
  }
  case Label::APPLY_SET: {
    return "APPLY_SET";
  }
  case Label::APPLY_DELAY: {
    return "APPLY_DELAY";
  }
  case Label::APPLY_BEGIN: {
    return "APPLY_BEGIN";
  }
  case Label::APPLY_LOAD: {
    return "APPLY_LOAD";
  }
  case Label::AFTER_APPLY_EVAL: {
    return "AFTER_APPLY_EVAL";
  }
  case Label::APPLY_EVAL: {
    return "APPLY_EVAL";
  }
  case Label::APPLY_CURRENT_MODULE: {
    return "APPLY_CURRENT_MODULE";
  }
  case Label::APPLY_USE_MODULES: {
    return "APPLY_USE_MODULES";
  }
  case Label::APPLY_RESOLVE_MODULE: {
    return "APPLY_RESOLVE_MODULE";
  }
  case Label::APPLY_EXPORT: {
    return "APPLY_EXPORT";
  }
  case Label::AFTER_EVAL_DEFINE_ARG: {
    return "AFTER_EVAL_DEFINE_ARG";
  }
  case Label::AFTER_EVAL_SET_ARG: {
    return "AFTER_EVAL_SET_ARG";
  }
  case Label::APPLY_LAMBDA: {
    return "APPLY_LAMBDA";
  }
  case Label::APPLY_QUOTE: {
    return "APPLY_QUOTE";
  }
  case Label::APPLY_QUASIQUOTE: {
    return "APPLY_QUASIQUOTE";
  }
  case Label::APPLY_FOR_EACH: {
    return "APPLY_FOR_EACH";
  }
  case Label::APPLY_MAP: {
    return "APPLY_MAP";
  }
  case Label::APPLY_FORCE: {
    return "APPLY_FORCE";
  }
  case Label::AFTER_EVAL_PROMISE: {
    return "AFTER_EVAL_PROMISE";
  }
  case Label::AFTER_EVAL_FOR_EACH_FIRST_EXPR: {
    return "AFTER_EVAL_FOR_EACH_FIRST_EXPR";
  }
  case Label::AFTER_EVAL_MAP_FIRST_EXPR: {
    return "AFTER_EVAL_MAP_FIRST_EXPR";
  }
  case Label::AFTER_EVAL_MAP_OTHER_EXPR: {
    return "AFTER_EVAL_MAP_OTHER_EXPR";
  }
  case Label::AFTER_EVAL_FIRST_EXPR: {
    return "AFTER_EVAL_FIRST_EXPR";
  }
  case Label::AFTER_EVAL_OTHER_EXPR: {
    return "AFTER_EVAL_OTHER_EXPR";
  }
  case Label::AFTER_EVAL_COND_TEST: {
    return "AFTER_EVAL_COND_TEST";
  }
  case Label::AFTER_EVAL_IF_PRED: {
    return "AFTER_EVAL_IF_PRED";
  }
  case Label::AFTER_EVAL_AND_EXPR: {
    return "AFTER_EVAL_AND_EXPR";
  }
  case Label::AFTER_EVAL_OR_EXPR: {
    return "AFTER_EVAL_OR_EXPR";
  }
  case Label::AFTER_EVAL_CALL_WITH_VALUES_PRODUCER: {
    return "AFTER_EVAL_CALL_WITH_VALUES_PRODUCER";
  }
  default: {
    PSCM_ERROR("pos: {0}", int(label));
    PSCM_THROW_EXCEPTION("TODO: pos to_string " + pscm::to_string(int(label)));
  }
  }
}

bool Cell::is_self_evaluated() const {
  switch (tag_) {
  case Tag::PAIR:
  case Tag::SYMBOL: {
    return false;
  }
  default: {
    return true;
  }
  }
}

Cell Cell::is_eqv(const Cell& rhs) const {
  if (tag_ != rhs.tag_) {
    return Cell::bool_false();
  }
  if (tag_ == Tag::SYMBOL) {
    return Cell(*to_sym() == *rhs.to_sym());
  }
  else if (tag_ == Tag::NUMBER) {
    return Cell(*to_num() == *rhs.to_num());
  }
  else if (tag_ == Tag::CHAR) {
    return Cell(*to_char() == *rhs.to_char());
  }
  else if (tag_ == Tag::STRING) {
    return Cell(to_str()->empty() && rhs.to_str()->empty());
  }
  else if (tag_ == Tag::KEYWORD) {
    return Cell(*to_keyword() == *rhs.to_keyword());
  }

  bool eq = (data_ == rhs.data_);
  if (eq) {
    return Cell::bool_true();
  }
  else {
    return Cell::bool_false();
  }
}

Cell Cell::is_eq(const pscm::Cell& rhs) const {
  return is_eqv(rhs);
}

HashCodeType Cell::hash_code() const {
  if (is_none() || is_nil()) {
    return 0;
  }
  if (is_str()) {
    return to_str()->hash_code();
  }
  if (is_bool()) {
    if (to_bool()) {
      return 257;
    }
    else {
      return 258;
    }
  }
  if (is_char()) {
    auto ch = to_char();
    if (ch->is_eof()) {
      return 256;
    }
    else {
      return ch->to_int();
    }
  }
  if (is_keyword()) {
    return to_keyword()->hash_code();
  }
  if (is_sym()) {
    return to_sym()->hash_code();
  }
  PSCM_ASSERT(data_);
  auto code = (HashCodeType *)data_;
  return *code;
}

bool Cell::is_eq(Cell lhs, Cell rhs) {
  return lhs.is_eq(rhs).to_bool();
}

bool Cell::is_eqv(Cell lhs, Cell rhs) {
  return lhs.is_eqv(rhs).to_bool();
}

bool Cell::is_equal(Cell lhs, Cell rhs) {
  return lhs == rhs;
}

static Cell scm_cell_cmp(Cell args, Cell::ScmCmp cmp_func) {
  PSCM_ASSERT(args.is_pair());
  auto obj1 = car(args);
  auto obj2 = cadr(args);
  auto eq = cmp_func(obj1, obj2);
  return Cell(eq);
}

PSCM_DEFINE_BUILTIN_PROC(Cell, "eq?") {
  return scm_cell_cmp(args, Cell::is_eq);
}

PSCM_DEFINE_BUILTIN_PROC(Cell, "eqv?") {
  return scm_cell_cmp(args, Cell::is_eqv);
}

PSCM_DEFINE_BUILTIN_PROC(Cell, "equal?") {
  return scm_cell_cmp(args, Cell::is_equal);
}
} // namespace pscm

namespace std {
std::size_t hash<pscm::Cell>::operator()(const pscm::Cell& cell) const {
  auto s = cell.to_string();
  return s.hashCode();
}
} // namespace std