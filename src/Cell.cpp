//
// Created by PikachuHy on 2023/2/23.
//

#include "pscm/Cell.h"
#include "pscm/Char.h"
#include "pscm/Continuation.h"
#include "pscm/Function.h"
#include "pscm/Macro.h"
#include "pscm/Number.h"
#include "pscm/Pair.h"
#include "pscm/Port.h"
#include "pscm/Procedure.h"
#include "pscm/Promise.h"
#include "pscm/Str.h"
#include "pscm/Symbol.h"
#include "pscm/common_def.h"
#include <cassert>
#include <cstring>
#include <sstream>
#include <unordered_set>

namespace pscm {
Cell nil = Cell::nil();

Cell::Cell(Number *num) {
  ref_count_++;
  tag_ = Tag::NUMBER;
  data_ = num;
}

Cell::Cell(Char *ch) {
  ref_count_++;
  tag_ = Tag::CHAR;
  data_ = ch;
}

Cell::Cell(String *str) {
  ref_count_++;
  tag_ = Tag::STRING;
  data_ = str;
}

Cell::Cell(Symbol *sym) {
  ref_count_++;
  tag_ = Tag::SYMBOL;
  data_ = sym;
}

Cell::Cell(Pair *pair) {
  ref_count_++;
  tag_ = Tag::PAIR;
  data_ = pair;
}

Cell::Cell(Vec *vec) {
  ref_count_++;
  tag_ = Tag::VECTOR;
  data_ = vec;
}

Cell::Cell(Function *f) {
  ref_count_++;
  tag_ = Tag::FUNCTION;
  data_ = (void *)f;
}

Cell::Cell(const Procedure *proc) {
  ref_count_++;
  tag_ = Tag::PROCEDURE;
  data_ = (void *)proc;
}

Cell::Cell(Macro *f) {
  ref_count_++;
  tag_ = Tag::MACRO;
  data_ = (void *)f;
}

Cell::Cell(Promise *p) {
  ref_count_++;
  tag_ = Tag::PROMISE;
  data_ = (void *)p;
}

Cell::Cell(Continuation *cont) {
  ref_count_++;
  tag_ = Tag::CONTINUATION;
  data_ = (void *)cont;
}

Cell::Cell(Port *port) {
  ref_count_++;
  tag_ = Tag::PORT;
  data_ = (void *)port;
}

Cell::Cell(SmallObject *smob) {
  ref_count_++;
  tag_ = Tag::SMOB;
  data_ = (void *)smob;
}

Cell::Cell(Module *module) {
  ref_count_++;
  tag_ = Tag::MODULE;
  data_ = (void *)module;
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

std::string Cell::to_string() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

std::string Cell::pretty_string() const {
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
    auto sym = to_symbol();
    if (sym->name().find(' ') != std::string_view::npos) {
      std::stringstream ss;
      ss << "\e[;34m";
      ss << *this;
      ss << "\e[0m";
      return ss.str();
    }
    else {
      std::stringstream ss;
      ss << *this;
      return ss.str();
    }
  }
  if (is_macro()) {
    auto macro = to_macro();
    auto name = std::string(macro->name());
    std::stringstream ss;
    ss << "\e[;35m";
    ss << name;
    ss << "\e[0m";
    return ss.str();
  }
  if (is_func()) {
    auto func = to_func();
    return std::string(func->name());
  }
  if (is_pair()) {
    std::stringstream ss;
    if (car(*this).is_macro()) {
      auto macro = car(*this).to_macro();
      if (macro->name() == "quote") {
        ss << "'";
        ss << cadr(*this).pretty_string();
        return ss.str();
      }
      else if (macro->name() == "quasiquote") {
        ss << "`";
        ss << cadr(*this).pretty_string();
        return ss.str();
      }
      else if (macro->name() == "unquote") {
        ss << ",";
        ss << cadr(*this).pretty_string();
        return ss.str();
      }
    }
    else if (car(*this).is_sym()) {
      auto sym = car(*this).to_symbol();
      if (sym->name() == "quasiquote") {
        ss << "`";
        ss << cadr(*this).pretty_string();
        return ss.str();
      }
      else if (sym->name() == "unquote") {
        ss << ",";
        ss << cadr(*this).pretty_string();
        return ss.str();
      }
      else if (sym->name() == "unquote-splicing") {
        ss << ",@";
        ss << cadr(*this).pretty_string();
        return ss.str();
      }
    }

    ss << '(';
    auto it = *this;
    ss << car(it).pretty_string();
    it = cdr(it);
    while (it.is_pair()) {
      ss << " ";
      if (car(it).is_sym() && car(it).to_symbol()->name() == "unquote") {
        if (cddr(it).is_nil()) {
          ss << ".";
          ss << " ";
          ss << ",";
          ss << cadr(it).pretty_string();
          break;
        }
      }
      ss << car(it).pretty_string();
      it = cdr(it);
    }
    ss << ')';
    return ss.str();
  }
  if (is_num()) {
    std::stringstream ss;
    ss << *to_number();
    return ss.str();
  }
  if (is_str()) {
    return std::string(to_str()->str());
  }
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

Pair *Cell::to_pair(SourceLocation loc) const {
  PSCM_ASSERT_WITH_LOC(is_pair(), loc);
  return (Pair *)(data_);
}

Cell::Vec *Cell::to_vec(SourceLocation loc) const {
  PSCM_ASSERT_WITH_LOC(is_vec(), loc);
  return (Vec *)(data_);
}

Symbol *Cell::to_symbol(SourceLocation loc) const {
  PSCM_ASSERT_WITH_LOC(is_sym(), loc);
  return (Symbol *)(data_);
}

Char *Cell::to_char(SourceLocation loc) const {
  PSCM_ASSERT_WITH_LOC(is_char(), loc);
  return (Char *)(data_);
}

String *Cell::to_str(SourceLocation loc) const {
  PSCM_ASSERT_WITH_LOC(is_str(), loc);
  return (String *)(data_);
}

Number *Cell::to_number(SourceLocation loc) const {
  PSCM_ASSERT_WITH_LOC(is_num(), loc);
  return (Number *)(data_);
}

bool Cell::to_bool(SourceLocation loc) const {
  PSCM_ASSERT_WITH_LOC(is_bool(), loc);
  return data_ != nullptr;
}

Function *Cell::to_func(SourceLocation loc) const {
  PSCM_ASSERT_WITH_LOC(is_func(), loc);
  return (Function *)(data_);
}

Macro *Cell::to_macro(SourceLocation loc) const {
  PSCM_ASSERT_WITH_LOC(is_macro(), loc);
  return (Macro *)(data_);
}

Procedure *Cell::to_proc(SourceLocation loc) const {
  PSCM_ASSERT_WITH_LOC(is_proc(), loc);
  return (Procedure *)(data_);
}

Promise *Cell::to_promise(SourceLocation loc) const {
  PSCM_ASSERT_WITH_LOC(is_promise(), loc);
  return (Promise *)(data_);
}

Continuation *Cell::to_cont(SourceLocation loc) const {
  PSCM_ASSERT_WITH_LOC(is_cont(), loc);
  return (Continuation *)(data_);
}

Port *Cell::to_port(SourceLocation loc) const {
  PSCM_ASSERT_WITH_LOC(is_port(), loc);
  return (Port *)(data_);
}

SmallObject *Cell::to_smob(SourceLocation loc) const {
  PSCM_ASSERT_WITH_LOC(is_smob(), loc);
  return (SmallObject *)(data_);
}

Module *Cell::to_module(SourceLocation loc) const {
  PSCM_ASSERT_WITH_LOC(is_module(), loc);
  return (Module *)(data_);
}

Cell Cell::ex(const char *msg) {
  static Cell ret{ Tag::EXCEPTION, nullptr };
  auto len = strlen(msg);
  char *s = new char[len + 1];
  std::memcpy(s, msg, len);
  ret.data_ = (void *)s;
  return ret;
}

std::ostream& operator<<(std::ostream& out, const Cell& cell) {
  if (cell.tag_ == Cell::Tag::NONE) {
    return out << "NONE";
  }
  if (cell.tag_ == Cell::Tag::EXCEPTION) {
    return out << "EXCEPTION: " << (const char *)cell.data_;
  }
  if (cell.tag_ == Cell::Tag::NIL) {
    return out << "()";
  }
  if (cell.tag_ == Cell::Tag::SYMBOL) {
    return out << *cell.to_symbol();
  }
  if (cell.tag_ == Cell::Tag::FUNCTION) {
    return out << *cell.to_func();
  }
  if (cell.tag_ == Cell::Tag::MACRO) {
    return out << *cell.to_macro();
  }
  if (cell.tag_ == Cell::Tag::NUMBER) {
    return out << *cell.to_number();
  }
  if (cell.tag_ == Cell::Tag::CHAR) {
    return out << *cell.to_char();
  }
  if (cell.tag_ == Cell::Tag::STRING) {
    return out << *cell.to_str();
  }
  if (cell.tag_ == Cell::Tag::BOOL) {
    if (cell.to_bool()) {
      return out << "#t";
    }
    else {
      return out << "#f";
    }
  }
  if (cell.tag_ == Cell::Tag::PAIR) {
    out << "(";
    // TODO:
    std::unordered_set<Pair *> p_set;
    int num = 0;
    auto p = cell.to_pair();
    p_set.insert(p);
    while (p) {
      out << p->first;
      if (p->second.is_pair()) {
        out << " ";
        p = p->second.to_pair();
        if (p_set.contains(p)) {
          out << ". ";
          out << "#" << num << "#";
          break;
        }
      }
      else if (p->second.is_nil()) {
        break;
      }
      else {
        out << " . ";
        if (p->second.is_pair()) {
          auto p2 = p->second.to_pair();
          if (p_set.contains(p2)) {
            out << ". ";
            out << "#" << num << "#";
            break;
          }
        }
        out << p->second;
        break;
      }
    }
    out << ")";
    return out;
  }
  if (cell.tag_ == Cell::Tag::VECTOR) {
    out << "#";
    out << "(";
    auto vec = cell.to_vec();
    if (!vec->empty()) {
      for (int i = 0; i < vec->size() - 1; ++i) {
        out << vec->at(i);
        out << " ";
      }
      out << vec->back();
    }
    out << ")";
    return out;
  }
  if (cell.tag_ == Cell::Tag::PROCEDURE) {
    return out << *cell.to_proc();
  }
  if (cell.tag_ == Cell::Tag::PROMISE) {
    return out << *cell.to_promise();
  }
  if (cell.tag_ == Cell::Tag::CONTINUATION) {
    return out << *cell.to_cont();
  }
  if (cell.tag_ == Cell::Tag::PORT) {
    return out << cell.to_port()->to_string();
  }
  SPDLOG_ERROR("TODO: {}", int(cell.tag_));
  //  PSCM_THROW_EXCEPTION("TODO: cell tag ");
  return out << "TODO";
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
  for (auto ch : s) {
    port.write_char(ch);
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
    return *lhs.to_number() == *rhs.to_number();
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
    return *lhs.to_symbol() == *rhs.to_symbol();
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
    if (l != r) {
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

std::string to_string(Label label) {
  std::stringstream ss;
  ss << label;
  return ss.str();
}

std::ostream& operator<<(std::ostream& out, const Label& pos) {
  switch (pos) {
  case Label::EVAL: {
    out << "EVAL";
    break;
  }
  case Label::DONE: {
    out << "DONE";
    break;
  }
  case Label::APPLY: {
    out << "APPLY";
    break;
  }
  case Label::EVAL_ARGS: {
    out << "EVAL_ARGS";
    break;
  }
  case Label::AFTER_EVAL_FIRST_ARG: {
    out << "AFTER_EVAL_FIRST_ARG";
    break;
  }
  case Label::AFTER_EVAL_OTHER_ARG: {
    out << "AFTER_EVAL_OTHER_ARG";
    break;
  }
  case Label::AFTER_EVAL_ARGS: {
    out << "AFTER_EVAL_ARGS";
    break;
  }
  case Label::AFTER_EVAL_OP: {
    out << "AFTER_EVAL_OP";
    break;
  }
  case Label::AFTER_APPLY: {
    out << "AFTER_APPLY";
    break;
  }
  case Label::APPLY_FUNC: {
    out << "APPLY_FUNC";
    break;
  }
  case Label::APPLY_PROC: {
    out << "APPLY_PROC";
    break;
  }
  case Label::APPLY_MACRO: {
    out << "APPLY_MACRO";
    break;
  }
  case Label::AFTER_APPLY_USER_DEFINED_MACRO: {
    out << "AFTER_APPLY_USER_DEFINED_MACRO";
    break;
  }
  case Label::APPLY_CONT: {
    out << "APPLY_CONT";
    break;
  }
  case Label::AFTER_APPLY_FUNC: {
    out << "AFTER_APPLY_FUNC";
    break;
  }
  case Label::AFTER_APPLY_PROC: {
    out << "AFTER_APPLY_PROC";
    break;
  }
  case Label::AFTER_APPLY_MACRO: {
    out << "AFTER_APPLY_MACRO";
    break;
  }
  case Label::AFTER_APPLY_CONT: {
    out << "AFTER_APPLY_CONT";
    break;
  }
  case Label::APPLY_APPLY: {
    out << "APPLY_APPLY";
    break;
  }
  case Label::APPLY_DEFINE: {
    out << "APPLY_DEFINE";
    break;
  }
  case Label::APPLY_DEFINE_MACRO: {
    out << "APPLY_DEFINE_MACRO";
    break;
  }
  case Label::APPLY_COND: {
    out << "APPLY_COND";
    break;
  }
  case Label::APPLY_IF: {
    out << "APPLY_IF";
    break;
  }
  case Label::APPLY_AND: {
    out << "APPLY_AND";
    break;
  }
  case Label::APPLY_OR: {
    out << "APPLY_OR";
    break;
  }
  case Label::APPLY_SET: {
    out << "APPLY_SET";
    break;
  }
  case Label::APPLY_DELAY: {
    out << "APPLY_DELAY";
    break;
  }
  case Label::APPLY_BEGIN: {
    out << "APPLY_BEGIN";
    break;
  }
  case Label::APPLY_LOAD: {
    out << "APPLY_LOAD";
    break;
  }
  case Label::APPLY_EVAL: {
    out << "APPLY_EVAL";
    break;
  }
  case Label::AFTER_EVAL_DEFINE_ARG: {
    out << "AFTER_EVAL_DEFINE_ARG";
    break;
  }
  case Label::AFTER_EVAL_SET_ARG: {
    out << "AFTER_EVAL_SET_ARG";
    break;
  }
  case Label::APPLY_LAMBDA: {
    out << "APPLY_LAMBDA";
    break;
  }
  case Label::APPLY_QUOTE: {
    out << "APPLY_QUOTE";
    break;
  }
  case Label::APPLY_QUASIQUOTE: {
    out << "APPLY_QUASIQUOTE";
    break;
  }
  case Label::APPLY_FOR_EACH: {
    out << "APPLY_FOR_EACH";
    break;
  }
  case Label::APPLY_MAP: {
    out << "APPLY_MAP";
    break;
  }
  case Label::APPLY_FORCE: {
    out << "APPLY_FORCE";
    break;
  }
  case Label::AFTER_EVAL_PROMISE: {
    out << "AFTER_EVAL_PROMISE";
    break;
  }
  case Label::AFTER_EVAL_FOR_EACH_FIRST_EXPR: {
    out << "AFTER_EVAL_FOR_EACH_FIRST_EXPR";
    break;
  }
  case Label::AFTER_EVAL_MAP_FIRST_EXPR: {
    out << "AFTER_EVAL_MAP_FIRST_EXPR";
    break;
  }
  case Label::AFTER_EVAL_MAP_OTHER_EXPR: {
    out << "AFTER_EVAL_MAP_OTHER_EXPR";
    break;
  }
  case Label::AFTER_EVAL_FIRST_EXPR: {
    out << "AFTER_EVAL_FIRST_EXPR";
    break;
  }
  case Label::AFTER_EVAL_OTHER_EXPR: {
    out << "AFTER_EVAL_OTHER_EXPR";
    break;
  }
  case Label::AFTER_EVAL_COND_TEST: {
    out << "AFTER_EVAL_COND_TEST";
    break;
  }
  case Label::AFTER_EVAL_IF_PRED: {
    out << "AFTER_EVAL_IF_PRED";
    break;
  }
  case Label::AFTER_EVAL_AND_EXPR: {
    out << "AFTER_EVAL_AND_EXPR";
    break;
  }
  case Label::AFTER_EVAL_OR_EXPR: {
    out << "AFTER_EVAL_OR_EXPR";
    break;
  }
  case Label::AFTER_EVAL_CALL_WITH_VALUES_PRODUCER: {
    out << "AFTER_EVAL_CALL_WITH_VALUES_PRODUCER";
    break;
  }
  default: {
    SPDLOG_ERROR("pos: {}", int(pos));
    PSCM_THROW_EXCEPTION("TODO: pos to_string " + std::to_string(int(pos)));
  }
  }
  return out;
}

bool Cell::is_self_evaluated() const {
  switch (tag_) {
  case Tag::NUMBER:
  case Tag::CHAR:
  case Tag::STRING:
  case Tag::BOOL:
  case Tag::VECTOR:
  case Tag::MACRO:
  case Tag::PROCEDURE:
  case Tag::FUNCTION:
  case Tag::NIL:
  case Tag::PROMISE:
  case Tag::PORT:
  case Tag::CONTINUATION: {
    return true;
  }
  default: {
    return false;
  }
  }
}

Cell Cell::is_eqv(const Cell& rhs) const {
  if (tag_ != rhs.tag_) {
    return Cell::bool_false();
  }
  if (tag_ == Tag::SYMBOL) {
    return Cell(*to_symbol() == *rhs.to_symbol());
  }
  else if (tag_ == Tag::NUMBER) {
    return Cell(*to_number() == *rhs.to_number());
  }
  else if (tag_ == Tag::CHAR) {
    return Cell(*to_char() == *rhs.to_char());
  }
  else if (tag_ == Tag::STRING) {
    return Cell(to_str()->empty() && rhs.to_str()->empty());
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

std::string SourceLocation::to_string() const {
  auto name = std::string(filename);
  auto pos = name.find_last_of('/');
  name = name.substr(pos + 1);
  return name + ":" + std::to_string(linenum); // + " " + std::string(funcname);
}
} // namespace pscm
