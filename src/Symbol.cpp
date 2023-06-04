//
// Created by PikachuHy on 2023/2/25.
//

#include "pscm/Symbol.h"
#include "pscm/ApiManager.h"
#include "pscm/Cell.h"
#include "pscm/Pair.h"
#include "pscm/SchemeProxy.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <fstream>
#include <iostream>
#include <ostream>

namespace pscm {
Symbol callcc("call-with-current-continuation");
Symbol call_with_values("call-with-values");
Symbol values("values");
Symbol cond_else("else");
Symbol sym_if("if");
Symbol Symbol::for_each = Symbol("for-each");
Symbol Symbol::map = Symbol("map");
Symbol Symbol::load = Symbol("load");
Symbol Symbol::quasiquote = Symbol("quasiquote");
Symbol Symbol::unquote_splicing = Symbol("unquote-splicing");

std::ostream& operator<<(std::ostream& out, const Symbol& sym) {
  auto name = sym.name_;
  if (name.find(' ') != std::string::npos) {
    out << "#";
    out << "{";
    for (auto ch : name) {
      if (ch == ' ') {
        out << "\\";
      }
      out << ch;
    }
    out << "}";
    out << "#";
    return out;
  }
  return out << name;
}

bool Symbol::operator==(const Symbol& sym) const {
  return name_ == sym.name_;
}

HashCodeType Symbol::hash_code() const {
  HashCodeType code = 0;
  for (char ch : name_) {
    code = int(ch) + code * 37;
  }
  return code;
}

void Symbol::print_debug_info() {
  if (filename_.empty()) {
    return;
  }
  std::cout << name_ << " from " << filename_ << ":" << row_ << ":" << col_ << std::endl;
  std::fstream in;
  in.open(filename_);
  if (!in.is_open()) {
    SPDLOG_ERROR("open file error: {}", filename_);
  }
  std::string line;
  for (size_t i = 0; i < row_; i++) {
    std::getline(in, line);
  }
  std::cout << line << std::endl;
  for (size_t i = 1; i < col_; i++) {
    std::cout << ' ';
  }
  std::cout << "^" << std::endl;
}

Symbol operator""_sym(const char *data, std::size_t len) {
  return Symbol(std::string(data, len));
}

Symbol *gensym() {
  static int index = 0;
  auto sym = new Symbol(" g" + std::to_string(index++));
  return sym;
}

PSCM_DEFINE_BUILTIN_PROC(Symbol, "gensym") {
  return gensym();
}

PSCM_DEFINE_BUILTIN_MACRO(Symbol, "defined?", Label::APPLY_IS_DEFINED) {
  PSCM_ASSERT(args.is_pair());
  auto val = scm.eval(env, car(args));
  PSCM_ASSERT(val.is_sym());
  auto sym = val.to_symbol();
  auto has_sym = env->contains(sym);
  return Cell(has_sym);
}

} // namespace pscm