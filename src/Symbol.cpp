//
// Created by PikachuHy on 2023/2/25.
//
#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/Symbol.h"
#include "pscm/ApiManager.h"
#include "pscm/Cell.h"
#include "pscm/Pair.h"
#include "pscm/SchemeProxy.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include "pscm/misc/ICUCompat.h"
#include <fstream>
#include <iostream>
#include <ostream>
#endif
namespace pscm {
PSCM_INLINE_LOG_DECLARE("pscm.core.Symbol");
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
  return out << sym.to_string();
}

UString Symbol::to_string() const{
  if (name_.indexOf(' ') != -1) {
    UString out("#{");
    out.append(
      name_
    ).findAndReplace(
      " ", "\\ "
    ).append(
      "}#"
    );
    return out;
  } else {
    UString out(name_);
    return out;
  }
}

bool Symbol::operator==(const Symbol& sym) const {
  return name_ == sym.name_;
}

HashCodeType Symbol::hash_code() const {
  return name_.hashCode();
}

void Symbol::print_debug_info() {
  if (filename_.isEmpty()) {
    return;
  }
  std::cout << name_ << " from " << filename_ << ":" << row_ << ":" << col_ << std::endl;
  std::fstream in;
  open_fstream(in, filename_);
  if (!in.is_open()) {
    PSCM_ERROR("open file error: {0}", filename_);
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
  return Symbol(UString(data, len));
}

Symbol *gensym() {
  static int index = 0;
  auto sym = new Symbol(" g" + pscm::to_string(index++));
  return sym;
}

PSCM_DEFINE_BUILTIN_PROC(Symbol, "gensym") {
  return gensym();
}

PSCM_DEFINE_BUILTIN_MACRO(Symbol, "defined?", Label::APPLY_IS_DEFINED) {
  PSCM_ASSERT(args.is_pair());
  auto val = scm.eval(env, car(args));
  PSCM_ASSERT(val.is_sym());
  auto sym = val.to_sym();
  auto has_sym = env->contains(sym);
  return Cell(has_sym);
}

} // namespace pscm