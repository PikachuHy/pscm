//
// Created by PikachuHy on 2023/7/21.
//
module;
#include "pscm/ApiManager.h"
#include "pscm/Char.h"
#include "pscm/Continuation.h"
#include "pscm/Evaluator.h"
#include "pscm/Exception.h"
#include "pscm/Expander.h"
#include "pscm/Function.h"
#include "pscm/HashTable.h"
#include "pscm/Keyword.h"
#include "pscm/Logger.h"
#include "pscm/Macro.h"
#include "pscm/Module.h"
#include "pscm/Number.h"
#include "pscm/Pair.h"
#include "pscm/Parser.h"
#include "pscm/Port.h"
#include "pscm/Procedure.h"
#include "pscm/Promise.h"
#include "pscm/Scheme.h"
#include "pscm/SchemeProxy.h"
#include "pscm/Str.h"
#include "pscm/Symbol.h"
#include "pscm/SymbolTable.h"
#include "pscm/scm_utils.h"
#include "pscm/version.h"
export module pscm;
export import pscm.logger;
export import pscm.misc;
export import pscm.compat;

export namespace pscm {
using pscm::apply;
using pscm::call_with_values;
using pscm::callcc;
using pscm::cond_else;
using pscm::lambda;
using pscm::nil;
using pscm::quote;
using pscm::sym_if;
using pscm::values;

using pscm::Cell;
using pscm::Evaluator;
using pscm::HashCodeType;
using pscm::Parser;
using pscm::Scheme;
using pscm::SchemeProxy;
using pscm::operator<<;
using pscm::Char;
using pscm::Complex;
using pscm::Continuation;
using pscm::Exception;
using pscm::Function;
using pscm::gensym;
using pscm::HashTable;
using pscm::Keyword;
using pscm::Label;
using pscm::Macro;
using pscm::Module;
using pscm::Number;
using pscm::Port;
using pscm::Procedure;
using pscm::Promise;
using pscm::Rational;
using pscm::SmallObject;
using pscm::String;
using pscm::Symbol;
using pscm::operator""_sym;
using pscm::operator""_num;
using pscm::operator""_str;
using pscm::AList;
using pscm::ApiManager;
using pscm::expand_case;
using pscm::expand_do;
using pscm::expand_let;
using pscm::expand_let_star;
using pscm::expand_letrec;
using pscm::QuasiQuotationExpander;
using pscm::SymbolTable;
using pscm::VersionInfo;
// pair
using pscm::caaar;
using pscm::caadr;
using pscm::caar;
using pscm::cadar;
using pscm::caddar;
using pscm::caddr;
using pscm::cadr;
using pscm::car;
using pscm::cdadr;
using pscm::cdar;
using pscm::cddr;
using pscm::cdr;
using pscm::cons;
using pscm::Pair;

//
using pscm::for_each;
using pscm::list;
using pscm::list_length;
using pscm::map;
using pscm::reverse_argl;
} // namespace pscm
