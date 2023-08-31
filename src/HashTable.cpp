#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/HashTable.h"
#include "pscm/ApiManager.h"
#include "pscm/Function.h"
#include "pscm/Number.h"
#include "pscm/SchemeProxy.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <spdlog/fmt/fmt.h>
#endif
namespace pscm {
PSCM_INLINE_LOG_DECLARE("pscm.core.HashTable");

PSCM_DEFINE_BUILTIN_PROC(HashTable, "make-hash-table") {

  HashTable *hash_table;
  if (args.is_pair()) {
    auto arg = car(args);
    PSCM_ASSERT(arg.is_num());
    auto num = arg.to_num();
    PSCM_ASSERT(num->is_int());
    auto capacity = num->to_int();
    hash_table = new HashTable(capacity);
  }
  else {
    hash_table = new HashTable();
  }
  return hash_table;
}

static Cell scm_hash_get_handle(Cell args, Cell::ScmCmp cmp_func) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  PSCM_ASSERT(arg1.is_hash_table());
  auto key = cadr(args);
  auto hash_table = arg1.to_hash_table();
  auto entry = hash_table->get(key, cmp_func);
  return entry;
}

static Cell scm_hash_ref(Cell args, Cell::ScmCmp cmp_func) {
  auto entry = scm_hash_get_handle(args, cmp_func);
  if (entry == Cell::bool_false()) {
    return Cell::bool_false();
  }
  return cdr(entry);
}

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hash-ref") {
  return scm_hash_ref(args, Cell::is_equal);
};

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hashq-ref") {
  return scm_hash_ref(args, Cell::is_eq);
};

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hashv-ref") {
  return scm_hash_ref(args, Cell::is_eqv);
};

static Cell scm_hash_create_handle(Cell args, Cell::ScmCmp cmp_func) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  PSCM_ASSERT(arg1.is_hash_table());
  auto key = cadr(args);
  auto value = caddr(args);
  auto hash_table = arg1.to_hash_table();
  auto entry = hash_table->set(key, value, cmp_func);
  return entry;
}

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hashq-create-handle!") {
  return scm_hash_create_handle(args, Cell::is_eq);
};

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hashv-create-handle!") {
  return scm_hash_create_handle(args, Cell::is_eqv);
};

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hash-create-handle!") {
  return scm_hash_create_handle(args, Cell::is_equal);
};

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hash-get-handle") {
  return scm_hash_get_handle(args, Cell::is_equal);
};

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hashq-get-handle") {
  return scm_hash_get_handle(args, Cell::is_eq);
};

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hashv-get-handle") {
  return scm_hash_get_handle(args, Cell::is_eqv);
};

static Cell scm_hash_set(Cell args, Cell::ScmCmp cmp_func) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  PSCM_ASSERT(arg1.is_hash_table());
  auto key = cadr(args);
  auto value = caddr(args);
  auto hash_table = arg1.to_hash_table();
  hash_table->set(key, value, cmp_func);
  return value;
}

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hash-set!") {
  return scm_hash_set(args, Cell::is_equal);
};

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hashq-set!") {
  return scm_hash_set(args, Cell::is_eq);
};

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hashv-set!") {
  return scm_hash_set(args, Cell::is_eqv);
};

static Cell scm_hash_remove(Cell args, Cell::ScmCmp cmp_func) {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  PSCM_ASSERT(arg1.is_hash_table());
  auto key = cadr(args);
  auto hash_table = arg1.to_hash_table();
  hash_table->remove(key, cmp_func);
  return Cell::none();
}

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hash-remove!") {
  return scm_hash_remove(args, Cell::is_equal);
};

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hashq-remove!") {
  return scm_hash_remove(args, Cell::is_eq);
};

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hashv-remove!") {
  return scm_hash_remove(args, Cell::is_eqv);
};

PSCM_DEFINE_BUILTIN_MACRO_PROC_WRAPPER(HashTable, "hash-fold", Label::TODO, "(proc init table)") {
  PSCM_ASSERT(args.is_pair());
  auto proc = car(args);
  auto init = cadr(args);
  auto table = caddr(args);
  PSCM_ASSERT(proc.is_sym());
  PSCM_ASSERT(init.is_sym());
  PSCM_ASSERT(table.is_sym());
  proc = env->get(proc.to_sym());
  init = env->get(init.to_sym());
  table = env->get(table.to_sym());
  PSCM_ASSERT(table.is_hash_table());
  auto hash_table = table.to_hash_table();
  auto val = init;
  hash_table->for_each([&val, &scm, env, proc](Cell key, Cell value) {
    val = scm.eval(env, list(proc, list(quote, key), list(quote, value), list(quote, val)));
  });
  return list(quote, val);
};

PSCM_DEFINE_BUILTIN_MACRO_PROC_WRAPPER(HashTable, "hash-for-each-handle", Label::TODO, "(proc table)") {
  PSCM_ASSERT(args.is_pair());
  auto proc = car(args);
  auto table = cadr(args);
  PSCM_ASSERT(proc.is_sym());
  PSCM_ASSERT(table.is_sym());
  proc = env->get(proc.to_sym());
  table = env->get(table.to_sym());
  PSCM_ASSERT(table.is_hash_table());
  auto hash_table = table.to_hash_table();
  hash_table->for_each_handle([&scm, env, proc](Cell entry) {
    scm.eval(env, list(proc, list(quote, entry)));
  });
  return Cell::none();
};

Cell HashTable::set(Cell key, Cell value, Cell::ScmCmp cmp_func) {
  auto hash_code = key.hash_code();
  auto idx = hash_code % map_.size();
  auto bucket = map_[idx];
  while (bucket.is_pair()) {
    auto entry = car(bucket);
    if (cmp_func(key, car(entry))) {
      entry.to_pair()->second = value;
      return entry;
    }
    bucket = cdr(bucket);
  }
  Cell entry = cons(key, value);
  map_[idx] = cons(entry, map_[idx]);
  return entry;
}

Cell HashTable::get(Cell key, Cell::ScmCmp cmp_func) {
  auto hash_code = key.hash_code();
  auto idx = hash_code % map_.size();
  auto bucket = map_[idx];
  while (bucket.is_pair()) {
    auto entry = car(bucket);
    if (cmp_func(key, car(entry))) {
      return entry;
    }
    bucket = cdr(bucket);
  }
  return Cell::bool_false();
}

Cell HashTable::remove(Cell key, Cell::ScmCmp cmp_func) {
  auto hash_code = key.hash_code();
  auto idx = hash_code % map_.size();
  auto bucket = map_[idx];
  auto last = cons(nil, bucket);
  auto new_bucket = last;
  while (bucket.is_pair()) {
    auto entry = car(bucket);
    if (cmp_func(key, car(entry))) {
      last->second = cdr(bucket);
      map_[idx] = new_bucket->second;
      return entry;
    }
    last = bucket.to_pair();
    bucket = cdr(bucket);
  }
  return Cell::bool_false();
}

void HashTable::for_each(std::function<void(Cell, Cell)> func) {
  for (Cell bucket : map_) {
    pscm::for_each(
        [func](Cell expr, auto) {
          func(car(expr), cdr(expr));
        },
        bucket);
  }
}

void HashTable::for_each_handle(std::function<void(Cell)> func) {
  for (Cell bucket : map_) {
    pscm::for_each(
        [func](Cell expr, auto) {
          func(expr);
        },
        bucket);
  }
}

UString HashTable::to_string() const{
  UString os;
  os += '#';
  os += '<';
  os += "hash-table";
  os += ' ';
  os += pscm::to_string(size_);
  os += '/';
  os += pscm::to_string(capacity_);
  os += ' ';
  os += pscm::to_string(this);
  os += '>';
  return os;
}

static Cell scm_hash(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto key = car(args);
  auto arg2 = cadr(args);
  PSCM_ASSERT(arg2.is_num());
  auto num = arg2.to_num();
  PSCM_ASSERT(num->is_int());
  auto hash_code = key.hash_code();
  return new Number(hash_code);
}

PSCM_DEFINE_BUILTIN_PROC(Hash, "hashq") {
  return scm_hash(args);
}

PSCM_DEFINE_BUILTIN_PROC(Hash, "hashv") {
  return scm_hash(args);
}

PSCM_DEFINE_BUILTIN_PROC(Hash, "hash") {
  return scm_hash(args);
}
} // namespace pscm
