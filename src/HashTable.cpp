#include "pscm/HashTable.h"
#include "pscm/ApiManager.h"
#include "pscm/Function.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"

namespace pscm {

PSCM_DEFINE_BUILTIN_PROC(HashTable, "make-hash-table") {
  if (args.is_pair()) {
    auto arg = car(args);
    PSCM_ASSERT(arg.is_num());
    auto num = arg.to_number();
    PSCM_ASSERT(num->is_int());
    auto capacity = num->to_int();
    return new HashTable(capacity);
  }
  else {
    return new HashTable();
  }
}

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hash-ref") {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  PSCM_ASSERT(arg1.is_hash_table());
  auto key = cadr(args);
  auto hash_table = arg1.to_hash_table();
  auto ret = hash_table->get_or(key, Cell::bool_false());
  return ret;
};

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hash-get-handle") {
  PSCM_THROW_EXCEPTION("not implement now");
  return Cell::none();
};

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hash-set!") {
  PSCM_ASSERT(args.is_pair());
  auto arg1 = car(args);
  PSCM_ASSERT(arg1.is_hash_table());
  auto key = cadr(args);
  auto value = caddr(args);
  auto hash_table = arg1.to_hash_table();
  hash_table->set(key, value);
  return value;
};

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hash-remove!") {
  PSCM_THROW_EXCEPTION("not implement now");
  return Cell::none();
};

PSCM_DEFINE_BUILTIN_PROC(HashTable, "hash-fold") {
  PSCM_THROW_EXCEPTION("not implement now");
  return Cell::none();
};

void HashTable::set(Cell key, Cell value) {
  map_[key] = value;
}

Cell HashTable::get_or(Cell key, Cell default_value) {
  auto it = map_.find(key);
  if (it == map_.end()) {
    return default_value;
  }
  else {
    return it->second;
  }
}

std::ostream& operator<<(std::ostream& os, const HashTable& hash_table) {
  os << '#';
  os << '<';
  os << "hash-table";
  os << ' ';
  os << hash_table.size_;
  os << '/';
  os << hash_table.capacity_;
  os << '>';
  return os;
}

} // namespace pscm
