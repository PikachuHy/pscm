#include "pscm/Cell.h"
#include <unordered_map>

namespace pscm {
class HashTable {
public:
  HashTable(std::size_t capacity = 32)
      : capacity_(capacity) {
  }

  friend std::ostream& operator<<(std::ostream& os, const HashTable& hash_table);
  void set(Cell key, Cell value);
  Cell get_or(Cell key, Cell default_value = Cell::bool_false());

private:
  std::size_t size_ = 0;
  std::size_t capacity_;
  std::unordered_map<Cell, Cell> map_;
};

} // namespace pscm