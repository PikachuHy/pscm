#include "pscm/Cell.h"
#include <functional>
#include <unordered_map>

namespace pscm {
class HashTable {
public:
  HashTable(std::size_t capacity = 32)
      : capacity_(capacity) {
    map_.resize(capacity);
    std::fill(map_.begin(), map_.end(), nil);
    size_ = 0;
  }

  UString to_string() const;
  Cell set(Cell key, Cell value, Cell::ScmCmp cmp_func = Cell::is_equal);
  Cell get(Cell key, Cell::ScmCmp cmp_func = Cell::is_equal);
  Cell remove(Cell key, Cell::ScmCmp cmp_func = Cell::is_equal);
  void for_each(std::function<void(Cell, Cell)> func);
  void for_each_handle(std::function<void(Cell)> func);

private:
  Cell::Vec map_;
  std::size_t size_;
  std::size_t capacity_;
};

} // namespace pscm