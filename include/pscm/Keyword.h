#include "pscm/Cell.h"

namespace pscm {
class Keyword {
public:
  Keyword(Symbol *sym)
      : sym_(sym) {
  }

  friend bool operator==(const Keyword& lhs, const Keyword& rhs);
  HashCodeType hash_code() const;
  UString to_string() const;

  Symbol *sym() const {
    return sym_;
  }

private:
  Symbol *sym_;
};
} // namespace pscm