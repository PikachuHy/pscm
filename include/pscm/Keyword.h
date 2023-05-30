#include "pscm/Cell.h"

namespace pscm {
class Keyword {
public:
  Keyword(Symbol *sym)
      : sym_(sym) {
  }

  friend std::ostream& operator<<(std::ostream& os, const Keyword& keyword);
  friend bool operator==(const Keyword& lhs, const Keyword& rhs);
  HashCodeType hash_code() const;

private:
  Symbol *sym_;
};
} // namespace pscm