#include "pscm/Cell.h"

namespace pscm {
class Keyword {
public:
  Keyword(Symbol *sym)
      : sym_(sym) {
  }

  friend std::ostream& operator<<(std::ostream& os, const Keyword& keyword);

private:
  Symbol *sym_;
};
} // namespace pscm