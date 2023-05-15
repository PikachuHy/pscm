#include "pscm/Module.h"

namespace pscm {
Cell current_module(Cell args) {
  return new Module();
}
} // namespace pscm