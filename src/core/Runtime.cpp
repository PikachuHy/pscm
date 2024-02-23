
#include "Runtime.h"
#include <pscm/common_def.h>

namespace pscm::core {
PSCM_INLINE_LOG_DECLARE("pscm.core.Runtime");

int64_t car_array(Array *input) {
  PSCM_ASSERT(input);
  PSCM_ASSERT(input->size > 0);
  return input->data[0];
}

} // namespace pscm::core
