#include "Mangler.h"
#include "Value.h"
#include <pscm/common_def.h>

namespace pscm::core {

std::string Mangler::mangle(const std::string& callee, const std::vector<const Type *>& arg_type_list) const {
  PSCM_INLINE_LOG_DECLARE("pscm.core.mangle_name");
  std::stringstream ss;
  ss << callee;
  for (auto arg_type : arg_type_list) {
    ss << "_";
    ss << arg_type->to_string();
  }
  return ss.str();
}
} // namespace pscm::core
