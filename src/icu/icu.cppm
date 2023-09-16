module;
#include "pscm/icu/Displayable.h"
export module pscm.icu;

export namespace pscm {
using U_ICU_NAMESPACE::operator<<;
using pscm::operator<<;
using pscm::operator""_u;
using pscm::to_string;
using pscm::UFormattable;
using pscm::UFormatter;
using pscm::UIterator;
using pscm::UIteratorP;
using pscm::USet;
using pscm::UString;
} // namespace pscm