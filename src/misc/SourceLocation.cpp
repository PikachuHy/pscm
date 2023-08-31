//
// Created by PikachuHy on 2023/7/16.
//

#include "pscm/misc/SourceLocation.h"
#include "pscm/misc/ICUCompat.h"

namespace pscm {

UString SourceLocation::to_string() const {
  auto _name = UString(filename);
  auto pos = _name.lastIndexOf('/');
  auto name = UString(filename);
  _name.extractBetween(pos + 1, _name.length(), name);
  return name + ":" + pscm::to_string(linenum); // + " " + UString(funcname);
}

} // namespace pscm