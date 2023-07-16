//
// Created by PikachuHy on 2023/7/16.
//

#include "pscm/misc/SourceLocation.h"

namespace pscm {

std::string SourceLocation::to_string() const {
  auto name = std::string(filename);
  auto pos = name.find_last_of('/');
  name = name.substr(pos + 1);
  return name + ":" + std::to_string(linenum); // + " " + std::string(funcname);
}

} // namespace pscm