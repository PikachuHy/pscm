//
// Created by PikachuHy on 2023/7/16.
//

#pragma once
#include <string>

namespace pscm {

struct SourceLocation {
  constexpr SourceLocation(const char *filename = __builtin_FILE(), unsigned int linenum = __builtin_LINE(),
                           const char *funcname = __builtin_FUNCTION())
      : filename(filename)
      , linenum(linenum)
      , funcname(funcname) {
  }

  const char *filename;
  unsigned int linenum;
  const char *funcname;

  std::string to_string() const;
};

} // namespace pscm
