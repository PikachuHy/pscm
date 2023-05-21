#pragma once
#include "pscm/Cell.h"
#include <string>

namespace pscm {
class Module {
public:
  Module(Cell name)
      : name_(name) {
  }

  friend std::ostream& operator<<(std::ostream& os, const Module& m);

private:
  Cell name_;
};
} // namespace pscm