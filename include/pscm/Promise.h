#pragma once
#include "pscm/Cell.h"
#include <optional>

namespace pscm {
class Promise {
public:
  Promise(Procedure *proc)
      : proc_(proc) {
  }

  bool ready() const {
    return ret_.has_value();
  }

  void set_result(Cell ret);

  Cell result() const;

  Procedure *proc() const {
    return proc_;
  }

  friend std::ostream& operator<<(std::ostream& out, const Promise& proc);

private:
  Procedure *proc_;
  std::optional<Cell> ret_;
};
} // namespace pscm