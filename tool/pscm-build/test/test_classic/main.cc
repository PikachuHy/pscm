#include <iostream>

import a;
import c;
import d;

auto main() -> int {
  a::a();
  b::b();
  c::c_interface();
  c::c_implementation();
  d::d_interface();
  d::d_implementation();
  return 0;
}