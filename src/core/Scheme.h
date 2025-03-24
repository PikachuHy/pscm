#pragma once
#include <string>

namespace pscm::core {
class Value;
class SchemeImpl;

class Scheme {
public:
  Scheme();
  ~Scheme();
  bool load(const std::string& filename, bool print = false);
  Value *eval(const char *code);
  void set_logger_level(int level);

private:
  SchemeImpl *impl_;
};

} // namespace pscm::core
