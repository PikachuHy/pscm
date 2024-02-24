#pragma once

namespace pscm::core {
class Value;
class SchemeImpl;

class Scheme {
public:
  Scheme();
  ~Scheme();

  Value *eval(const char *code);
  void set_logger_level(int level);

private:
  SchemeImpl *impl_;
};

} // namespace pscm::core
