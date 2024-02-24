#pragma once
#include <memory>
#include <vector>

namespace pscm::core {
class Value;
class ParserImpl;

class Parser {
public:
  explicit Parser(std::string code);
  ~Parser();
  std::vector<Value *> parse_all();
  Value *parse_one();

private:
  ParserImpl *impl_;
};

} // namespace pscm::core
