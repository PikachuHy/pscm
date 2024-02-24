#include "Parser.h"
#include "pscm/Parser.h"

#include "Procedure.h"
#include "Value.h"

#include <pscm/Number.h>
#include <pscm/Str.h>
#include <pscm/Symbol.h>
#include <pscm/common_def.h>

#include <iostream>
#include <pscm/Pair.h>
using namespace std::string_literals;

namespace pscm::core {
PSCM_INLINE_LOG_DECLARE("pscm.core.Parser");

Procedure *create_proc(const std::vector<Value *>& list) {
  PSCM_ASSERT(list.size() == 3);
  auto head = list[1];
  std::vector<Value *> body(list.begin() + 2, list.end());
  if (auto p = dynamic_cast<DottedListValue *>(head); p) {
    auto name = dynamic_cast<SymbolValue *>(p->value1()[0]);
    PSCM_ASSERT(name);
    std::vector<SymbolValue *> args;
    args.reserve(p->value1().size() - 1);
    for (int i = 1; i < p->value1().size(); ++i) {
      auto arg = dynamic_cast<SymbolValue *>(p->value1()[i]);
      PSCM_ASSERT(arg);
      args.push_back(arg);
    }
    auto vararg = dynamic_cast<const SymbolValue *>(p->value2());
    auto proc = new Procedure(name, args, body, nullptr, (SymbolValue *)vararg);
    return proc;
  }
  else if (auto p = dynamic_cast<ListValue *>(head); p) {
    auto name = dynamic_cast<SymbolValue *>(p->value()[0]);
    PSCM_ASSERT(name);
    std::vector<SymbolValue *> args;
    args.reserve(p->value().size() - 1);
    for (int i = 1; i < p->value().size(); ++i) {
      auto arg = dynamic_cast<SymbolValue *>(p->value()[i]);
      PSCM_ASSERT(arg);
      args.push_back(arg);
    }
    auto proc = new Procedure(name, args, body, nullptr);
    return proc;
  }
  else {
    PSCM_THROW_EXCEPTION("Unsupported type: " + head->to_string());
  }
}

class ParserImpl {
public:
  ParserImpl(std::string code)
      : parser_(code.c_str()) {
  }

  Value *parse() {
    auto cell = parser_.parse();
    auto ret = convert_cell_to_value(cell);
    return ret;
  }

  Value *convert_cell_to_value(Cell cell) {
    if (cell.is_none()) {
      return nullptr;
    }
    if (cell.is_bool()) {
      auto value = cell.to_bool();
      return value ? (Value *)TrueValue::instance() : (Value *)FalseValue::instance();
    }
    else if (cell.is_sym()) {
      auto value = cell.to_sym();
      return new SymbolValue(cell.to_std_string());
    }
    else if (cell.is_str()) {
      auto value = cell.to_str();
      std::string converted;
      value->str().toUTF8String(converted);
      return new StringValue(std::move(converted));
    }
    else if (cell.is_num()) {
      if (auto value = cell.to_num(); value->is_int()) {
        auto int_value = value->to_int();
        return new IntegerValue(int_value);
      }
      else {
        PSCM_THROW_EXCEPTION("Unsupported number type: " + cell.to_string());
      }
    }
    else if (cell.is_pair()) {
      std::vector<Value *> value_list;
      while (cell.is_pair()) {
        auto item = car(cell);
        auto value = convert_cell_to_value(item);
        PSCM_ASSERT(value);
        value_list.push_back(value);
        cell = cdr(cell);
      }
      if (cell.is_nil()) {
        if (value_list.size() >= 3 && value_list[0]->to_string() == "define") {
          if (auto list = dynamic_cast<ListValue *>(value_list[1]); list) {
            if (auto sym = dynamic_cast<SymbolValue *>(list->value()[0]); sym) {
              auto proc = create_proc(value_list);
              return proc;
            }
          }
        }
        return new ListValue(std::move(value_list));
      }
      else {
        auto last_value = convert_cell_to_value(cell);
        return new DottedListValue(std::move(value_list), last_value);
      }
    }
    else {
      PSCM_THROW_EXCEPTION("Unsupported type: " + cell.to_string());
    }
  }

  pscm::Parser parser_;
};

Parser::Parser(std::string code)
    : impl_(new ParserImpl(std::move(code))) {
}

Parser::~Parser() {
  delete impl_;
}

std::vector<Value *> Parser::parse_all() {
  std::vector<Value *> ret;
  auto value = parse_one();
  while (value != nullptr) {
    ret.push_back(value);
    value = parse_one();
  }
  return ret;
}

Value *Parser::parse_one() {
  PSCM_ASSERT(impl_);
  return impl_->parse();
}
} // namespace pscm::core
