#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/ApiManager.h"
#include "pscm/Cell.h"
#include "pscm/Char.h"
#include "pscm/Function.h"
#include "pscm/Macro.h"
#include "pscm/Pair.h"
#include "pscm/Parser.h"
#include "pscm/Port.h"
#include "pscm/Procedure.h"
#include "pscm/SchemeProxy.h"
#include "pscm/Str.h"
#include "pscm/Symbol.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/icu/ICUCompat.h"
#include "pscm/scm_utils.h"
#include "unicode/ucnv.h"
#if defined(WASM_PLATFORM)
#else
#include "unicode/ustream.h"
#endif
#include "unicode/utf8.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <variant>
#endif
namespace pscm {
PSCM_INLINE_LOG_DECLARE("pscm.core.Port");

std::tuple<UChar32, uint8_t> read_utf8(std::istream& stream) {
  char charbuf[5] = {};
  char *charbufbegin = charbuf;
  char& first = charbuf[0];
  if (stream.eof() || stream.fail()) {
    return std::make_tuple(EOF, 0);
  }
  stream.read(charbuf, 1);
  if (stream.fail()) {
    return std::make_tuple(EOF, stream.gcount());
  }
  else if (U8_IS_SINGLE(first)) {
    PSCM_ASSERT(first != 0);
    return std::make_tuple(first, 1);
  }
  else if (U8_IS_LEAD(first)) {
    uint8_t length = U8_COUNT_TRAIL_BYTES_UNSAFE(first);
    stream.read(&charbuf[1], length);
    if (stream.fail()) {
      PSCM_THROW_EXCEPTION("uncomplete utf-8 codepoint in stream"_u);
    }

    UChar32 ch;
    U8_GET_OR_FFFD(reinterpret_cast<uint8_t *>(charbufbegin), 0, 0, 5, ch);
    if (ch == 0xFFFD) {
      PSCM_WARN("corrupted character, buffer content: {0}", UString(charbuf))
    }
    return std::make_tuple(ch, length + 1);
  }
  else {
    PSCM_THROW_EXCEPTION("incorrect position of stream"_u);
  }
}

void unget_utf8(std::istream& stream, std::tuple<UChar32, uint8_t>& res) {
  for (uint8_t i = 0; i < std::get<1>(res); i++) {
    stream.unget();
  }
}

class StandardPort : public Port {
public:
  StandardPort(bool is_input)
      : is_input_(is_input) {
  }

  ~StandardPort() {
    close();
  }

  bool is_input_port() const override {
    return is_input_;
  }

  bool is_output_port() const override {
    return !is_input_;
  }

  void close() override {
    // do nothing
  }

  UChar32 read_char() override {
    PSCM_ASSERT(is_input_);
    auto res = read_utf8(std::cin);
    return std::get<0>(res);
  }

  UChar32 peek_char() override {
    PSCM_ASSERT(is_input_);
    auto res = read_utf8(std::cin);
    unget_utf8(std::cin, res);
    return std::get<0>(res);
  }

  void write_char(UChar32 ch) override {
    PSCM_ASSERT(!is_input_);
    uint8_t charbuf[5];
    uint8_t offset = 0;
    uint8_t length = U8_LENGTH(ch);
    bool isError = false;
    U8_APPEND(charbuf, offset, 5, ch, isError);
    PSCM_ASSERT(!isError);
    std::cout.write(reinterpret_cast<char *>(charbuf), length);
  }

  UString to_string() const override {
    UString out;
    out += "#";
    out += "<";
    if (is_input_port()) {
      out += "input: standard input";
    }
    else if (is_output_port()) {
      out += "output: standard output";
    }
    else {
      PSCM_ASSERT("Invalid port");
    }
    out += ">";
    return out;
  }

  Cell read() override {
    Parser parser(this);
    auto expr = parser.parse();
    return expr;
  }

  void write(Cell obj) override {
    std::cout << obj.to_string();
  }

  Type type() const override {
    return Type::STANDARD_PORT;
  }

  bool is_input_;
};

class StringReadPort : public Port {
public:
  StringReadPort(const UString& s)
      : iter_(s) {
  }

  bool is_input_port() const override {
    return true;
  }

  bool is_output_port() const override {
    return false;
  }

  void close() override {
    // do nothing
  }

  UChar32 read_char() override {
    UChar32 ch = iter_.next32PostInc();
    if (ch != UIteratorDone) {
      return ch;
    }
    else {
      return EOF;
    }
  }

  UChar32 peek_char() override {
    return iter_.current32();
  }

  void write_char(UChar32 ch) override {
    PSCM_ASSERT(false);
  }

  UString to_string() const override {
    UString out;
    out += "#";
    out += "<";
    out += "input";
    out += ": ";
    out += "string ";
    out += pscm::to_string(this);
    out += ">";
    return out;
  }

  Cell read() override {
    Parser parser(&iter_);
    auto expr = parser.parse();
    return expr;
  }

  void write(Cell obj) override {
    PSCM_ASSERT(false);
  }

  Type type() const override {
    return Type::STRING_PORT;
  }

private:
  UIterator iter_;
};

class StringOutputPort : public Port {
public:
  StringOutputPort() {
  }

  bool is_input_port() const override {
    return false;
  }

  bool is_output_port() const override {
    return true;
  }

  void close() override {
    // do nothing
  }

  UChar32 read_char() override {
    PSCM_ASSERT(false);
    return EOF;
  }

  UChar32 peek_char() override {
    PSCM_ASSERT(false);
    return EOF;
  }

  void write_char(UChar32 ch) override {
    s_.append(ch);
  }

  UString to_string() const override {
    UString out;
    out += "#";
    out += "<";
    if (is_input_port()) {
      out += "input";
    }
    else if (is_output_port()) {
      out += "output";
    }
    else {
      PSCM_ASSERT("Invalid port");
    }
    out += ": ";
    out += "string ";
    out += pscm::to_string(this);
    out += ">";
    return out;
  }

  Cell read() override {
    PSCM_ASSERT(false);
    return Cell::nil();
  }

  void write(Cell obj) override {
    s_ += obj.to_string();
  }

  Type type() const override {
    return Type::STRING_PORT;
  }

  UString str() const {
    return s_;
  }

private:
  UString s_;
};

FilePort::FilePort(const UString& filename, std::ios_base::openmode mode)
    : filename_(filename)
    , mode_(mode) {
  open_fstream(f_, filename_, mode);
  if (!f_.is_open()) {
    PSCM_THROW_EXCEPTION("open file failed: " + filename_);
  }
}

bool FilePort::is_input_port() const {
  return mode_ & std::ios::in;
}

bool FilePort::is_output_port() const {
  return mode_ & std::ios::out;
}

void FilePort::close() {
  f_.close();
}

UChar32 FilePort::read_char() {
  PSCM_ASSERT(is_input_port());
  auto res = read_utf8(f_);
  return std::get<0>(res);
}

UChar32 FilePort::peek_char() {
  PSCM_ASSERT(is_input_port());
  auto res = read_utf8(f_);
  unget_utf8(f_, res);
  return std::get<0>(res);
}

void FilePort::write_char(UChar32 ch) {
  PSCM_ASSERT(is_output_port());
  uint8_t charbuf[5];
  uint8_t offset = 0;
  uint8_t length = U8_LENGTH(ch);
  bool isError = false;
  U8_APPEND(charbuf, offset, 5, ch, isError);
  PSCM_ASSERT(!isError);
  f_.write(reinterpret_cast<char *>(charbuf), length);
}

UString FilePort::to_string() const {
  UString out;
  out += "#";
  out += "<";
  if (is_input_port()) {
    out += "input";
  }
  else if (is_output_port()) {
    out += "output";
  }
  else {
    PSCM_ASSERT("Invalid port");
  }
  out += ": ";
  out += filename_;
  out += ">";
  return out;
}

Cell FilePort::read() {
  Parser parser(this);
  auto expr = parser.parse();
  return expr;
}

void FilePort::write(Cell obj) {
  PSCM_INFO("object written: `{0}`", obj.to_string())
  f_ << obj.to_string();
}

Port::Type FilePort::type() const {
  return Type::FILE_PORT;
}

Cell is_input_port(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_port());
  auto port = arg.to_port();
  return Cell(port->is_input_port());
}

Cell is_output_port(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_port());
  auto port = arg.to_port();
  return Cell(port->is_output_port());
}

Cell current_input_port(Cell args) {
  static StandardPort port(true);
  return Cell(&port);
}

Cell current_output_port(Cell args) {
  static StandardPort port(false);
  return Cell(&port);
}

Cell open_input_file(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_str());
  auto s = arg.to_str();
  PSCM_ASSERT(s);
  auto port = new FilePort(s->str(), std::ios::in);
  return Cell(port);
}

Cell open_output_file(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_str());
  auto s = arg.to_str();
  PSCM_ASSERT(s);
  auto port = new FilePort(s->str(), std::ios::out);
  return Cell(port);
}

Cell close_input_port(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_port());
  auto port = arg.to_port();
  port->close();
  return Cell::none();
}

Cell close_output_port(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_port());
  auto port = arg.to_port();
  port->close();
  return Cell::none();
}

Cell proc_read(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_port());
  auto port = arg.to_port();
  auto expr = port->read();
  return expr;
}

Cell read_char(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_port());
  auto port = arg.to_port();
  auto ch = port->read_char();
  return Char::from(ch);
}

Cell peek_char(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_port());
  auto port = arg.to_port();
  auto ch = port->peek_char();
  return Char::from(ch);
}

Cell is_eof_object(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_char());
  return Cell(arg == Char::from(EOF));
}

Cell is_char_ready(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_THROW_EXCEPTION("not supported now");
}

Cell write_char(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_char());
  auto port = cdr(args);
  auto ch = arg.to_char();
  if (port.is_nil()) {
    UString str(ch->to_string());
    std::cout << str;
  }
  else {
    port = car(port);
    PSCM_ASSERT(port.is_port());
    auto p = port.to_port();
    p->write_char(ch->to_int());
  }
  return Cell::none();
}

PSCM_DEFINE_BUILTIN_PROC(Port, "make-soft-port") {
  PSCM_ASSERT(args.is_pair());
  auto pv = car(args);
  auto mode = cadr(args);
  PSCM_ASSERT(pv.is_vec());
  PSCM_ASSERT(mode.is_str());
  auto v = pv.to_vec();
  auto s = mode.to_str()->str();
  PSCM_ASSERT(v->size() == 5 || v->size() == 6);
  for (size_t i = 0; i < 5; i++) {
    if (!v->at(i).is_proc()) {
      PSCM_THROW_EXCEPTION("Type error, required proc, but got: " + v->at(i).to_string());
    }
  }
  // TODO:
  return Cell::none();
}

Cell transcript_on(Cell args) {
  PSCM_THROW_EXCEPTION("not supported now");
}

Cell transcript_off(Cell args) {
  PSCM_THROW_EXCEPTION("not supported now");
}

Cell display(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  auto port = cdr(args);
  if (port.is_nil()) {
    port = current_output_port(args);
  }
  else {
    port = car(port);
  }
  PSCM_ASSERT(port.is_port());
  auto p = port.to_port();
  arg.display(*p);
  return Cell::none();
}

Cell write(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  auto port = cdr(args);
  if (port.is_nil()) {
    port = current_output_port(args);
  }
  else {
    port = car(port);
  }
  PSCM_ASSERT(port.is_port());
  auto p = port.to_port();
  p->write(arg);
  return Cell::none();
}

Cell builtin_create_string_port(Cell args) {
  if (args.is_nil()) {
    auto port = new StringOutputPort();
    return Cell(port);
  }
  else {
    auto arg = car(args);
    PSCM_ASSERT(arg.is_str());
    auto str = arg.to_str();
    auto port = new StringReadPort(str->str());
    return Cell(port);
  }
}

Cell builtin_string_port_to_string(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_port());
  auto port = arg.to_port();
  PSCM_ASSERT(port->type() == Port::Type::STRING_PORT && port->is_output_port());
  auto sp = dynamic_cast<StringOutputPort *>(port);
  auto s = sp->str();
  return new String(s);
}

/*
(let ((port (builtin_create_string_port)))
      (apply proc (list port))
      (builtin_string_port_to_string port))
*/
Procedure *Procedure::create_call_with_output_string(SymbolTable *env) {
  auto name = new Symbol("call-with-output-string");
  auto proc = new Symbol("proc");
  auto func_create = new Function("builtin_create_string_port", builtin_create_string_port);
  auto func_str = new Function("builtin_string_port_to_string", builtin_string_port_to_string);
  auto port_sym = new Symbol("port");
  auto call_proc = list(new Symbol("apply"), proc, list(new Symbol("list"), port_sym));
  Cell body = list(new Symbol("let"), list(list(port_sym, list(func_create))), call_proc, list(func_str, port_sym));
  PSCM_DEBUG("call-with-output-string body: {0}", body.pretty_string());
  Cell args = list(proc);
  body = cons(body, nil);
  return new Procedure(name, args, body, env);
}

/*
(let ((port (builtin_create_string_port str)))
      (apply proc (list port)))
*/
Procedure *Procedure::create_call_with_input_string(SymbolTable *env) {
  auto name = new Symbol("call-with-input-string");
  auto str = new Symbol("string");
  auto proc = new Symbol("proc");
  auto func_create = new Function("builtin_create_string_port", builtin_create_string_port);
  auto port_sym = new Symbol("port");
  auto call_proc = list(new Symbol("apply"), proc, list(new Symbol("list"), port_sym));
  Cell body = list(new Symbol("let"), list(list(port_sym, list(func_create, str))), call_proc);
  PSCM_DEBUG("call-with-input-string body: {0}", body.pretty_string());
  Cell args = list(str, proc);
  body = cons(body, nil);
  return new Procedure(name, args, body, env);
}

PSCM_DEFINE_BUILTIN_MACRO_PROC_WRAPPER(Port, "load", Label::APPLY_LOAD, "(filename)") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_sym());
  auto sym = arg.to_sym();
  auto val = env->get(sym);
  PSCM_ASSERT(val.is_str());
  auto s = val.to_str();
  auto filename = s->str();
  bool ok = scm.load(filename);
  return Cell(ok);
}

PSCM_DEFINE_BUILTIN_PROC(Port, "force-output") {
  return Cell::none();
}
} // namespace pscm