#include "pscm/Port.h"
#include "pscm/ApiManager.h"
#include "pscm/Cell.h"
#include "pscm/Char.h"
#include "pscm/Function.h"
#include "pscm/Macro.h"
#include "pscm/Pair.h"
#include "pscm/Parser.h"
#include "pscm/Procedure.h"
#include "pscm/SchemeProxy.h"
#include "pscm/Str.h"
#include "pscm/Symbol.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <variant>

namespace pscm {
class StandardPort : public Port {
public:
  StandardPort(bool is_input)
      : is_input_(is_input) {
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

  char read_char() override {
    PSCM_ASSERT(is_input_);
    char ch;
    std::cin >> ch;
    return ch;
  }

  char peek_char() override {
    PSCM_ASSERT(is_input_);
    char ch;
    ch = std::cin.peek();
    return ch;
  }

  void write_char(int ch) override {
    PSCM_ASSERT(!is_input_);
    char c = ch;
    std::cout << c;
  }

  std::string to_string() const override {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  Cell read() override {
    Parser parser((std::istream *)&std::cin);
    auto expr = parser.parse();
    return expr;
  }

  void write(Cell obj) override {
    std::cout << obj;
  }

  Type type() const override {
    return Type::STANDARD_PORT;
  }

  friend std::ostream& operator<<(std::ostream& out, const StandardPort& port) {
    out << "#";
    out << "<";
    if (port.is_input_port()) {
      out << "input: standard input";
    }
    else if (port.is_output_port()) {
      out << "output: standard output";
    }
    else {
      PSCM_ASSERT("Invalid port");
    }
    out << ">";
    return out;
  }

  bool is_input_;
};

class StringPort : public Port {
public:
  StringPort(std::ios_base::openmode mode)
      : mode_(mode) {
  }

  StringPort(std::string s, std::ios_base::openmode mode)
      : mode_(mode) {
    ss_ << s;
  }

  bool is_input_port() const override {
    return mode_ & std::ios::in;
  }

  bool is_output_port() const override {
    return mode_ & std::ios::out;
  }

  void close() override {
    // do nothing
  }

  char read_char() override {
    PSCM_ASSERT(is_input_port());
    if (ss_.eof()) {
      return EOF;
    }
    char ch;
    ss_.read(&ch, 1);
    return ch;
  }

  char peek_char() override {
    PSCM_ASSERT(is_input_port());
    char ch;
    ch = ss_.peek();
    return ch;
  }

  void write_char(int ch) override {
    PSCM_ASSERT(is_output_port());
    ss_.write((const char *)&ch, 1);
  }

  std::string to_string() const override {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  Cell read() override {
    Parser parser((std::istream *)&ss_);
    auto expr = parser.parse();
    return expr;
  }

  void write(Cell obj) override {
    ss_ << obj;
  }

  Type type() const override {
    return Type::STRING_PORT;
  }

  std::string str() const {
    return ss_.str();
  }

  friend std::ostream& operator<<(std::ostream& out, const StringPort& port) {
    out << "#";
    out << "<";
    if (port.is_input_port()) {
      out << "input";
    }
    else if (port.is_output_port()) {
      out << "output";
    }
    else {
      PSCM_ASSERT("Invalid port");
    }
    out << ": ";
    out << "string ";
    out << (void *)&port;
    out << ">";
    return out;
  }

private:
  std::stringstream ss_;
  std::ios_base::openmode mode_;
};

class FilePort : public Port {
public:
  FilePort(StringView filename, std::ios_base::openmode mode)
      : filename_(filename)
      , mode_(mode) {
    f_.open(filename_, mode);
    if (!f_.is_open()) {
      PSCM_THROW_EXCEPTION("open file failed: " + filename_);
    }
  }

  bool is_input_port() const override {
    return mode_ & std::ios::in;
  }

  bool is_output_port() const override {
    return mode_ & std::ios::out;
  }

  void close() override {
    f_.close();
  }

  char read_char() override {
    PSCM_ASSERT(is_input_port());
    if (f_.eof()) {
      return EOF;
    }
    char ch;
    f_.read(&ch, 1);
    return ch;
  }

  char peek_char() override {
    PSCM_ASSERT(is_input_port());
    char ch;
    ch = f_.peek();
    return ch;
  }

  void write_char(int ch) override {
    PSCM_ASSERT(is_output_port());
    f_.write((const char *)&ch, 1);
  }

  std::string to_string() const override {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  Cell read() override {
    Parser parser((std::istream *)&f_);
    auto expr = parser.parse();
    return expr;
  }

  void write(Cell obj) override {
    f_ << obj;
  }

  Type type() const override {
    return Type::FILE_PORT;
  }

  friend std::ostream& operator<<(std::ostream& out, const FilePort& port) {
    out << "#";
    out << "<";
    if (port.is_input_port()) {
      out << "input";
    }
    else if (port.is_output_port()) {
      out << "output";
    }
    else {
      PSCM_ASSERT("Invalid port");
    }
    out << ": ";
    out << port.filename_;
    out << ">";
    return out;
  }

private:
  std::string filename_;
  std::fstream f_;
  std::ios_base::openmode mode_;
};

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
    std::cout << char(ch->to_int());
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
    auto port = new StringPort(std::ios::out);
    return Cell(port);
  }
  else {
    auto arg = car(args);
    PSCM_ASSERT(arg.is_str());
    auto str = arg.to_str();
    auto port = new StringPort(str->str(), std::ios::in);
    return Cell(port);
  }
}

Cell builtin_string_port_to_string(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_port());
  auto port = arg.to_port();
  PSCM_ASSERT(port->type() == Port::Type::STRING_PORT);
  auto sp = (StringPort *)port;
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
  SPDLOG_DEBUG("call-with-output-string body: {}", body.pretty_string());
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
  SPDLOG_DEBUG("call-with-input-string body: {}", body.pretty_string());
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
  bool ok = scm.load(std::string(filename).c_str());
  return Cell(ok);
}

PSCM_DEFINE_BUILTIN_PROC(Port, "force-output") {
  return Cell::none();
}
} // namespace pscm