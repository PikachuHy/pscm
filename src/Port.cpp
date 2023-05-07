#include "pscm/Port.h"
#include "pscm/Cell.h"
#include "pscm/Char.h"
#include "pscm/Pair.h"
#include "pscm/Parser.h"
#include "pscm/Str.h"
#include "pscm/common_def.h"
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

// class StringPort : public Port {};

class FilePort : public Port {
public:
  FilePort(std::string_view filename, std::ios_base::openmode mode)
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

} // namespace pscm