#include "pscm/Cell.h"
#include <string>
#include <string_view>

namespace pscm {
class FilePort;

class Port {
public:
  enum class Type {
    STANDARD_PORT,
    FILE_PORT,
    STRING_PORT,
  };

  virtual bool is_input_port() const = 0;
  virtual bool is_output_port() const = 0;
  virtual void close() = 0;
  virtual char read_char() = 0;
  virtual char peek_char() = 0;
  virtual void write_char(int ch) = 0;
  virtual std::string to_string() const = 0;
  virtual Cell read() = 0;
  virtual void write(Cell obj) = 0;
  virtual Type type() const = 0;
};

} // namespace pscm