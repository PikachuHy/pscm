#include "pscm/Cell.h"
#include <string>
#include <string_view>

namespace pscm {

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
  virtual UChar32 read_char() = 0;
  virtual UChar32 peek_char() = 0;
  virtual void write_char(UChar32 ch) = 0;
  virtual Cell read() = 0;
  virtual void write(Cell obj) = 0;
  virtual Type type() const = 0;
  virtual UString to_string() const = 0;
};

class FilePort : public Port {
public:
  FilePort(const UString& filename, std::ios_base::openmode mode);

  bool is_input_port() const override;
  bool is_output_port() const override;
  void close() override;
  UChar32 read_char() override;
  UChar32 peek_char() override;
  void write_char(UChar32 ch) override;
  Cell read() override;
  void write(Cell obj) override;
  Type type() const override;
  UString to_string() const override;

private:
  UString filename_;
  std::fstream f_;
  std::ios_base::openmode mode_;
};

} // namespace pscm