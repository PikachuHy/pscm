module;
#include "fmt/args.h"
#include "fmt/chrono.h"
#include "fmt/color.h"
#include "fmt/compile.h"
#include "fmt/format.h"
#include "fmt/os.h"
#include "fmt/printf.h"
#include "fmt/std.h"
#include "fmt/xchar.h"
#include "format.cc"
#include "os.cc"
#include <fmt/ranges.h>
export module fmt;
export namespace fmt {
using fmt::basic_format_string;
using fmt::format;
using fmt::format_context;
using fmt::format_parse_context;
using fmt::format_to;
using fmt::formatter;
using fmt::print;
using fmt::println;
}