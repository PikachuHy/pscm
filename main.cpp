//
// Created by PikachuHy on 2023/4/5.
//
#include <iostream>
#include <pscm/Scheme.h>
#include <spdlog/spdlog.h>
#include <string>
using namespace pscm;

void show_usage() {
  std::cout << R"(
Usage: pscm OPTION ...
Evaluate Scheme code, interactively or from a script.

  [-s] FILE      load Scheme source code from FILE, and exit

  -h, --help     display this help and exit
  -v, --version  display version information and exit

please report bugs to https://github.com/PikachuHy/pscm/issues
)";
}

int main(int argc, char **argv) {
  spdlog::set_level(spdlog::level::err);
  Scheme scm(true);
  auto version = scm.eval("(version)");
  std::cout << "Welcome to PikachuHy's Scheme" << std::endl;
  std::cout << "version: " << version << std::endl;
  int index = 1;
  while (index < argc) {
    std::string arg = argv[index];
    if (arg == "-v" || arg == "--version") {
      std::cout << "PikachuHy's Scheme " << version << std::endl;
      std::cout << "Copyright (c) 2023 PikachuHy" << std::endl;
      return 0;
    }
    else if (arg == "-h" || arg == "--help") {
      show_usage();
      return 0;
    }
    else if (arg == "-s") {
      if (index + 1 < argc) {
        scm.load(argv[index + 1]);
        return 0;
      }
      else {
        std::cout << "missing argument to `-s' switch" << std::endl;
        show_usage();
        return 0;
      }
    }
    else {
      scm.load(argv[index]);
      return 0;
    }
  }
  return 0;
}
