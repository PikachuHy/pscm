//
// Created by PikachuHy on 2023/4/5.
//
#include <unicode/ustream.h>
#ifdef PSCM_USE_CXX20_MODULES
import pscm;
import std;
#else
#include <iostream>
#include <pscm/Scheme.h>
#include <string>
#endif
using namespace std::string_literals;
using namespace pscm;

void show_usage() {
  std::cout << R"(
Usage: pscm OPTION ...
Evaluate Scheme code, interactively or from a script.

  [-s] FILE      load Scheme source code from FILE, and exit

  -h, --help     display this help and exit
  -m, --mode     evaluation mode: DIRECT REGISTER_MACHINE
  -v, --version  display version information and exit

please report bugs to https://github.com/PikachuHy/pscm/issues
)";
}

int main(int argc, char **argv) {
  bool use_register_machine = false;
  int index = 1;
  while (index < argc) {
    std::string arg = argv[index];
    if (arg == "-v" || arg == "--version") {
      Scheme scm;
      auto version = scm.eval("(version)");
      std::cout << "PikachuHy's Scheme " << version.to_string() << std::endl;
      std::cout << "Copyright (c) 2023 PikachuHy" << std::endl;
      return 0;
    }
    if (arg == "-m" || arg == "--mode") {
      if (index + 1 < argc) {
        auto val = argv[index + 1];
        if (val == "DIRECT"s) {
          index += 2;
          use_register_machine = false;
        }
        else if (val == "REGISTER_MACHINE"s) {
          index += 2;
          use_register_machine = true;
        }
        else {
          std::cout << "bad -m value: " << val << std::endl;
          show_usage();
          return 0;
        }
      }
      else {
        std::cout << "missing argument to `-m' switch" << std::endl;
        show_usage();
        return 0;
      }
    }
    else if (arg == "-h" || arg == "--help") {
      show_usage();
      return 0;
    }
    else if (arg == "-s" || arg == "--test") {
      Scheme new_scm(use_register_machine);
      if (index + 1 < argc) {
        bool ok = new_scm.load(argv[index + 1], arg == "--test");
        if (!ok) {
          return 1;
        }
        return 0;
      }
      else {
        std::cout << "missing argument to `-s' switch" << std::endl;
        show_usage();
        return 0;
      }
    }
    else {
      Scheme new_scm(use_register_machine);
      bool ok = new_scm.load(argv[index]);
      if (!ok) {
        return 1;
      }
      return 0;
    }
  }
  Scheme scm;
  auto version = scm.eval("(version)");
  std::cout << "Welcome to PikachuHy's Scheme" << std::endl;
  std::cout << "version: " << version.to_string() << std::endl;
  scm.repl();
  return 0;
}
