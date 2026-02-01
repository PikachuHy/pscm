#include <csignal>
#include <cstdlib>
#include <cxxabi.h>   // __cxa_demangle
#include <dlfcn.h>    // Dl_info
#include <execinfo.h> // backtrace
#include <iostream>
#include <sstream>
#include <vector>

#if defined(__APPLE__)
#include <mach-o/dyld.h> // macOS
#endif
#include <unistd.h>
#include "eval.h"  // For print_eval_stack()

std::string get_executable_path() {
#if defined(__APPLE__)
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::vector<char> buf(size);
  _NSGetExecutablePath(buf.data(), &size);
  return buf.data();
#elif defined(__linux__)
  char buf[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (len != -1) {
    buf[len] = '\0';
    return buf;
  }
#endif
  return "";
}

void print_stacktrace(int signal) {
  void *callstack[128];
  int frames = backtrace(callstack, 128);

  std::cerr << "\nStack trace:\n";
  std::string s;
  for (int i = 3; i < frames - 1; i++) {
    char addr[32] = { 0 };
    snprintf(addr, 32, " %p", callstack[i]);
    s += addr;
  }

  char cmd[512] = { 0 };
  snprintf(cmd, 512, "atos -p %d -o %s %s", getpid(), get_executable_path().c_str(), s.c_str());
  FILE *pipe = popen(cmd, "r");
  if (!pipe)
    return;

  char buffer[256] = { 0 };
  std::string result;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result += buffer;
  }
  pclose(pipe);
  std::istringstream iss(result);
  std::string line;
  int idx = 0;
  while (std::getline(iss, line)) {
    printf("#%d %s\n", idx, line.c_str());
    idx++;
  }
  printf("\n");
  
  // If pscm is currently evaluating, also print the eval call stack
  print_eval_stack();
  
  exit(1);
}

void setup_abort_handler() {
  signal(SIGABRT, print_stacktrace);
  signal(SIGSEGV, print_stacktrace);
  signal(SIGILL, print_stacktrace);
}
