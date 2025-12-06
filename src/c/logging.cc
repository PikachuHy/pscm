#include <stdio.h>
#include <string.h>

void print_basename(const char *path) {
  auto len = strlen(path);
  int i = len;
  while (i > 0) {
    if (path[i] == '/') {
      i++;
      break;
    }
    i--;
  }
  for (int idx = i; idx <= len; idx++) {
    printf("%c", path[idx]);
  }
}
