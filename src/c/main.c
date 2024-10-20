#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdnoreturn.h>
#include <string.h>

jmp_buf my_jump_buffer;

// noreturn void foo(int status) {
//   printf("foo(%d) called\n", status);
//   longjmp(my_jump_buffer, status + 1); // will return status+1 out of setjmp
// }

// int main(void) {
//   volatile int count = 0;          // modified local vars in setjmp scope must be volatile
//   if (setjmp(my_jump_buffer) != 5) // compare against constant in an if
//     foo(++count);
// }

long *cont_base;
long *cont_top;
long stack_len;
long *stack_data;
int count = 0;

void f() {
  printf("enter f\n");
  int i = 1;
  for (; i < 4; i++) {
    printf("hi %d\n", i);
    if (i == 2) {
      long stack_top;
      cont_top = &stack_top;
      stack_len = (long)cont_base - (long)cont_top;
      printf("%p - %p\n", cont_base, cont_top);
      printf("stack len: %ld\n", stack_len);
      stack_data = (long *)malloc(sizeof(long) * stack_len);
      memcpy((void *)stack_data, (void *)cont_top, sizeof(long) * stack_len);
      int ret = setjmp(my_jump_buffer);
      if (ret != 0) {
        // memcpy(cont_top, stack_data, sizeof(long) * stack_len);
        printf("jump back: %d\n", i);
        // exit(0);
      }
      //   i = ret;
      int tmp = i;
      printf("setjmp %d, i = %d\n", ret, i);
      printf("tmp: %d\n", tmp);
    }
  }
}

int main() {
  long stack_base;
  cont_base = &stack_base;
  f();
  count++;
  printf("count: %d\n", count);
  if (count < 3) {
    printf("longjmp %d\n", count);
    memcpy(cont_top, stack_data, sizeof(long) * stack_len);
    longjmp(my_jump_buffer, count);
  }
  printf("count: %d\n", count);
  memcpy(cont_top, stack_data, sizeof(long) * stack_len);
  longjmp(my_jump_buffer, count);
  // f();
  //   longjmp(my_jump_buffer, 22);
  // longjmp(my_jump_buffer, count);
  return 0;
}