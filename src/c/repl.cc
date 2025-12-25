#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pscm.h"

// Check if parentheses are balanced in the input string
static bool is_balanced(const char *s) {
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  bool in_comment = false;
  
  for (const char *p = s; *p; p++) {
    if (escaped) {
      escaped = false;
      continue;
    }
    
    if (*p == '\\') {
      escaped = true;
      continue;
    }
    
    if (in_comment) {
      if (*p == '\n') {
        in_comment = false;
      }
      continue;
    }
    
    if (*p == '"') {
      in_string = !in_string;
      continue;
    }
    
    if (in_string) {
      continue;
    }
    
    if (*p == ';') {
      in_comment = true;
      continue;
    }
    
    if (*p == '(') {
      depth++;
    } else if (*p == ')') {
      depth--;
      if (depth < 0) {
        return false;  // Unmatched closing parenthesis
      }
    }
  }
  
  return depth == 0;
}


void repl() {
  char line_buffer[4096];  // Buffer for each line
  size_t total_size = 4096;
  char *input_buffer = (char *)malloc(total_size);
  if (!input_buffer) {
    fprintf(stderr, "ERROR: failed to allocate memory\n");
    return;
  }
  input_buffer[0] = '\0';
  
  while (true) {
    // Show prompt: "pscm> " for first line, "  ...> " for continuation
    if (input_buffer[0] == '\0') {
      printf("pscm> ");
    } else {
      printf("  ...> ");
    }
    
    if (fgets(line_buffer, sizeof(line_buffer), stdin) != NULL) {
      // Check if input line is too long
      size_t line_len = strlen(line_buffer);
      if (line_len > 0 && line_buffer[line_len - 1] != '\n') {
        // Input was truncated, clear remaining input
        int c;
        while ((c = getchar()) != '\n' && c != EOF) {
          // Discard remaining characters
        }
        printf("Warning: input line too long, truncated\n");
      }
      
      // Append line to input buffer
      size_t current_len = strlen(input_buffer);
      size_t needed_size = current_len + line_len + 1;
      
      if (needed_size > total_size) {
        // Reallocate buffer if needed
        total_size = needed_size * 2;
        char *new_buffer = (char *)realloc(input_buffer, total_size);
        if (!new_buffer) {
          fprintf(stderr, "ERROR: failed to reallocate memory\n");
          free(input_buffer);
          return;
        }
        input_buffer = new_buffer;
      }
      
      strcat(input_buffer, line_buffer);
      
      // Check if parentheses are balanced
      if (is_balanced(input_buffer)) {
        // Skip whitespace to check if input is empty
        const char *p = input_buffer;
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
          p++;
        }
        
        // Only parse if there's actual content
        if (*p != '\0') {
          // Expression is complete, parse and evaluate
          auto ast = parse(input_buffer);
          print_ast(ast);
          printf("\n");
          auto val = eval(ast);
          printf(" --> ");
          print_ast(val);
          printf("\n");
        }
        
        // Reset buffer for next expression
        input_buffer[0] = '\0';
      }
      // Otherwise, continue reading more lines
    }
    else {
      // EOF or error
      if (feof(stdin)) {
        printf("\n");
        free(input_buffer);
        return;  // Normal exit
      }
      printf("read failed\n");
      free(input_buffer);
      return;
    }
  }
}

