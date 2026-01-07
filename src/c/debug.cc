#include "pscm.h"
#include "eval.h"
#include <string.h>
#include <stdint.h>

// Type for option values (compatible with Guile 1.8)
typedef uintptr_t scm_t_bits;

// Debug options structure (compatible with Guile 1.8)
struct scm_t_debug_option {
  unsigned int type;      // SCM_OPTION_BOOLEAN, SCM_OPTION_INTEGER, SCM_OPTION_SCM
  const char *name;       // Option name
  scm_t_bits val;         // Option value
  const char *doc;        // Documentation string
};

// Option type constants
#define SCM_OPTION_BOOLEAN 0
#define SCM_OPTION_INTEGER 1
#define SCM_OPTION_SCM     2

// Debug options array (similar to Guile 1.8)
// Based on Guile 1.8's scm_debug_opts in eval.c
static scm_t_debug_option scm_debug_opts[] = {
  { SCM_OPTION_BOOLEAN, "cheap", 1,
    "*This option is now obsolete.  Setting it has no effect." },
  { SCM_OPTION_BOOLEAN, "breakpoints", 0, "*Check for breakpoints." },
  { SCM_OPTION_BOOLEAN, "trace", 0, "*Trace mode." },
  { SCM_OPTION_BOOLEAN, "procnames", 1,
    "Record procedure names at definition." },
  { SCM_OPTION_BOOLEAN, "backwards", 0,
    "Display backtrace in anti-chronological order." },
  { SCM_OPTION_INTEGER, "width", 79, "Maximal width of backtrace." },
  { SCM_OPTION_INTEGER, "indent", 10, "Maximal indentation in backtrace." },
  { SCM_OPTION_INTEGER, "frames", 3,
    "Maximum number of tail-recursive frames in backtrace." },
  { SCM_OPTION_INTEGER, "maxdepth", 1000,
    "Maximal number of stored backtrace frames." },
  { SCM_OPTION_INTEGER, "depth", 20, "Maximal length of printed backtrace." },
  { SCM_OPTION_BOOLEAN, "backtrace", 0, "Show backtrace on error." },
  { SCM_OPTION_BOOLEAN, "debug", 0, "Use the debugging evaluator." },
  { SCM_OPTION_INTEGER, "stack", 20000, "Stack size limit (measured in words; 0 = no check)." },
  { SCM_OPTION_SCM, "show-file-name", 1, "Show file names and line numbers in backtraces when not `#f'.  A value of `base' displays only base names, while `#t' displays full names."},
  { SCM_OPTION_BOOLEAN, "warn-deprecated", 0, "Warn when deprecated features are used." }
};

#define SCM_N_DEBUG_OPTIONS 15

// Helper function to find option by name
static int find_debug_option(const char *name) {
  for (unsigned int i = 0; i < SCM_N_DEBUG_OPTIONS; i++) {
    if (strcmp(scm_debug_opts[i].name, name) == 0) {
      return (int)i;
    }
  }
  return -1;
}

// Get current option setting as a list
// Format matches Guile 1.8: (name value name value ...) or (name name ...) for boolean options
// For boolean options, only enabled ones are included (just the name)
// For SCM/INTEGER options, always include name and value
static SCM_List *get_debug_option_setting() {
  SCM_List dummy = make_list_dummy();
  SCM_List *tail = &dummy;
  
  // Process options in reverse array order to match Guile's scm_cons behavior
  // Start from the last option and work backwards
  for (int i = SCM_N_DEBUG_OPTIONS - 1; i >= 0; i--) {
    if (scm_debug_opts[i].type == SCM_OPTION_BOOLEAN) {
      // Only include enabled boolean options
      if (scm_debug_opts[i].val) {
        SCM_Symbol *sym = make_sym(scm_debug_opts[i].name);
        SCM_List *node = make_list(wrap(sym));
        tail->next = node;
        tail = node;
      }
    } else {
      // Include all SCM and INTEGER options with their values
      SCM_Symbol *sym = make_sym(scm_debug_opts[i].name);
      SCM *val = nullptr;
      
      if (scm_debug_opts[i].type == SCM_OPTION_INTEGER) {
        // Create a number from the integer value
        val = long_to_scm((long)scm_debug_opts[i].val);
      } else if (scm_debug_opts[i].type == SCM_OPTION_SCM) {
        val = (SCM *)(intptr_t)scm_debug_opts[i].val;
        if (!val) {
          val = scm_bool_false();
        } else if (val == (SCM *)(intptr_t)1) {
          val = scm_bool_true();
        }
      }
      
      if (val) {
        SCM_List *name_node = make_list(wrap(sym));
        SCM_List *val_node = make_list(val);
        name_node->next = val_node;
        tail->next = name_node;
        tail = val_node;
      }
    }
  }
  
  return dummy.next;
}

// Helper to extract symbol from possibly quoted form
static SCM_Symbol *extract_symbol(SCM *arg) {
  if (!arg) return nullptr;
  
  // Check if it's a quote form: (quote symbol)
  if (is_pair(arg)) {
    SCM_List *arg_list = cast<SCM_List>(arg);
    if (arg_list->data && is_sym(arg_list->data)) {
      SCM_Symbol *quote_sym = cast<SCM_Symbol>(arg_list->data);
      if (strcmp(quote_sym->data, "quote") == 0 && arg_list->next && arg_list->next->data) {
        if (is_sym(arg_list->next->data)) {
          return cast<SCM_Symbol>(arg_list->next->data);
        }
      }
    }
  } else if (is_sym(arg)) {
    return cast<SCM_Symbol>(arg);
  }
  
  return nullptr;
}

// Change option setting according to args list
// Note: When args is provided, it REPLACES the current setting (like Guile 1.8)
static void change_debug_option_setting(SCM_List *args) {
  // If args is nullptr, don't change anything (preserve current values)
  if (!args) {
    return;
  }
  
  // Create a temporary copy of current values
  scm_t_bits temp_vals[SCM_N_DEBUG_OPTIONS];
  for (unsigned int i = 0; i < SCM_N_DEBUG_OPTIONS; i++) {
    temp_vals[i] = scm_debug_opts[i].val;  // Copy current value for all options
  }
  
  // Now reset boolean options to false (they will be set if in args)
  for (unsigned int i = 0; i < SCM_N_DEBUG_OPTIONS; i++) {
    if (scm_debug_opts[i].type == SCM_OPTION_BOOLEAN) {
      temp_vals[i] = 0;  // Initialize to false (will be set if in args)
    }
  }
  
  // Process args list
  // Format: (name value name value ...) or (name name ...) for boolean options
  SCM_List *current = args;
  while (current) {
    if (!current->data) {
      current = current->next;
      continue;
    }
    
    // Skip boolean values (#t, #f) - they are values from previous options
    if (is_bool(current->data)) {
      current = current->next;
      continue;
    }
    
    // Extract symbol from possibly quoted form
    SCM_Symbol *name_sym = extract_symbol(current->data);
    if (!name_sym) {
      // Not a symbol - might be a value from previous option, skip it
      current = current->next;
      continue;
    }
    
    int opt_idx = find_debug_option(name_sym->data);
    
    if (opt_idx < 0) {
      eval_error("debug-options: unknown option name: %s", name_sym->data);
      return;
    }
    
    switch (scm_debug_opts[opt_idx].type) {
      case SCM_OPTION_BOOLEAN:
        // For boolean options, check if next item is #f
        // If it's #f, don't set the option (leave it as false)
        // Otherwise, set it to true
        if (current->next && current->next->data) {
          SCM *next_val = current->next->data;
          // Check if it's #f
          if (is_bool(next_val) && next_val == scm_bool_false()) {
            // Don't set the option (leave it as false)
            current = current->next->next;
          } else {
            // Set to true
            temp_vals[opt_idx] = 1;
            current = current->next;
          }
        } else {
          // No value, set to true (just the option name means enabled)
          temp_vals[opt_idx] = 1;
          current = current->next;
        }
        break;
      case SCM_OPTION_INTEGER:
        if (!current->next || !current->next->data) {
          eval_error("debug-options: missing value for option: %s", name_sym->data);
          return;
        }
        if (!is_num(current->next->data)) {
          eval_error("debug-options: expected number for option: %s", name_sym->data);
          return;
        }
        temp_vals[opt_idx] = (scm_t_bits)(intptr_t)current->next->data->value;
        current = current->next->next;
        break;
      case SCM_OPTION_SCM:
        if (!current->next || !current->next->data) {
          eval_error("debug-options: missing value for option: %s", name_sym->data);
          return;
        }
        // Extract value from possibly quoted form
        SCM *value = current->next->data;
        if (is_pair(value)) {
          SCM_List *value_list = cast<SCM_List>(value);
          if (value_list->data && is_sym(value_list->data)) {
            SCM_Symbol *quote_sym = cast<SCM_Symbol>(value_list->data);
            if (strcmp(quote_sym->data, "quote") == 0 && value_list->next && value_list->next->data) {
              value = value_list->next->data;
            }
          }
        }
        // For SCM options, if value is #f, set to 0 (SCM_BOOL_F)
        // If value is #t, set to 1
        // Otherwise, store the pointer value
        if (is_bool(value)) {
          if (value == scm_bool_false()) {
            temp_vals[opt_idx] = 0;  // SCM_BOOL_F is represented as 0
          } else {
            temp_vals[opt_idx] = 1;  // SCM_BOOL_T is represented as 1
          }
        } else {
          temp_vals[opt_idx] = (scm_t_bits)(intptr_t)value;
        }
        current = current->next->next;
        break;
    }
  }
  
  // Apply changes
  for (unsigned int i = 0; i < SCM_N_DEBUG_OPTIONS; i++) {
    scm_debug_opts[i].val = temp_vals[i];
  }
}

// debug-options-interface: Core function for managing debug options
SCM *scm_c_debug_options_interface(SCM_List *args) {
  if (!args || !args->data) {
    // No arguments: return current option setting
    SCM_List *setting = get_debug_option_setting();
    return setting ? wrap(setting) : scm_nil();
  }
  
  // Check if argument is a list (option setting) or something else (documentation request)
  if (is_pair(args->data)) {
    // It's a list: apply option setting
    SCM_List *old_setting = get_debug_option_setting();
    change_debug_option_setting(cast<SCM_List>(args->data));
    // Return new setting (not old setting, to match Guile behavior)
    SCM_List *new_setting = get_debug_option_setting();
    return new_setting ? wrap(new_setting) : scm_nil();
  } else {
    // Not a list: return documented option setting (for help)
    // For now, just return current setting
    SCM_List *setting = get_debug_option_setting();
    return setting ? wrap(setting) : scm_nil();
  }
}

// Initialize debug options system
void init_debug_options() {
  // Register Scheme functions
  scm_define_vararg_function("debug-options-interface", scm_c_debug_options_interface);
  
  // Define debug-set!, debug-enable, and debug-disable as macros (like Guile 1.8)
  // debug-set! expands to: (debug-options-interface (append (debug-options-interface) (list 'opt val)))
  const char *debug_set_macro_def = 
    "(define-macro (debug-set! opt val)\n"
    "  `(debug-options-interface\n"
    "     (append (debug-options-interface)\n"
    "             (list ',opt ,val))))";
  
  // debug-enable: append flags to current options
  const char *debug_enable_macro_def =
    "(define-macro (debug-enable . flags)\n"
    "  `(debug-options-interface\n"
    "     (append (list ,@flags)\n"
    "             (debug-options-interface))))";
  
  // debug-disable: remove flags from current options
  const char *debug_disable_macro_def =
    "(define-macro (debug-disable . flags)\n"
    "  `(debug-options-interface\n"
    "     (let ((options (debug-options-interface))\n"
    "           (flags-to-remove (list ,@flags)))\n"
    "       (let filter-options ((remaining options)\n"
    "                            (result '()))\n"
    "         (cond\n"
    "          ((null? remaining) (reverse result))\n"
    "          ((and (pair? remaining)\n"
    "                (memq (car remaining) flags-to-remove))\n"
    "           (filter-options (cdr remaining) result))\n"
    "          ((pair? remaining)\n"
    "           (filter-options (cdr remaining) (cons (car remaining) result)))\n"
    "          (else result))))))";
  
  SCM *result1 = scm_c_eval_string(debug_set_macro_def);
  if (!result1) {
    eval_error("Failed to define debug-set! macro");
  }
  
  SCM *result2 = scm_c_eval_string(debug_enable_macro_def);
  if (!result2) {
    eval_error("Failed to define debug-enable macro");
  }
  
  SCM *result3 = scm_c_eval_string(debug_disable_macro_def);
  if (!result3) {
    eval_error("Failed to define debug-disable macro");
  }
}

