#include "pscm.h"
#include "eval.h"
#include <string.h>
#include <stdint.h>

// Type for option values (compatible with Guile 1.8)
typedef uintptr_t scm_t_bits;

// Read options structure (compatible with Guile 1.8)
struct scm_t_read_option {
  unsigned int type;      // SCM_OPTION_BOOLEAN, SCM_OPTION_INTEGER, SCM_OPTION_SCM
  const char *name;       // Option name
  scm_t_bits val;         // Option value
  const char *doc;        // Documentation string
};

// Option type constants
#define SCM_OPTION_BOOLEAN 0
#define SCM_OPTION_INTEGER 1
#define SCM_OPTION_SCM     2

// Read options array (similar to Guile 1.8)
// Note: Guile 1.8 default is (keywords #f positions)
// positions is enabled by default in Guile 1.8
static scm_t_read_option scm_read_opts[] = {
  { SCM_OPTION_BOOLEAN, "copy", 0,
    "Copy source code expressions." },
  { SCM_OPTION_BOOLEAN, "positions", 1,  // Default enabled in Guile 1.8
    "Record positions of source code expressions." },
  { SCM_OPTION_BOOLEAN, "case-insensitive", 0,
    "Convert symbols to lower case." },
  { SCM_OPTION_SCM, "keywords", 0,  // 0 means SCM_BOOL_F (false)
    "Style of keyword recognition: #f, 'prefix or 'postfix." }
};

#define SCM_N_READ_OPTIONS 4

// Accessor macros for read options
#define SCM_COPY_SOURCE_P      scm_read_opts[0].val
#define SCM_RECORD_POSITIONS_P scm_read_opts[1].val
#define SCM_CASE_INSENSITIVE_P scm_read_opts[2].val
#define SCM_KEYWORD_STYLE      scm_read_opts[3].val

// Helper function to find option by name
static int find_read_option(const char *name) {
  for (unsigned int i = 0; i < SCM_N_READ_OPTIONS; i++) {
    if (strcmp(scm_read_opts[i].name, name) == 0) {
      return (int)i;
    }
  }
  return -1;
}

// Get current option setting as a list
// Format matches Guile 1.8: (name value name value ...) or (name name ...) for boolean options
// For boolean options, only enabled ones are included (just the name)
// For SCM options, always include name and value (even if #f)
// Note: Guile 1.8 builds the list in reverse order (using scm_cons), so we need to match that
// Guile 1.8 order: copy, positions, case-insensitive, keywords (array order)
// But scm_cons builds backwards, so final order is: keywords, case-insensitive, positions, copy
// However, Guile 1.8 output shows: (keywords #f positions), which means it's in array order
// Let's check: Guile uses scm_cons which prepends, so if we process 0,1,2,3 we get 3,2,1,0
// But output is (keywords #f positions), which is index 3, then index 1
// So it seems Guile processes in order 0,1,2,3 but only includes enabled/SCM options
// Let's match Guile's output exactly: (keywords #f positions)
static SCM_List *get_read_option_setting() {
  SCM_List dummy = make_list_dummy();
  SCM_List *tail = &dummy;
  
  // Process options in array order: copy, positions, case-insensitive, keywords
  // But build in reverse order to match Guile's scm_cons behavior
  // Actually, let's just match Guile's output: (keywords #f positions)
  // This means: keywords (index 3), then positions (index 1)
  // So we process in reverse: 3, 2, 1, 0, but only add enabled/SCM options
  
  // First add keywords (SCM option, always included)
  {
    SCM_Symbol *sym = make_sym(scm_read_opts[3].name);
    SCM *val = (SCM *)(intptr_t)scm_read_opts[3].val;
    if (!val) {
      val = scm_bool_false();
    }
    SCM_List *name_node = make_list(wrap(sym));
    SCM_List *val_node = make_list(val);
    name_node->next = val_node;
    tail->next = name_node;
    tail = val_node;
  }
  
  // Then add case-insensitive if enabled (index 2)
  if (scm_read_opts[2].val) {
    SCM_Symbol *sym = make_sym(scm_read_opts[2].name);
    SCM_List *node = make_list(wrap(sym));
    tail->next = node;
    tail = node;
  }
  
  // Then add positions if enabled (index 1)
  if (scm_read_opts[1].val) {
    SCM_Symbol *sym = make_sym(scm_read_opts[1].name);
    SCM_List *node = make_list(wrap(sym));
    tail->next = node;
    tail = node;
  }
  
  // Finally add copy if enabled (index 0)
  if (scm_read_opts[0].val) {
    SCM_Symbol *sym = make_sym(scm_read_opts[0].name);
    SCM_List *node = make_list(wrap(sym));
    tail->next = node;
    tail = node;
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
// So we need to initialize all boolean options to false, then set only the ones in args
// However, we need to preserve the current values first, then reset boolean options
static void change_read_option_setting(SCM_List *args) {
  // If args is nullptr, don't change anything (preserve current values)
  if (!args) {
    return;
  }
  
  // Create a temporary copy of current values
  scm_t_bits temp_vals[SCM_N_READ_OPTIONS];
  for (unsigned int i = 0; i < SCM_N_READ_OPTIONS; i++) {
    temp_vals[i] = scm_read_opts[i].val;  // Copy current value for all options
  }
  
  // Now reset boolean options to false (they will be set if in args)
  for (unsigned int i = 0; i < SCM_N_READ_OPTIONS; i++) {
    if (scm_read_opts[i].type == SCM_OPTION_BOOLEAN) {
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
    
    int opt_idx = find_read_option(name_sym->data);
    
    if (opt_idx < 0) {
      eval_error("read-options: unknown option name: %s", name_sym->data);
      return;
    }
    
    switch (scm_read_opts[opt_idx].type) {
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
          eval_error("read-options: missing value for option: %s", name_sym->data);
          return;
        }
        if (!is_num(current->next->data)) {
          eval_error("read-options: expected number for option: %s", name_sym->data);
          return;
        }
        temp_vals[opt_idx] = (scm_t_bits)(intptr_t)current->next->data->value;
        current = current->next->next;
        break;
      case SCM_OPTION_SCM:
        if (!current->next || !current->next->data) {
          eval_error("read-options: missing value for option: %s", name_sym->data);
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
        // Otherwise, store the pointer value
        if (is_bool(value) && value == scm_bool_false()) {
          temp_vals[opt_idx] = 0;  // SCM_BOOL_F is represented as 0
        } else {
          temp_vals[opt_idx] = (scm_t_bits)(intptr_t)value;
        }
        current = current->next->next;
        break;
    }
  }
  
  // Apply changes
  for (unsigned int i = 0; i < SCM_N_READ_OPTIONS; i++) {
    scm_read_opts[i].val = temp_vals[i];
  }
  
  // Special handling: if copy is enabled, also enable positions
  if (SCM_COPY_SOURCE_P) {
    SCM_RECORD_POSITIONS_P = 1;
  }
}

// read-options-interface: Core function for managing read options
SCM *scm_c_read_options_interface(SCM_List *args) {
  if (!args || !args->data) {
    // No arguments: return current option setting
    SCM_List *setting = get_read_option_setting();
    return setting ? wrap(setting) : scm_nil();
  }
  
  // Check if argument is a list (option setting) or something else (documentation request)
  if (is_pair(args->data)) {
    // It's a list: apply option setting
    SCM_List *old_setting = get_read_option_setting();
    change_read_option_setting(cast<SCM_List>(args->data));
    // Return new setting (not old setting, to match Guile behavior)
    SCM_List *new_setting = get_read_option_setting();
    return new_setting ? wrap(new_setting) : scm_nil();
  } else {
    // Not a list: return documented option setting (for help)
    // For now, just return current setting
    SCM_List *setting = get_read_option_setting();
    return setting ? wrap(setting) : scm_nil();
  }
}

// read-set!: Set a specific option to a value
SCM *scm_c_read_set(SCM_List *args) {
  if (!args || !args->data || !args->next || !args->next->data) {
    eval_error("read-set!: expected 2 arguments (option-name value)");
    return scm_none();
  }
  
  // Extract option name - handle both symbol and (quote symbol) forms
  SCM *name_arg = args->data;
  SCM_Symbol *name_sym = nullptr;
  
  // Check if it's a quote form: (quote symbol)
  if (is_pair(name_arg)) {
    SCM_List *name_list = cast<SCM_List>(name_arg);
    if (name_list->data && is_sym(name_list->data)) {
      SCM_Symbol *quote_sym = cast<SCM_Symbol>(name_list->data);
      if (strcmp(quote_sym->data, "quote") == 0 && name_list->next && name_list->next->data) {
        // It's (quote symbol), extract the symbol
        if (is_sym(name_list->next->data)) {
          name_sym = cast<SCM_Symbol>(name_list->next->data);
        }
      }
    }
  } else if (is_sym(name_arg)) {
    name_sym = cast<SCM_Symbol>(name_arg);
  }
  
  if (!name_sym) {
    eval_error("read-set!: first argument must be a symbol");
    return scm_none();
  }
  
  // Extract value - handle quote forms
  SCM *value = args->next->data;
  
  // If value is (quote something), extract the something
  if (is_pair(value)) {
    SCM_List *value_list = cast<SCM_List>(value);
    if (value_list->data && is_sym(value_list->data)) {
      SCM_Symbol *quote_sym = cast<SCM_Symbol>(value_list->data);
      if (strcmp(quote_sym->data, "quote") == 0 && value_list->next && value_list->next->data) {
        // It's (quote something), use the something as value
        value = value_list->next->data;
      }
    }
  }
  
  // Get current setting and append new option (like Guile 1.8's read-set! macro)
  SCM_List *current = get_read_option_setting();
  
  // Build new option: (name value)
  SCM_Symbol *name_sym_copy = make_sym(name_sym->data);
  SCM_List *new_opt_name = make_list(wrap(name_sym_copy));
  SCM_List *new_opt_val = make_list(value);
  new_opt_name->next = new_opt_val;
  
  // Append to current setting
  if (current) {
    // Find tail of current setting
    SCM_List *tail = current;
    while (tail->next) {
      tail = tail->next;
    }
    tail->next = new_opt_name;
  } else {
    current = new_opt_name;
  }
  
  // Apply the combined setting
  change_read_option_setting(current);
  
  // Return current setting
  SCM_List *setting = get_read_option_setting();
  return setting ? wrap(setting) : scm_nil();
}

// read-enable: Enable boolean options
SCM *scm_c_read_enable(SCM_List *args) {
  if (!args) {
    eval_error("read-enable: expected at least one argument");
    return scm_none();
  }
  
  // Get current setting
  SCM_List *current = get_read_option_setting();
  
  // Add enabled options to current setting
  SCM_List *new_setting = current;
  SCM_List *tail = nullptr;
  if (new_setting) {
    // Find tail
    tail = new_setting;
    while (tail->next) {
      tail = tail->next;
    }
  }
  
  // Add new options to enable
  SCM_List *enable_list = args;
  while (enable_list) {
    SCM_Symbol *enable_sym = extract_symbol(enable_list->data);
    if (!enable_sym) {
      eval_error("read-enable: expected symbol for option name");
      return scm_none();
    }
    
    // Check if option is already in setting
    bool found = false;
    SCM_List *check = current;
    while (check) {
      if (is_sym(check->data)) {
        SCM_Symbol *check_sym = cast<SCM_Symbol>(check->data);
        if (strcmp(check_sym->data, enable_sym->data) == 0) {
          found = true;
          break;
        }
      }
      check = check->next;
    }
    
    if (!found) {
      // Add to setting
      SCM_Symbol *sym_copy = make_sym(enable_sym->data);
      SCM_List *node = make_list(wrap(sym_copy));
      if (!new_setting) {
        new_setting = node;
        tail = node;
      } else {
        tail->next = node;
        tail = node;
      }
    }
    
    enable_list = enable_list->next;
  }
  
  // Apply new setting
  if (new_setting) {
    change_read_option_setting(new_setting);
  }
  
  // Return current setting
  SCM_List *setting = get_read_option_setting();
  return setting ? wrap(setting) : scm_nil();
}

// read-disable: Disable boolean options
SCM *scm_c_read_disable(SCM_List *args) {
  if (!args) {
    eval_error("read-disable: expected at least one argument");
    return scm_none();
  }
  
  // Get current setting
  SCM_List *current = get_read_option_setting();
  
  // Build new setting without disabled options
  SCM_List dummy = make_list_dummy();
  SCM_List *tail = &dummy;
  
  SCM_List *current_item = current;
  while (current_item) {
    if (is_sym(current_item->data)) {
      // Check if this option should be disabled
      bool should_disable = false;
      SCM_List *disable_list = args;
      while (disable_list) {
        SCM_Symbol *disable_sym = extract_symbol(disable_list->data);
        if (disable_sym) {
          SCM_Symbol *current_sym = cast<SCM_Symbol>(current_item->data);
          if (strcmp(current_sym->data, disable_sym->data) == 0) {
            should_disable = true;
            break;
          }
        }
        disable_list = disable_list->next;
      }
      
      if (!should_disable) {
        // Keep this option
        SCM_List *node = make_list(current_item->data);
        tail->next = node;
        tail = node;
      }
    } else {
      // Non-symbol item (value for non-boolean option), keep it
      SCM_List *node = make_list(current_item->data);
      tail->next = node;
      tail = node;
    }
    
    current_item = current_item->next;
  }
  
  // Apply new setting
  if (dummy.next) {
    change_read_option_setting(dummy.next);
  } else {
    // All options disabled, create empty setting
    change_read_option_setting(nullptr);
  }
  
  // Return current setting
  SCM_List *setting = get_read_option_setting();
  return setting ? wrap(setting) : scm_nil();
}

// Initialize read options system
void init_read_options() {
  // Register Scheme functions
  scm_define_vararg_function("read-options-interface", scm_c_read_options_interface);
  
  // Define read-set!, read-enable, and read-disable as macros (like Guile 1.8)
  // read-set! expands to: (read-options-interface (append (read-options-interface) (list 'opt val)))
  const char *read_set_macro_def = 
    "(define-macro (read-set! opt val)\n"
    "  `(read-options-interface\n"
    "     (append (read-options-interface)\n"
    "             (list ',opt ,val))))";
  
  // read-enable: append flags to current options
  // Guile implementation: (read-options-interface (append flags (read-options-interface)))
  // flags is a rest parameter, so we need to quote each flag in the macro expansion
  // Handle both (read-enable copy) and (read-enable 'copy) forms
  const char *read_enable_macro_def =
    "(define-macro (read-enable . flags)\n"
    "  `(read-options-interface\n"
    "     (append (list ,@flags)\n"
    "             (read-options-interface))))";
  
  // read-disable: remove flags from current options
  // Guile implementation: uses delq! in a loop to remove each flag
  // flags is a rest parameter, so we need to quote each flag in the macro expansion
  // Handle both (read-disable copy) and (read-disable 'copy) forms
  const char *read_disable_macro_def =
    "(define-macro (read-disable . flags)\n"
    "  `(read-options-interface\n"
    "     (let ((options (read-options-interface))\n"
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
  
  SCM *result1 = scm_c_eval_string(read_set_macro_def);
  if (!result1) {
    eval_error("Failed to define read-set! macro");
  }
  
  SCM *result2 = scm_c_eval_string(read_enable_macro_def);
  if (!result2) {
    eval_error("Failed to define read-enable macro");
  }
  
  SCM *result3 = scm_c_eval_string(read_disable_macro_def);
  if (!result3) {
    eval_error("Failed to define read-disable macro");
  }
}

