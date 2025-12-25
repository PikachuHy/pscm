#include "pscm.h"
#include "eval.h"
#include <stdio.h>
#include <stdlib.h>


// Forward declarations for functions used by continuation.cc
SCM_List *copy_wind_chain(SCM_List *chain);
SCM_List *unwind_wind_chain(SCM_List *target);
void rewind_wind_chain(SCM_List *common, SCM_List *target);

// Helper function to copy a wind chain
SCM_List *copy_wind_chain(SCM_List *chain) {
  if (!chain) {
    return nullptr;
  }
  auto new_chain = make_list(chain->data);
  auto current = chain->next;
  auto new_current = new_chain;
  while (current) {
    new_current->next = make_list(current->data);
    new_current = new_current->next;
    current = current->next;
  }
  return new_chain;
}

// Helper function to find common prefix of two wind chains
// Compares entries by their data pointer (the (in_guard . out_guard) pair)
SCM_List *find_common_prefix(SCM_List *chain1, SCM_List *chain2) {
  if (!chain1 || !chain2) {
    return nullptr;  // If either is empty, no common prefix
  }
  
  // Compare entries by their data pointer
  SCM_List *c1 = chain1;
  SCM_List *c2 = chain2;
  int common_count = 0;
  
  while (c1 && c2 && c1->data == c2->data) {
    common_count++;
    c1 = c1->next;
    c2 = c2->next;
  }
  
  if (common_count == 0) {
    return nullptr;  // No common prefix
  }
  
  // Build the common prefix list
  SCM_List *common = nullptr;
  SCM_List **tail = &common;
  SCM_List *c = chain1;
  for (int i = 0; i < common_count; i++) {
    *tail = make_list(c->data);
    tail = &((*tail)->next);
    c = c->next;
  }
  
  return common;
}

// Unwind wind chain: call out_guard for each entry from current to target
// Returns the common prefix
SCM_List *unwind_wind_chain(SCM_List *target) {
  SCM_List *current = g_wind_chain;
  
  // Find common prefix
  SCM_List *common = find_common_prefix(current, target);
  
  // Unwind from current to common (call out_guard for each entry not in common)
  // Build a list of entries to unwind (in reverse order, so we unwind in correct order)
  SCM_List *to_unwind = nullptr;
  SCM_List *c = current;
  
  // Collect entries that need to be unwound
  while (c) {
    // Check if this entry is in common
    bool in_common = false;
    if (common) {
      SCM_List *check = common;
      while (check) {
        if (check->data == c->data) {
          in_common = true;
          break;
        }
        check = check->next;
      }
    }
    
    if (!in_common) {
      // Add to unwind list (in reverse order)
      auto new_entry = make_list(c->data);
      new_entry->next = to_unwind;
      to_unwind = new_entry;
    }
    else {
      // We've reached the common prefix, stop
      break;
    }
    
    c = c->next;
  }
  
  // Now unwind in reverse order (which is the correct order)
  while (to_unwind) {
    SCM *entry = to_unwind->data;
    if (is_pair(entry)) {
      SCM_List *pair = cast<SCM_List>(entry);
      if (pair->next && pair->next->data) {
        // Call out_guard
        SCM *out_guard = pair->next->data;
        if (is_proc(out_guard)) {
          auto proc = cast<SCM_Procedure>(out_guard);
          apply_procedure(&g_env, proc, nullptr);
        }
      }
    }
    to_unwind = to_unwind->next;
  }
  
  // Update wind chain to common prefix
  g_wind_chain = common;
  return common;
}

// Rewind wind chain: call in_guard for each entry from common to target
void rewind_wind_chain(SCM_List *common, SCM_List *target) {
  // Build list of entries to rewind (from target, excluding common)
  SCM_List *to_rewind = nullptr;
  SCM_List *target_current = target;
  
  // Count how many entries in common
  int common_count = 0;
  SCM_List *c = common;
  while (c) {
    common_count++;
    c = c->next;
  }
  
  // Skip common entries in target
  c = target;
  for (int i = 0; i < common_count && c; i++) {
    c = c->next;
  }
  
  // Collect entries to rewind (in reverse order)
  while (c) {
    auto new_entry = make_list(c->data);
    new_entry->next = to_rewind;
    to_rewind = new_entry;
    c = c->next;
  }
  
  // Rewind entries (call in_guard) and add to wind chain
  while (to_rewind) {
    SCM *entry = to_rewind->data;
    if (is_pair(entry)) {
      SCM_List *pair = cast<SCM_List>(entry);
      if (pair->data) {
        // Call in_guard
        SCM *in_guard = pair->data;
        if (is_proc(in_guard)) {
          auto proc = cast<SCM_Procedure>(in_guard);
          apply_procedure(&g_env, proc, nullptr);
        }
      }
    }
    
    // Add to wind chain
    auto new_entry = make_list(entry);
    new_entry->next = g_wind_chain;
    g_wind_chain = new_entry;
    
    to_rewind = to_rewind->next;
  }
}

// Dynamic-wind special form handler
SCM *eval_dynamic_wind(SCM_Environment *env, SCM_List *l) {
  assert(l->next && l->next->next && l->next->next->next);
  
  // Evaluate the three arguments: in_guard, thunk, out_guard
  SCM *in_guard_val = eval_with_env(env, l->next->data);
  SCM *thunk_val = eval_with_env(env, l->next->next->data);
  SCM *out_guard_val = eval_with_env(env, l->next->next->next->data);
  
  // Verify they are procedures
  if (!is_proc(in_guard_val) || !is_proc(thunk_val) || !is_proc(out_guard_val)) {
    eval_error("dynamic-wind: all arguments must be procedures");
  }
  
  // Call in_guard
  auto in_guard_proc = cast<SCM_Procedure>(in_guard_val);
  apply_procedure(env, in_guard_proc, nullptr);
  
  // Create wind entry: (in_guard . out_guard)
  auto wind_entry = make_list(in_guard_val);
  wind_entry->next = make_list(out_guard_val);
  auto wind_entry_wrapped = wrap(wind_entry);
  
  // Save old wind chain and add new entry
  SCM_List *old_wind_chain = g_wind_chain;
  auto new_wind_entry = make_list(wind_entry_wrapped);
  new_wind_entry->next = g_wind_chain;
  g_wind_chain = new_wind_entry;
  
  // Call thunk
  auto thunk_proc = cast<SCM_Procedure>(thunk_val);
  SCM *result = apply_procedure(env, thunk_proc, nullptr);
  
  // Restore old wind chain
  g_wind_chain = old_wind_chain;
  
  // Call out_guard
  auto out_guard_proc = cast<SCM_Procedure>(out_guard_val);
  apply_procedure(env, out_guard_proc, nullptr);
  
  return result;
}
