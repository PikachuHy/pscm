#include "pscm.h"
#include <stdio.h>
#include <string.h>

// Helper function to set source location
void set_source_location(SCM *scm, const char *filename, int line, int column) {
  if (!scm) return;
  if (!scm->source_loc) {
    scm->source_loc = new SCM_SourceLocation();
  }
  scm->source_loc->filename = filename;
  scm->source_loc->line = line;
  scm->source_loc->column = column;
}

// Helper function to copy source location from one node to another
void copy_source_location(SCM *dest, SCM *src) {
  if (!dest || !src || !src->source_loc) return;
  if (!dest->source_loc) {
    dest->source_loc = new SCM_SourceLocation();
  }
  dest->source_loc->filename = src->source_loc->filename;
  dest->source_loc->line = src->source_loc->line;
  dest->source_loc->column = src->source_loc->column;
}

// Helper function to recursively copy source location to all nodes in a tree
void copy_source_location_recursive(SCM *dest, SCM *src) {
  if (!dest || !src || !src->source_loc) return;
  
  // Copy to current node
  copy_source_location(dest, src);
  
  // If both are lists, recursively copy to children
  if (is_pair(dest) && is_pair(src)) {
    SCM_List *dest_list = (SCM_List *)dest->value;
    SCM_List *src_list = (SCM_List *)src->value;
    SCM_List *dest_current = dest_list;
    SCM_List *src_current = src_list;
    
    while (dest_current && src_current) {
      if (dest_current->data && src_current->data) {
        copy_source_location_recursive(dest_current->data, src_current->data);
      }
      dest_current = dest_current->next;
      src_current = src_current->next;
    }
  }
}

// Helper function to get source location string
const char *get_source_location_str(SCM *scm) {
  if (!scm || !scm->source_loc) {
    return nullptr;
  }
  static char buffer[512];
  const char *filename = scm->source_loc->filename ? scm->source_loc->filename : "<unknown>";
  snprintf(buffer, sizeof(buffer), "%s:%d:%d", 
           filename,
           scm->source_loc->line,
           scm->source_loc->column);
  return buffer;
}

