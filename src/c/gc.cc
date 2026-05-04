#include "pscm.h"
#include "gc.h"

#include <sys/mman.h>
#include <unistd.h>

// Portability: MAP_ANONYMOUS might not be defined on all systems (use MAP_ANON
// as fallback, which is the BSD spelling).
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// =========================================================================
// Constants
// =========================================================================

static const size_t INITIAL_HEAP_SIZE = 8 * 1024 * 1024; // 8 MB
static const size_t GROWTH_SIZE       = 4 * 1024 * 1024; // 4 MB

// Size classes for fixed-size buckets (object data size in bytes).
static const size_t SIZE_CLASSES[] = { 16, 32, 64, 128 };
static const int    NUM_FIXED_BUCKETS = 4;
static const int    LARGE_BUCKET      = 4;  // first-fit for blocks >= 129B
static const int    NUM_BUCKETS       = 5;

// Minimum GC threshold.
static const size_t MIN_THRESHOLD = 2 * 1024 * 1024; // 2 MB

// =========================================================================
// Global GC state (definitions for externs in gc.h)
// =========================================================================

char            *g_heap_start = nullptr;
char            *g_heap_end   = nullptr;
static char     *g_gc_stack_top = nullptr;  // set from main() for full-stack scan
TraceFn          trace_fns[GC_TYPE_COUNT] = { nullptr };
SweepFn          sweep_fns[GC_TYPE_COUNT] = { nullptr };
RootRegistration g_root_registry[MAX_ROOTS];
int              g_num_roots = 0;

// =========================================================================
// Heap segment management
// =========================================================================

struct HeapSegment {
  char         *start;   // mmap'd region start
  size_t        size;    // region size
  char         *bump;    // next free byte for bump allocation
  HeapSegment  *next;    // next segment in the linked list
};

// Static pool so we never need malloc inside the GC.
static const int MAX_SEGMENTS = 16;
static HeapSegment g_seg_pool[MAX_SEGMENTS];
static int         g_next_seg = 0; // next free slot in the pool

static HeapSegment *g_segments     = nullptr; // segment list head
static HeapSegment *g_current_seg  = nullptr; // segment used for bump allocation

// =========================================================================
// Free lists
// =========================================================================

static GCBlock *free_lists[NUM_BUCKETS] = { nullptr };

// =========================================================================
// GC accounting
// =========================================================================

static size_t g_allocated_since_gc = 0;  // bytes allocated since last GC
static size_t g_last_live_bytes    = 0;  // live bytes after last GC
static size_t g_gc_threshold       = 0;  // threshold to trigger next GC

// =========================================================================
// Forward declarations of trace functions
// =========================================================================

static void trace_scm(GCBlock *, MarkStack *);
static void trace_list(GCBlock *, MarkStack *);
static void trace_proc(GCBlock *, MarkStack *);
static void trace_func(GCBlock *, MarkStack *);
static void trace_cont(GCBlock *, MarkStack *);
static void trace_module(GCBlock *, MarkStack *);
static void trace_port(GCBlock *, MarkStack *);
static void trace_vector(GCBlock *, MarkStack *);
static void trace_hash(GCBlock *, MarkStack *);
static void trace_env(GCBlock *, MarkStack *);
static void trace_string(GCBlock *, MarkStack *);
static void trace_symbol(GCBlock *, MarkStack *);
static void trace_number(GCBlock *, MarkStack *);
static void trace_promise(GCBlock *, MarkStack *);
static void trace_macro(GCBlock *, MarkStack *);
static void trace_variable(GCBlock *, MarkStack *);
static void trace_smob(GCBlock *, MarkStack *);

// =========================================================================
// Helper: convert an arbitrary pointer to a GCBlock and push if valid
// =========================================================================

static void trace_ptr(void *ptr, MarkStack *stack) {
  if (!ptr) return;
  GCBlock *block = (GCBlock *)((char *)ptr - GC_HEADER_SIZE);
  if (is_valid_block(block)) {
    push_if_unmarked(block, stack);
  }
}

// =========================================================================
// push_if_unmarked
// =========================================================================

void push_if_unmarked(GCBlock *block, MarkStack *stack) {
  if (!is_valid_block(block)) return;
  if (!block->mark) {
    block->mark = 1;
    stack->push_back(block);
  }
}

// =========================================================================
// Size-to-bucket mapping
// =========================================================================

static int size_to_bucket(size_t size) {
  if (size <= 16)  return 0;
  if (size <= 32)  return 1;
  if (size <= 64)  return 2;
  if (size <= 128) return 3;
  return LARGE_BUCKET;
}

// =========================================================================
// Free <-> list helpers
// =========================================================================

static void free_to_list(GCBlock *block) {
  int b = size_to_bucket(block->size);
  block->next_free = free_lists[b];
  free_lists[b] = block;
}

// =========================================================================
// Bump allocation (within the current segment)
// =========================================================================

static void *bump_allocate(size_t total_size) {
  // Align total_size up to 16 bytes.
  if (total_size & 15) {
    total_size = (total_size + 15) & ~(size_t)15;
  }

  // Align current bump pointer up to 16 bytes.
  char *bump = g_current_seg->bump;
  if ((uintptr_t)bump & 15) {
    bump = (char *)(((uintptr_t)bump + 15) & ~(uintptr_t)15);
  }

  // If not enough room, grow the heap.
  if (bump + total_size > g_current_seg->start + g_current_seg->size) {
    size_t grow_sz = total_size > GROWTH_SIZE ? total_size : GROWTH_SIZE;
    // mmap a new segment
    long page_size = sysconf(_SC_PAGESIZE);
    grow_sz = (grow_sz + page_size - 1) & ~(size_t)(page_size - 1);

    void *mem = mmap(nullptr, grow_sz, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
      fprintf(stderr, "FATAL: gc heap mmap failed\n");
      fflush(stderr);
      abort();
    }

    if (g_next_seg >= MAX_SEGMENTS) {
      fprintf(stderr, "FATAL: gc heap segment pool exhausted (max %d)\n", MAX_SEGMENTS);
      fflush(stderr);
      abort();
    }
    HeapSegment *seg = &g_seg_pool[g_next_seg++];
    seg->start = (char *)mem;
    seg->size  = grow_sz;
    seg->bump  = seg->start;
    seg->next  = nullptr;

    // Append to segment list.
    if (!g_segments) {
      g_segments = seg;
    } else {
      HeapSegment *last = g_segments;
      while (last->next) last = last->next;
      last->next = seg;
    }
    g_current_seg = seg;

    // Update global heap bounds.
    if (seg->start < g_heap_start) g_heap_start = seg->start;
    if (seg->start + seg->size > g_heap_end) g_heap_end = seg->start + seg->size;

    bump = seg->start;
  }

  g_current_seg->bump = bump + total_size;
  return bump;
}

// =========================================================================
// gc_alloc
// =========================================================================

void *gc_alloc(TypeTag tag, size_t size) {
  size_t total_size = sizeof(GCBlock) + size;

  // Trigger GC if threshold exceeded.
  if (g_allocated_since_gc >= g_gc_threshold) {
    run_gc();
  }
  g_allocated_since_gc += total_size;

  GCBlock *block = nullptr;

  int bucket = size_to_bucket(size);
  if (bucket < NUM_FIXED_BUCKETS) {
    // Fixed-size bucket.
    size_t class_size = SIZE_CLASSES[bucket];

    // Try the free list first.
    block = free_lists[bucket];
    if (block) {
      free_lists[bucket] = block->next_free;
      block->next_free = nullptr;
    } else {
      // Bump-allocate a new block of the appropriate fixed size.
      block = (GCBlock *)bump_allocate(sizeof(GCBlock) + class_size);
    }

    // Ensure the block header size field matches the class.
    block->size = class_size;

  } else {
    // Large-object bucket (first-fit).
    GCBlock *prev = nullptr;
    block = free_lists[LARGE_BUCKET];
    while (block) {
      if (block->size >= size) {
        // Found a suitable block.
        if (prev)
          prev->next_free = block->next_free;
        else
          free_lists[LARGE_BUCKET] = block->next_free;
        block->next_free = nullptr;
        break;
      }
      prev  = block;
      block = block->next_free;
    }

    if (!block) {
      // No suitable free block -- bump-allocate fresh.
      block = (GCBlock *)bump_allocate(sizeof(GCBlock) + size);
      block->size = size;
    } else {
      // Use the block as-is with its original size.  The sequential
      // layout during sweep depends on each block's size field
      // reflecting its original allocation extent; shrinking it here
      // would create an untracked gap between this block and the next.
      // The caller receives a pointer to the start of the data area
      // and must only write up to SIZE bytes (the extra space is
      // internal fragmentation, which is acceptable for this GC).
      // block->size is NOT updated.
    }
  }

  // Initialise the header.
  block->mark     = 0;
  block->age      = 0;
  block->type_tag = tag;
  block->gc_magic = GC_MAGIC;
  block->next_free = nullptr;

  // Zero the object payload.
  char *obj_start = (char *)block_to_obj(block);
  memset(obj_start, 0, size);

  return obj_start;
}

// =========================================================================
// Root registration
// =========================================================================

void gc_set_stack_top(void *marker) {
  g_gc_stack_top = (char *)marker;
}

void gc_register_root(SCM **ptr, const char *name) {
  if (g_num_roots >= MAX_ROOTS) {
    fprintf(stderr, "FATAL: too many GC roots (max %d)\n", MAX_ROOTS);
    fflush(stderr);
    abort();
  }
  g_root_registry[g_num_roots].ptr  = ptr;
  g_root_registry[g_num_roots].name = name;
  g_num_roots++;
}

// =========================================================================
// Conservative root scanning
// =========================================================================

static void mark_if_valid_block(void *ptr, MarkStack *stack) {
  if (!ptr) return;
  char *cptr = (char *)ptr;
  if (cptr < g_heap_start || cptr >= g_heap_end) return;

  // The most common case: ptr points to the start of object data,
  // which is GC_HEADER_SIZE bytes after the block header.
  uintptr_t block_addr = (uintptr_t)cptr - GC_HEADER_SIZE;

  // Block header must be 16-byte aligned.
  if (block_addr & (sizeof(GCBlock) - 1)) return;

  GCBlock *block = (GCBlock *)block_addr;
  if (block->gc_magic == GC_MAGIC) {
    push_if_unmarked(block, stack);
  }
}

static void scan_conservative_roots(MarkStack *stack) {
  // 1. Registered roots.
  for (int i = 0; i < g_num_roots; i++) {
    if (g_root_registry[i].ptr) {
      mark_if_valid_block((void *)*g_root_registry[i].ptr, stack);
    }
  }

  // 1a. Explicitly trace the global environment's linked list.
  // g_env is an inline global in the data segment, not a GC-managed
  // block, so its entries (including user defines) must be traced here.
  SCM_Environment::List *elist = g_env.dummy.next;
  while (elist) {
    if (elist->data && elist->data->value) {
      mark_if_valid_block(elist->data->value, stack);
    }
    elist = elist->next;
  }

  // 2. Capture register values via setjmp.
  jmp_buf env;
  memset(&env, 0, sizeof(env));
  if (setjmp(env) == 0) {
    // Fall-through -- the setjmp call populates env.
  }
  // Scan the setjmp buffer word-by-word (jmp_buf may have
  // platform-dependent layout; scan bytewise but only at aligned offsets).
  for (size_t i = 0; i + sizeof(void *) <= sizeof(jmp_buf); i += sizeof(void *)) {
    void *val = *(void **)((char *)&env + i);
    mark_if_valid_block(val, stack);
  }

  // 3. Conservative scan of the C stack from the current frame
  //    up to the stack top marker set from main().  This covers ALL
  //    active frames (main → do_eval → eval → ...), not just frames
  //    below cont_base.
  char  stack_top; // address of a local variable = near the stack pointer
  char *scan_start = &stack_top;
  char *scan_end   = g_gc_stack_top;

  if (!scan_end) return; // no stack marker yet -- nothing to scan

  // Stack grows downward: scan_start < scan_end on most platforms.
  if (scan_start > scan_end) {
    char *tmp = scan_start;
    scan_start = scan_end;
    scan_end   = tmp;
  }

  // Align to pointer size.
  scan_start = (char *)((uintptr_t)scan_start & ~(sizeof(void *) - 1));
  scan_end   = (char *)((uintptr_t)scan_end   & ~(sizeof(void *) - 1));

  for (char *p = scan_start; p < scan_end; p += sizeof(void *)) {
    void *val = *(void **)p;
    mark_if_valid_block(val, stack);
  }
}

// =========================================================================
// Mark phase
// =========================================================================

static void mark_phase() {
  MarkStack stack;

  // Conservative scan of roots + C stack.
  scan_conservative_roots(&stack);

  // Trace loop: pop a block and call its trace function.
  while (!stack.empty()) {
    GCBlock *block = stack.back();
    stack.pop_back();

    TypeTag tag = (TypeTag)block->type_tag;
    if (tag < GC_TYPE_COUNT && trace_fns[tag]) {
      trace_fns[tag](block, &stack);
    }
  }
}

// =========================================================================
// Sweep phase
// =========================================================================

static void sweep_phase() {
  // 1. Clear all free lists (we rebuild them during sweep).
  for (int i = 0; i < NUM_BUCKETS; i++) {
    free_lists[i] = nullptr;
  }

  size_t total_live_bytes = 0;

  // 2. Iterate every segment in order.
  for (HeapSegment *seg = g_segments; seg; seg = seg->next) {
    char *ptr = seg->start;
    char *end = seg->bump;

    while (ptr < end) {
      GCBlock *block = (GCBlock *)ptr;
      if (block->gc_magic != GC_MAGIC) {
        // Not a valid block header -- stop scanning this segment.
        // This can happen if the segment has partially written pages
        // beyond the last allocated block.
        break;
      }

      size_t block_total = GC_HEADER_SIZE + block->size;

      if (block->mark) {
        block->mark = 0; // clear for the next cycle
        total_live_bytes += block_total;
      } else {
        // Free external resources before reclaiming the block.
        TypeTag tag = (TypeTag)block->type_tag;
        if (tag < GC_TYPE_COUNT && sweep_fns[tag]) {
          sweep_fns[tag](block);
        }
        free_to_list(block);
      }

      ptr += block_total;
    }
  }

  // 3. Update GC trigger.
  g_last_live_bytes = total_live_bytes;
  g_gc_threshold = (size_t)(total_live_bytes * 1.5);
  if (g_gc_threshold < MIN_THRESHOLD) {
    g_gc_threshold = MIN_THRESHOLD;
  }
}

// =========================================================================
// run_gc
// =========================================================================

void run_gc() {
  mark_phase();
  sweep_phase();
  g_allocated_since_gc = 0;
}

SCM *scm_gc() {
  run_gc();
  return scm_nil();
}

// =========================================================================
// Trace function implementations
// =========================================================================

// --- GC_SCM -----------------------------------------------------------
static void trace_scm(GCBlock *block, MarkStack *stack) {
  SCM *scm = (SCM *)block_to_obj(block);
  void *val = scm->value;
  if (!val) return;

  switch (scm->type) {
    // Types whose value field IS a pointer to another GC-managed object.
    case SCM::LIST:
    case SCM::PROC:
    case SCM::CONT:
    case SCM::FUNC:
    case SCM::MACRO:
    case SCM::HASH_TABLE:
    case SCM::VECTOR:
    case SCM::PORT:
    case SCM::MODULE:
    case SCM::VARIABLE:
    case SCM::PROMISE:
    case SCM::SMOB:
    case SCM::STR:
    case SCM::SYM:
    case SCM::RATIO:
      trace_ptr(val, stack);
      break;

    // Types whose value field is NOT a pointer -- skip.
    case SCM::NUM:
    case SCM::FLOAT:
    case SCM::CHAR:
    case SCM::BOOL:
    case SCM::NONE:
    case SCM::NIL:
      break;
  }
}

// --- GC_LIST ----------------------------------------------------------
static void trace_list(GCBlock *block, MarkStack *stack) {
  SCM_List *list = (SCM_List *)block_to_obj(block);
  trace_ptr(list->data, stack);
  trace_ptr(list->next, stack);
}

// --- GC_PROC ----------------------------------------------------------
static void trace_proc(GCBlock *block, MarkStack *stack) {
  SCM_Procedure *proc = (SCM_Procedure *)block_to_obj(block);
  trace_ptr(proc->name, stack);  // SCM_Symbol*
  trace_ptr(proc->args, stack);  // SCM_List*
  trace_ptr(proc->body, stack);  // SCM_List*
  trace_ptr(proc->env,  stack);  // SCM_Environment*
}

// --- GC_FUNC ----------------------------------------------------------
static void trace_func(GCBlock *block, MarkStack *stack) {
  SCM_Function *func = (SCM_Function *)block_to_obj(block);
  trace_ptr(func->name,    stack); // SCM_Symbol*
  trace_ptr(func->generic, stack); // SCM*
  // func_ptr is a C function pointer -- not traced.
}

// --- GC_CONT ----------------------------------------------------------
static void trace_cont(GCBlock *block, MarkStack *stack) {
  SCM_Continuation *cont = (SCM_Continuation *)block_to_obj(block);
  trace_ptr(cont->arg,          stack); // SCM*
  trace_ptr(cont->saved_module, stack); // SCM*
  trace_ptr(cont->wind_chain,   stack); // SCM_List*
  // stack_data / dst / stack_src are internal C data -- not traced.
}

// --- GC_MODULE --------------------------------------------------------
static void trace_module(GCBlock *block, MarkStack *stack) {
  SCM_Module *mod = (SCM_Module *)block_to_obj(block);
  trace_ptr(mod->obarray,            stack); // SCM_HashTable*
  trace_ptr(mod->uses,               stack); // SCM_List*
  trace_ptr(mod->binder,             stack); // SCM_Procedure*
  trace_ptr(mod->eval_closure,       stack); // SCM_Procedure*
  trace_ptr(mod->transformer,        stack); // SCM_Procedure*
  trace_ptr(mod->name,               stack); // SCM_List*
  trace_ptr(mod->kind,               stack); // SCM_Symbol*
  trace_ptr(mod->public_interface,   stack); // SCM_Module*
  trace_ptr(mod->exports,            stack); // SCM_List*
  trace_ptr(mod->autoload_specs,     stack); // SCM_List*
}

// --- GC_PORT ----------------------------------------------------------
static void trace_port(GCBlock *block, MarkStack *stack) {
  SCM_Port *port = (SCM_Port *)block_to_obj(block);
  // file / string_data / output_buffer are C resources -- not traced.
  trace_ptr(port->soft_procedures, stack); // SCM_Vector*
  // soft_modes is a C string -- not traced.
}

// --- GC_VECTOR --------------------------------------------------------
static void trace_vector(GCBlock *block, MarkStack *stack) {
  SCM_Vector *vec = (SCM_Vector *)block_to_obj(block);
  for (size_t i = 0; i < vec->length; i++) {
    trace_ptr(vec->elements[i], stack);
  }
}

// --- GC_HASH ----------------------------------------------------------
static void trace_hash(GCBlock *block, MarkStack *stack) {
  SCM_HashTable *ht = (SCM_HashTable *)block_to_obj(block);
  for (size_t i = 0; i < ht->capacity; i++) {
    trace_ptr(ht->buckets[i], stack);
  }
}

// --- GC_ENV -----------------------------------------------------------
static void trace_env(GCBlock *block, MarkStack *stack) {
  SCM_Environment *env = (SCM_Environment *)block_to_obj(block);

  // Walk the linked list of environment entries.
  // Entry nodes may or may not be GC-managed; we only trace the SCM*
  // value stored inside each entry.
  SCM_Environment::List *list = env->dummy.next;
  while (list) {
    if (list->data) {
      trace_ptr(list->data->value, stack);
    }
    list = list->next;
  }

  // Parent environment.
  trace_ptr(env->parent, stack);
}

// --- GC_STRING --------------------------------------------------------
static void trace_string(GCBlock *, MarkStack *) {
  // SCM_String contains char* and int -- no SCM* fields to trace.
}

// --- GC_SYMBOL --------------------------------------------------------
static void trace_symbol(GCBlock *, MarkStack *) {
  // SCM_Symbol contains char* and int -- no SCM* fields to trace.
}

// --- GC_NUMBER (SCM_Rational) -----------------------------------------
static void trace_number(GCBlock *, MarkStack *) {
  // SCM_Rational contains int64_t fields -- no SCM* fields to trace.
}

// --- GC_PROMISE -------------------------------------------------------
static void trace_promise(GCBlock *block, MarkStack *stack) {
  SCM_Promise *prom = (SCM_Promise *)block_to_obj(block);
  trace_ptr(prom->thunk, stack); // SCM*
  trace_ptr(prom->value, stack); // SCM*
}

// --- GC_MACRO ---------------------------------------------------------
static void trace_macro(GCBlock *block, MarkStack *stack) {
  SCM_Macro *macro = (SCM_Macro *)block_to_obj(block);
  trace_ptr(macro->name,             stack); // SCM_Symbol*
  trace_ptr(macro->transformer,      stack); // SCM_Procedure*
  trace_ptr(macro->env,              stack); // SCM_Environment*
  trace_ptr(macro->defining_module,  stack); // SCM_Module*
}

// --- GC_VARIABLE ------------------------------------------------------
static void trace_variable(GCBlock *block, MarkStack *stack) {
  SCM_Variable *var = (SCM_Variable *)block_to_obj(block);
  trace_ptr(var->value, stack); // SCM*
}

// =========================================================================
// Sweep function implementations — free external resources before reclaim
// =========================================================================

// --- GC_ENV -----------------------------------------------------------
static void sweep_env(GCBlock *block) {
  SCM_Environment *env = (SCM_Environment *)block_to_obj(block);
  SCM_Environment::List *list = env->dummy.next;
  while (list) {
    SCM_Environment::List *next = list->next;
    if (list->data) {
      if (list->data->key) {
        delete[] list->data->key;
      }
      delete list->data;
    }
    delete list;
    list = next;
  }
  env->dummy.next = nullptr;
}

// --- GC_SCM -----------------------------------------------------------
static void sweep_scm(GCBlock *block) {
  SCM *scm = (SCM *)block_to_obj(block);
  if (scm->source_loc) {
    if (scm->source_loc->filename) {
      free((void *)scm->source_loc->filename);
    }
    delete scm->source_loc;
    scm->source_loc = nullptr;
  }
}

// --- GC_VECTOR --------------------------------------------------------
static void sweep_vector(GCBlock *block) {
  SCM_Vector *vec = (SCM_Vector *)block_to_obj(block);
  if (vec->elements) {
    delete[] vec->elements;
    vec->elements = nullptr;
  }
}

// --- GC_HASH ----------------------------------------------------------
static void sweep_hash(GCBlock *block) {
  SCM_HashTable *ht = (SCM_HashTable *)block_to_obj(block);
  if (ht->buckets) {
    free(ht->buckets);
    ht->buckets = nullptr;
  }
}

// --- GC_STRING --------------------------------------------------------
static void sweep_string(GCBlock *block) {
  SCM_String *s = (SCM_String *)block_to_obj(block);
  if (s->data) {
    delete[] s->data;
    s->data = nullptr;
  }
}

// --- GC_SYMBOL --------------------------------------------------------
static void sweep_symbol(GCBlock *block) {
  SCM_Symbol *sym = (SCM_Symbol *)block_to_obj(block);
  if (sym->data) {
    delete[] sym->data;
    sym->data = nullptr;
  }
}

// --- GC_PORT ----------------------------------------------------------
static void sweep_port(GCBlock *block) {
  SCM_Port *port = (SCM_Port *)block_to_obj(block);
  if (port->file && port->file != stdin && port->file != stdout && port->file != stderr) {
    fclose((FILE *)port->file);
    port->file = nullptr;
  }
  if (port->string_data) {
    delete[] port->string_data;
    port->string_data = nullptr;
  }
  if (port->output_buffer) {
    free(port->output_buffer);
    port->output_buffer = nullptr;
  }
  if (port->soft_modes) {
    free(port->soft_modes);
    port->soft_modes = nullptr;
  }
}

// --- GC_CONT ----------------------------------------------------------
static void sweep_cont(GCBlock *block) {
  SCM_Continuation *cont = (SCM_Continuation *)block_to_obj(block);
  if (cont->stack_data) {
    free(cont->stack_data);
    cont->stack_data = nullptr;
  }
}

// --- GC_SMOB ----------------------------------------------------------
static void trace_smob(GCBlock *, MarkStack *) {
  // SCM_Smob contains long, void*, int64_t -- no SCM* fields to trace.
}

// =========================================================================
// gc_init
// =========================================================================

void gc_init() {
  // 1. Allocate the initial heap segment.
  long page_size = sysconf(_SC_PAGESIZE);
  size_t seg_size = (INITIAL_HEAP_SIZE + page_size - 1) & ~(size_t)(page_size - 1);

  void *mem = mmap(nullptr, seg_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mem == MAP_FAILED) {
    fprintf(stderr, "FATAL: gc_init mmap failed\n");
    fflush(stderr);
    abort();
  }

  HeapSegment *seg = &g_seg_pool[g_next_seg++];
  seg->start = (char *)mem;
  seg->size  = seg_size;
  seg->bump  = seg->start;
  seg->next  = nullptr;

  g_segments    = seg;
  g_current_seg = seg;

  g_heap_start = seg->start;
  g_heap_end   = seg->start + seg->size;

  // 2. Clear free lists (already zero-initialised statically).
  for (int i = 0; i < NUM_BUCKETS; i++) {
    free_lists[i] = nullptr;
  }

  // 3. Register trace functions.
  trace_fns[GC_SCM]       = trace_scm;
  trace_fns[GC_LIST]      = trace_list;
  trace_fns[GC_PROC]      = trace_proc;
  trace_fns[GC_FUNC]      = trace_func;
  trace_fns[GC_CONT]      = trace_cont;
  trace_fns[GC_MODULE]    = trace_module;
  trace_fns[GC_PORT]      = trace_port;
  trace_fns[GC_VECTOR]    = trace_vector;
  trace_fns[GC_HASH]      = trace_hash;
  trace_fns[GC_ENV]       = trace_env;
  trace_fns[GC_STRING]    = trace_string;
  trace_fns[GC_SYMBOL]    = trace_symbol;
  trace_fns[GC_NUMBER]    = trace_number;
  trace_fns[GC_PROMISE]   = trace_promise;
  trace_fns[GC_MACRO]     = trace_macro;
  trace_fns[GC_VARIABLE]  = trace_variable;
  trace_fns[GC_SMOB]      = trace_smob;

  // 4. Register sweep functions (free external resources).
  sweep_fns[GC_SCM]       = sweep_scm;
  sweep_fns[GC_ENV]       = sweep_env;
  sweep_fns[GC_VECTOR]    = sweep_vector;
  sweep_fns[GC_HASH]      = sweep_hash;
  sweep_fns[GC_STRING]    = sweep_string;
  sweep_fns[GC_SYMBOL]    = sweep_symbol;
  sweep_fns[GC_PORT]      = sweep_port;
  sweep_fns[GC_CONT]      = sweep_cont;

  // 5. Set initial GC threshold.
  g_gc_threshold = INITIAL_HEAP_SIZE / 2;
  g_allocated_since_gc = 0;
  g_last_live_bytes    = 0;
}
