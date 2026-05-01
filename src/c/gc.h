#pragma once

#include <stddef.h>
#include <stdint.h>
#include <vector>

struct SCM;

// GC block header - sits directly before the object data in memory.
// In C++ bitfield order (LSB first on little-endian platforms):
//   bit 0:       mark (1 bit)
//   bits 1-2:    age  (2 bits)
//   bits 3-10:   type_tag (8 bits)
//   bits 11-42:  size (32 bits)  -- size of the object DATA (not including header)
//   bits 43-63:  gc_magic (21 bits) -- validation magic
struct GCBlock {
  uint64_t mark : 1;
  uint64_t age : 2;
  uint64_t type_tag : 8;
  uint64_t size : 32;
  uint64_t gc_magic : 21;
  GCBlock *next_free;
};
static_assert(sizeof(GCBlock) == 16, "GCBlock must be exactly 16 bytes");

// Type tags for GC-managed objects
enum TypeTag : uint8_t {
  GC_SCM,      // struct SCM
  GC_LIST,     // struct SCM_List
  GC_PROC,     // struct SCM_Procedure
  GC_FUNC,     // struct SCM_Function
  GC_CONT,     // struct SCM_Continuation
  GC_MODULE,   // struct SCM_Module
  GC_PORT,     // struct SCM_Port
  GC_VECTOR,   // struct SCM_Vector
  GC_HASH,     // struct SCM_HashTable
  GC_ENV,      // struct SCM_Environment
  GC_STRING,   // struct SCM_String
  GC_SYMBOL,   // struct SCM_Symbol
  GC_NUMBER,   // struct SCM_Rational
  GC_PROMISE,  // struct SCM_Promise
  GC_MACRO,    // struct SCM_Macro
  GC_VARIABLE, // struct SCM_Variable
  GC_SMOB,     // struct SCM_Smob
  GC_EVAL_FRAME, // struct EvalStackFrame
  GC_TYPE_COUNT
};

// Mark stack type
using MarkStack = std::vector<GCBlock *>;

// Trace function pointer type.
// Given a live GCBlock and a MarkStack, the function should find all SCM*
// fields inside the object and call push_if_unmarked for each point that
// points to another GC-managed block.
using TraceFn = void (*)(GCBlock *, MarkStack *);

// Array of trace functions indexed by TypeTag.
extern TraceFn trace_fns[GC_TYPE_COUNT];

// Magic constant written into every GCBlock header for conservative
// scan validation.  21 bits wide.
static const uint32_t GC_MAGIC = 0x1A3F5B;

// Size of the GCBlock header in bytes.
#define GC_HEADER_SIZE 16

// Convert a GCBlock pointer to the object data pointer right after it.
#define block_to_obj(b) ((void *)((char *)(b) + GC_HEADER_SIZE))

// Convert an object data pointer back to its GCBlock.
#define obj_to_block(p) ((GCBlock *)((char *)(p) - GC_HEADER_SIZE))

// Heap bounds globals are defined in gc.cc.
extern char *g_heap_start;
extern char *g_heap_end;

// Validate that a pointer looks like a real GCBlock inside the managed
// heap (range check, alignment check, magic check).
inline bool is_valid_block(const void *p) {
  return (const char *)p >= g_heap_start &&
         (const char *)p + sizeof(GCBlock) <= g_heap_end &&
         ((uintptr_t)p & (sizeof(GCBlock) - 1)) == 0 &&
         ((const GCBlock *)p)->gc_magic == GC_MAGIC;
}

// ---------------------------------------------------------------------------
// Root registration
// ---------------------------------------------------------------------------

struct RootRegistration {
  SCM **ptr;         // Pointer to a location that holds an SCM*
  const char *name;  // Human-readable name for debugging
};

// Maximum number of registered roots (static array to avoid heap dependency).
static const int MAX_ROOTS = 2048;

extern RootRegistration g_root_registry[MAX_ROOTS];
extern int g_num_roots;

// Register a root pointer that the GC must trace.
void gc_register_root(SCM **ptr, const char *name);

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Initialise the GC subsystem (mmap initial heap, set threshold).
void gc_init();

// Allocate SIZE bytes of zero-initialised memory tagged with TAG.
// Returns a pointer to the object data area (after the GCBlock header).
void *gc_alloc(TypeTag tag, size_t size);

// Run a full mark-sweep cycle.
void run_gc();

// Push BLOCK onto STACK if it is not yet marked (also sets the mark bit).
// Called from trace functions and root scanning.
void push_if_unmarked(GCBlock *block, MarkStack *stack);
