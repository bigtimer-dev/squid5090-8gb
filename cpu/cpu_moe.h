// CPU expert compute for decode (see cpu_moe.c). Shared with cuda/ (the GPU side of the exchange) and src/.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    CPU_MOE_MODE_FULL = 0,      // x -> y (both banks CPU-served)
    CPU_MOE_MODE_UP_ONLY = 1,   // x -> act (w13 on host, w2 in VRAM)
    CPU_MOE_MODE_DOWN_ONLY = 2, // act -> y (w13 in VRAM, w2 on host)
    CPU_MOE_MODE_MULTI = 3,     // spec verify: n_tok tokens in the mt_* fields, each with its own slot mask (x -> y)
};

#define CPU_MOE_MT_MAX 8

// Lives in mapped pinned host memory. Request and response fields sit on separate pages so GPU writes of the
// request never share a cache line with the CPU's response writes.
typedef struct CpuMoeShared {
    // ---- GPU -> CPU (request) ----
    uint32_t req_seq;          // written last (after __threadfence_system)
    uint32_t layer;
    uint32_t mode;
    uint32_t slot_begin;       // the CPU computes top-k slots [slot_begin, 8); the GPU did the ones before
    uint32_t k_split;          // > 0: of slot slot_begin the GPU computes intermediate rows [0, k_split), the CPU the
                               // rest (k_split % 128 == 0); both sides then sum their partial down-projections
    uint32_t slot_mask;        // != 0: the CPU computes exactly the top-k slots in this bit mask (decode expert cache:
                               // the GPU computes the others from VRAM); slot_begin / k_split are then ignored
    uint32_t pad_[2];          // keeps x 16-byte aligned (uint4 stores on the GPU)
    int32_t ids[8];
    float wts[8];
    uint16_t x[2304];          // fp16 MoE input
    // ---- CPU -> GPU (response); act is GPU -> CPU for DOWN_ONLY ----
    uint16_t act[8][1152] __attribute__((aligned(4096)));  // fp16 activations, natural layout
    float y[2304] __attribute__((aligned(64)));
    uint32_t resp_seq __attribute__((aligned(64)));        // written last
    // ---- spec verify, mode MULTI (GPU -> CPU): token t's MoE input, top-8 and CPU slot mask (0: nothing for t) ----
    uint32_t n_tok __attribute__((aligned(4096)));
    uint32_t mt_mask[CPU_MOE_MT_MAX];
    int32_t mt_ids[CPU_MOE_MT_MAX][8];
    float mt_wts[CPU_MOE_MT_MAX][8];
    uint16_t mt_x[CPU_MOE_MT_MAX][2304] __attribute__((aligned(64)));
    // ---- CPU -> GPU: token t's result (written for tokens with a non-zero mask) ----
    float mt_y[CPU_MOE_MT_MAX][2304] __attribute__((aligned(4096)));
} CpuMoeShared;

int cpu_moe_available(void);
void cpu_moe_set_layer(int layer, const uint8_t* w13_codes, const uint8_t* w13_scales, const float* es13,
                       const uint8_t* w2_codes, const uint8_t* w2_scales, const float* es2);
// One expert from its own host copy (a demoted expert of a VRAM-resident layer; see SQUID_DECODE_CACHE_DEMOTE).
void cpu_moe_set_expert(int layer, int e, const uint8_t* w13_codes, const uint8_t* w13_scales, float es13,
                        const uint8_t* w2_codes, const uint8_t* w2_scales, float es2);
int cpu_moe_start(CpuMoeShared* shared_host, int n_threads, const int* cpus);
void cpu_moe_kick(void);
void cpu_moe_stop(void);
void cpu_moe_stats(uint64_t* requests, uint64_t* busy_ns, uint64_t* max_ns);
size_t cpu_moe_shared_bytes(void);
const char* cpu_moe_kernel_name(void);
void cpu_moe_pin_current_thread(int cpu);
int cpu_moe_write_file(const char* path, const void* data, size_t bytes);
// Teacher-forced scoring helpers (SQUID_SCORE_FORCE / SQUID_SCORE_OUT in the engine).
int cpu_moe_read_tokens(const char* path, int32_t* out, int max);
void cpu_moe_score_logits(const float* logits, int vocab, int32_t forced, float rec[5]);
void cpu_moe_reference(int layer, const uint16_t* x16, const int32_t* ids, const float* wts, float* y_out);

#ifdef __cplusplus
}
#endif
