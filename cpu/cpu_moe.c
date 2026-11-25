// CPU expert compute for the small-GPU fork (decode, B = 1).
//
// Expert banks that do not fit in VRAM live in pinned host RAM. Streaming them to the GPU over PCIe (13 GB/s on a
// laptop's x8 link) dominated decode time. The CPU reads RAM ~4x faster than PCIe delivers it, so for decode the
// GPU instead hands the CPU the MoE input (2304 fp16), the top-8 expert ids and routing weights through a small
// mapped pinned buffer; CPU threads compute the experts with AVX-512 and write the [2304] f32 result back.
//
// Numerics follow the GPU decode kernels (moe_up_tc_kernel / moe_down_tc_kernel): e2m1 weights times the E4M3
// block-16 scale (exact), times the f32 per-expert scale, f32 accumulation, softcapped SwiGLU, the intermediate
// activation rounded to fp16, and the routing-weighted sum over the 8 slots in slot order. tanh/exp use libm
// instead of the GPU's approximations, so results agree to fp16-activation precision, not bit-for-bit.
//
// AVX-512 kernels: "i16" (default, needs AVX512_VNNI) multiplies the weights as exact integers (2 x e2m1 is an
// integer in [-12, 12]) with each input quantized to int16 under a per-16-element-block power-of-two scale (error up
// to 2^-15 of the block max; vpdpwssd, 32 MACs per instruction). "i16x2" (SQUID_CPU_MOE_KERNEL=i16x2) adds a second
// int16 term for the residual, hi = round(x * 2^(14-e)) and lo = round(residual * 2^15): inputs are kept to 2^-30 of
// their block max. "fp32" (=fp32) converts every weight to f32 and uses FMA; ~2x the instructions of i16, and
// compute-bound with 8 threads inside the engine.
// None of them, nor the GPU path, is exact (the GPU uses fp16 MMA inputs and fast tanh/exp), and greedy output
// drifts apart after tens to hundreds of tokens between ANY two of them: a last-bit change eventually flips a
// near-tied top-8 expert choice. Teacher-forced over 1024 tokens of natural text (tools/score.sh), all three give
// the GPU path's NLL within 1-2 standard errors (1.769 GPU; fp32 1.767, i16 1.766, i16x2 1.762 +- 0.003), and fp32
// vs i16x2 (which agree to 1e-7) differ from each other as much as either differs from the GPU. So the fastest, i16,
// is the default.
//
// Protocol (one request in flight): the GPU writes x/ids/weights/layer/mode, __threadfence_system(), then
// req_seq = s. The coordinator thread (worker 0) spins on req_seq, wakes the workers through a generation
// counter, every worker computes its share, and the last one to finish writes y, then resp_seq = s.
#define _GNU_SOURCE
#include "cpu_moe.h"

#include <cpuid.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define HIDDEN 2304
#define INTER 1152
#define TOPK 8
#define MAX_THREADS 32
#define MAX_LAYERS 24
#define GATE_CAP 4.0f
#define UP_CAP 25.0f

// ---------------------------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------------------------

// e2m1 code -> value, replicated 4x so vpermb (which uses the low 6 index bits) ignores bits 4..5.
static const int8_t k_e2m1_x2_s8[64] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};
// 2 x e2m1 as int16, replicated so vpermw (low 5 index bits) ignores bit 4.
static const int16_t k_e2m1_x2_s16[32] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};
static const float k_e2m1_f[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

static inline float e4m3_to_f32(uint8_t c) {
    // E4M3 (bias 7): place eeee.mmm at f32 bits 26..20 and rescale by 2^120; exact for normals and subnormals.
    uint32_t bits = (uint32_t)(c & 0x7f) << 20;
    float f;
    memcpy(&f, &bits, 4);
    f *= 0x1p120f;
    return (c & 0x80) ? -f : f;
}

static inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }
static inline uint16_t f32_to_f16(float f) { return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT); }

// Rearrange x[K] (K % 128 == 0) so each 128-weight chunk is [x_even(64) | x_odd(64)]: the low nibbles of a chunk's
// 64 code bytes hold the even k, the high nibbles the odd k.
static void prep_x_split(const float* x, float* xs, int K) {
    for (int c = 0; c < K / 128; ++c) {
        for (int j = 0; j < 64; ++j) {
            xs[c * 128 + j] = x[c * 128 + 2 * j];
            xs[c * 128 + 64 + j] = x[c * 128 + 2 * j + 1];
        }
    }
}

// Scalar reference: dot(row, x) for one NVFP4 row (natural layout), x unsplit.
static float row_dot_ref(const uint8_t* codes, const uint8_t* sc, const float* x, int K) {
    double acc = 0.0;
    for (int b = 0; b < K / 16; ++b) {
        float s = e4m3_to_f32(sc[b]);
        double blk = 0.0;
        for (int i = 0; i < 16; ++i) {
            int k = b * 16 + i;
            uint8_t byte = codes[k >> 1];
            int nib = (k & 1) ? (byte >> 4) : (byte & 15);
            blk += (double)k_e2m1_f[nib] * x[k];
        }
        acc += blk * s;
    }
    return (float)acc;
}

#define AVX512_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi,avx512dq,fma,f16c")))

// Two rows at once (shares the x loads and scale permute constants). xs = split layout, K % 128 == 0.
// Scales of a 128-weight chunk: 8 bytes = blocks 0..7. lo byte j (k = 2j) and hi byte j (k = 2j+1) belong to block j/8.
AVX512_TARGET static inline void rows2_dot_avx512(
    const uint8_t* c0, const uint8_t* s0, const uint8_t* c1, const uint8_t* s1,
    const float* xs, int K, float* out0, float* out1) {
    const __m512i tbl = _mm512_loadu_si512((const void*)k_e2m1_x2_s8);
    const __m512i nib_lo = _mm512_set1_epi8(0x0f);
    const __m512i sc_mask = _mm512_set1_epi32(0x7f);
    const __m512 two_m121 = _mm512_set1_ps(0x1p120f * 0.5f);  // e4m3 rescale, and undo the x2 in the s8 table
    // scale permutes: group g (16 lanes) of lo/hi covers blocks 2g (lanes 0..7) and 2g+1 (lanes 8..15)
    const __m512i sidx0 = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i sidx1 = _mm512_set_epi32(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2);
    const __m512i sidx2 = _mm512_set_epi32(5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4);
    const __m512i sidx3 = _mm512_set_epi32(7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6);
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps();
    const int nch = K / 128;
    for (int c = 0; c < nch; ++c) {
        const float* xe = xs + c * 128;
        const float* xo = xe + 64;
        // row 0
        __m512i cb0 = _mm512_loadu_si512((const void*)(c0 + 64 * c));
        __m512i lo0 = _mm512_permutexvar_epi8(_mm512_and_si512(cb0, nib_lo), tbl);
        __m512i hi0 = _mm512_permutexvar_epi8(_mm512_srli_epi16(cb0, 4), tbl);
        __m512i sc0 = _mm512_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(s0 + 8 * c)));
        __m512 sf0 = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_and_si512(sc0, sc_mask), 20)), two_m121);
        // row 1
        __m512i cb1 = _mm512_loadu_si512((const void*)(c1 + 64 * c));
        __m512i lo1 = _mm512_permutexvar_epi8(_mm512_and_si512(cb1, nib_lo), tbl);
        __m512i hi1 = _mm512_permutexvar_epi8(_mm512_srli_epi16(cb1, 4), tbl);
        __m512i sc1 = _mm512_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(s1 + 8 * c)));
        __m512 sf1 = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_and_si512(sc1, sc_mask), 20)), two_m121);
#define GROUP(g, SIDX)                                                                                    \
    {                                                                                                     \
        __m512 xg_e = _mm512_loadu_ps(xe + 16 * (g));                                                     \
        __m512 xg_o = _mm512_loadu_ps(xo + 16 * (g));                                                     \
        __m512 sv0 = _mm512_permutexvar_ps(SIDX, sf0);                                                    \
        __m512 sv1 = _mm512_permutexvar_ps(SIDX, sf1);                                                    \
        __m512 wl0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(lo0, g)));          \
        __m512 wh0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(hi0, g)));          \
        __m512 wl1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(lo1, g)));          \
        __m512 wh1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(hi1, g)));          \
        a0 = _mm512_fmadd_ps(_mm512_mul_ps(wl0, sv0), xg_e, a0);                                          \
        a1 = _mm512_fmadd_ps(_mm512_mul_ps(wh0, sv0), xg_o, a1);                                          \
        b0 = _mm512_fmadd_ps(_mm512_mul_ps(wl1, sv1), xg_e, b0);                                          \
        b1 = _mm512_fmadd_ps(_mm512_mul_ps(wh1, sv1), xg_o, b1);                                          \
    }
        GROUP(0, sidx0)
        GROUP(1, sidx1)
        GROUP(2, sidx2)
        GROUP(3, sidx3)
#undef GROUP
    }
    *out0 = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    *out1 = _mm512_reduce_add_ps(_mm512_add_ps(b0, b1));
}


// ---- int16 path -------------------------------------------------------------------------------------------------
// An input vector x[K] quantized for the int16 kernel. Groups of 64 inputs q = 0..K/64-1: q16[64q + j] = x[64q + 2j]
// (j < 32, even k) and q16[64q + 32 + j] = x[64q + 2j + 1] (odd k), each scaled by 2^(14 - e_b) where e_b =
// floor(log2(max |x| of its 16-block)). sv[q][l] = 0.5 (undo 2 x e2m1) * 2^120 (E4M3 bits trick) * 2^(e_b - 14) for
// the block of int32 lane l of group q (block 4q + l/4).
typedef struct {
    int16_t q16[HIDDEN];
    int16_t lo16[HIDDEN];  // i16x2: second term, same layout, in units of 2^-15 of q16's
    float sv[HIDDEN / 64][16];
} XQ;

// Quantize one 16-block of x (x16[0..15] = x[16b .. 16b+15]) into xq.
static inline void quant_block(XQ* xq, int b, const float* x16) {
    float m = 0.f;
    for (int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(x16[i]));
    int e = 0;
    if (m > 0.f) {
        int E;
        frexpf(m, &E);  // m = f * 2^E, f in [0.5, 1)
        e = E - 1;
    }
    const float f = ldexpf(1.0f, 14 - e);
    const int q = b >> 2;  // group of 64
    for (int i = 0; i < 16; ++i) {
        const int k = 16 * b + i;
        const float xf = x16[i] * f;  // exact (power-of-two scale), |xf| < 2^15
        long v = lrintf(xf);
        if (v > 32767) v = 32767;
        if (v < -32767) v = -32767;
        const long lo = lrintf((xf - (float)v) * 32768.0f);  // |xf - v| <= 0.5 -> |lo| <= 2^14
        const int j = (k & 63) >> 1;
        xq->q16[64 * q + ((k & 1) ? 32 : 0) + j] = (int16_t)v;
        xq->lo16[64 * q + ((k & 1) ? 32 : 0) + j] = (int16_t)lo;
    }
    const float sv = ldexpf(1.0f, e + 105);
    for (int l = 4 * (b & 3); l < 4 * (b & 3) + 4; ++l) xq->sv[q][l] = sv;
}

static void quant_x(const float* x, int K, XQ* xq) {
    for (int b = 0; b < K / 16; ++b) quant_block(xq, b, x + 16 * b);
}

// AVX-512 version of quant_block with bit-identical output (the default; SQUID_CPU_MOE_QUANT=scalar for the
// scalar one). Same f32 operations: e = floor(log2 max|x|) (vgetexp, exact also for subnormals with DAZ off, as
// frexpf), power-of-two scales built from exponent bits, vcvtps2dq rounds to nearest even like lrintf under the
// default MXCSR, the same clamp, the same exact residual. glibc lrintf/frexpf/ldexpf calls were ~13 us per visit.
static int g_quant_scalar = 0;
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))) static inline void quant_block_v(XQ* xq, int b, __m512 x) {
    const float m = _mm512_reduce_max_ps(_mm512_abs_ps(x));
    int e = 0;
    if (m > 0.f) e = (int)_mm_cvtss_f32(_mm_getexp_ss(_mm_setzero_ps(), _mm_set_ss(m)));
    uint32_t fb = (uint32_t)(127 + 14 - e) << 23, svb = (uint32_t)(127 + e + 105) << 23;
    float f, sv;
    memcpy(&f, &fb, 4);
    memcpy(&sv, &svb, 4);
    const __m512 xf = _mm512_mul_ps(x, _mm512_set1_ps(f));
    __m512i v = _mm512_cvtps_epi32(xf);
    v = _mm512_max_epi32(_mm512_min_epi32(v, _mm512_set1_epi32(32767)), _mm512_set1_epi32(-32767));
    const __m512i lo = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_sub_ps(xf, _mm512_cvtepi32_ps(v)), _mm512_set1_ps(32768.0f)));
    // even k -> q16[64q + 8(b&3) + j], odd k -> q16[64q + 32 + 8(b&3) + j]
    const __m256i deint = _mm256_setr_epi16(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    const __m256i vq = _mm256_permutexvar_epi16(deint, _mm512_cvtepi32_epi16(v));
    const __m256i lq = _mm256_permutexvar_epi16(deint, _mm512_cvtepi32_epi16(lo));
    const int q = b >> 2, o = 64 * q + 8 * (b & 3);
    _mm_storeu_si128((__m128i*)(xq->q16 + o), _mm256_castsi256_si128(vq));
    _mm_storeu_si128((__m128i*)(xq->q16 + o + 32), _mm256_extracti128_si256(vq, 1));
    _mm_storeu_si128((__m128i*)(xq->lo16 + o), _mm256_castsi256_si128(lq));
    _mm_storeu_si128((__m128i*)(xq->lo16 + o + 32), _mm256_extracti128_si256(lq, 1));
    _mm_storeu_ps(&xq->sv[q][4 * (b & 3)], _mm_set1_ps(sv));
}

// x as fp16 (the shared buffer) -> quantized input, vectorized.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,f16c"))) static void quant_x_f16(const uint16_t* xh, int K, XQ* xq) {
    for (int b = 0; b < K / 16; ++b) quant_block_v(xq, b, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(xh + 16 * b))));
}

// Software prefetch of the weight stream (SQUID_CPU_MOE_PF=<row pairs ahead>, 0 = off; SQUID_CPU_MOE_PF_HINT=
// t0|t1|t2|nta, default t0). Spread over the dot loop: 2 code lines per 128-weight step, plus the pair's scale lines
// at the start. Prefetches do not change any arithmetic.
static int g_pf_pairs = 0;
static int g_pf_hint = 0;  // 0 = T0, 1 = T1, 2 = T2, 3 = NTA
static inline void pf_line(const void* a) {
    switch (g_pf_hint) {
        case 1: _mm_prefetch((const char*)a, _MM_HINT_T1); break;
        case 2: _mm_prefetch((const char*)a, _MM_HINT_T2); break;
        case 3: _mm_prefetch((const char*)a, _MM_HINT_NTA); break;
        default: _mm_prefetch((const char*)a, _MM_HINT_T0); break;
    }
}

#define AVX512_I16_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi,avx512dq,avx512vnni,fma,f16c")))

// Two rows at once. Per 64-weight group: 32 code bytes -> zero-extended to words; vpermw maps the low / high nibble
// (even / odd k) to 2 x e2m1; two vpdpwssd give int32 lane l = sum over k in {4l, 4l+1, 4l+2, 4l+3} (block l/4);
// one f32 FMA applies the per-lane scale (E4M3 block scale from a vpermb of the 4 scale bytes, times sv).
AVX512_I16_TARGET static inline void rows2_dot_i16(
    const uint8_t* c0, const uint8_t* s0, const uint8_t* c1, const uint8_t* s1,
    const XQ* xq, int K, float* out0, float* out1, const uint8_t* pfc, const uint8_t* pfs, int q0) {
    const __m512i tbl = _mm512_loadu_si512((const void*)k_e2m1_x2_s16);
    // byte 4l of the result <- scale byte l/4 (group 2h) or 4 + l/4 (group 2h+1); other bytes zeroed
    const __m512i sidx_a = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    const __m512i sidx_b = _mm512_set_epi32(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4);
    const __mmask64 m4 = 0x1111111111111111ull;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    const int ng = K / 64;
    for (int q = q0; q < ng; q += 2) {  // q0 (even) > 0: only weights k >= 64 q0 (partial slot)
        if (pfc) {  // the pair P ahead: K code bytes (2 lines per step), K/8 scale bytes (first steps)
            pf_line(pfc + 64 * q);
            pf_line(pfc + 64 * q + 64);
            if (32 * q <= K / 8) pf_line(pfs + 32 * q);
        }
        const __m512i sb0 = _mm512_castsi128_si512(_mm_loadl_epi64((const __m128i*)(s0 + 4 * q)));
        const __m512i sb1 = _mm512_castsi128_si512(_mm_loadl_epi64((const __m128i*)(s1 + 4 * q)));
#define GROUP16(QQ, SIDX)                                                                                          \
    {                                                                                                              \
        const __m512i xe = _mm512_loadu_si512((const void*)(xq->q16 + 64 * (QQ)));                                 \
        const __m512i xo = _mm512_loadu_si512((const void*)(xq->q16 + 64 * (QQ) + 32));                            \
        const __m512 sv = _mm512_loadu_ps(xq->sv[QQ]);                                                             \
        const __m512i cw0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(c0 + 32 * (QQ))));            \
        const __m512i cw1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(c1 + 32 * (QQ))));            \
        __m512i d0 = _mm512_dpwssd_epi32(_mm512_setzero_si512(), _mm512_permutexvar_epi16(cw0, tbl), xe);         \
        __m512i d1 = _mm512_dpwssd_epi32(_mm512_setzero_si512(), _mm512_permutexvar_epi16(cw1, tbl), xe);         \
        d0 = _mm512_dpwssd_epi32(d0, _mm512_permutexvar_epi16(_mm512_srli_epi16(cw0, 4), tbl), xo);               \
        d1 = _mm512_dpwssd_epi32(d1, _mm512_permutexvar_epi16(_mm512_srli_epi16(cw1, 4), tbl), xo);               \
        const __m512 sc0 = _mm512_mul_ps(                                                                          \
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_maskz_permutexvar_epi8(m4, SIDX, sb0), 20)), sv);         \
        const __m512 sc1 = _mm512_mul_ps(                                                                          \
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_maskz_permutexvar_epi8(m4, SIDX, sb1), 20)), sv);         \
        acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d0), sc0, acc0);                                                 \
        acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d1), sc1, acc1);                                                 \
    }
        GROUP16(q, sidx_a)
        GROUP16(q + 1, sidx_b)
#undef GROUP16
    }
    *out0 = _mm512_reduce_add_ps(acc0);
    *out1 = _mm512_reduce_add_ps(acc1);
}

// i16x2: as rows2_dot_i16 plus a second vpdpwssd pass over the low input terms; per lane
// (d_hi + 2^-15 d_lo) is formed in f32 (both exact integers below 2^24) before the scale FMA.
AVX512_I16_TARGET static inline void rows2_dot_i16x2(
    const uint8_t* c0, const uint8_t* s0, const uint8_t* c1, const uint8_t* s1,
    const XQ* xq, int K, float* out0, float* out1, const uint8_t* pfc, const uint8_t* pfs, int q0) {
    const __m512i tbl = _mm512_loadu_si512((const void*)k_e2m1_x2_s16);
    const __m512i sidx_a = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    const __m512i sidx_b = _mm512_set_epi32(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4);
    const __mmask64 m4 = 0x1111111111111111ull;
    const __m512 lo_w = _mm512_set1_ps(1.0f / 32768.0f);
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    const int ng = K / 64;
    for (int q = q0; q < ng; q += 2) {  // q0 (even) > 0: only weights k >= 64 q0 (partial slot)
        if (pfc) {  // the pair P ahead: K code bytes (2 lines per step), K/8 scale bytes (first steps)
            pf_line(pfc + 64 * q);
            pf_line(pfc + 64 * q + 64);
            if (32 * q <= K / 8) pf_line(pfs + 32 * q);
        }
        const __m512i sb0 = _mm512_castsi128_si512(_mm_loadl_epi64((const __m128i*)(s0 + 4 * q)));
        const __m512i sb1 = _mm512_castsi128_si512(_mm_loadl_epi64((const __m128i*)(s1 + 4 * q)));
#define GROUP16X2(QQ, SIDX)                                                                                        \
    {                                                                                                              \
        const __m512i xe = _mm512_loadu_si512((const void*)(xq->q16 + 64 * (QQ)));                                 \
        const __m512i xo = _mm512_loadu_si512((const void*)(xq->q16 + 64 * (QQ) + 32));                            \
        const __m512i le = _mm512_loadu_si512((const void*)(xq->lo16 + 64 * (QQ)));                                \
        const __m512i lo = _mm512_loadu_si512((const void*)(xq->lo16 + 64 * (QQ) + 32));                           \
        const __m512 sv = _mm512_loadu_ps(xq->sv[QQ]);                                                             \
        const __m512i cw0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(c0 + 32 * (QQ))));            \
        const __m512i cw1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(c1 + 32 * (QQ))));            \
        const __m512i we0 = _mm512_permutexvar_epi16(cw0, tbl), wo0 = _mm512_permutexvar_epi16(_mm512_srli_epi16(cw0, 4), tbl); \
        const __m512i we1 = _mm512_permutexvar_epi16(cw1, tbl), wo1 = _mm512_permutexvar_epi16(_mm512_srli_epi16(cw1, 4), tbl); \
        const __m512i h0 = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(_mm512_setzero_si512(), we0, xe), wo0, xo);     \
        const __m512i h1 = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(_mm512_setzero_si512(), we1, xe), wo1, xo);     \
        const __m512i l0 = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(_mm512_setzero_si512(), we0, le), wo0, lo);     \
        const __m512i l1 = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(_mm512_setzero_si512(), we1, le), wo1, lo);     \
        const __m512 sc0 = _mm512_mul_ps(                                                                          \
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_maskz_permutexvar_epi8(m4, SIDX, sb0), 20)), sv);         \
        const __m512 sc1 = _mm512_mul_ps(                                                                          \
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_maskz_permutexvar_epi8(m4, SIDX, sb1), 20)), sv);         \
        const __m512 v0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(l0), lo_w, _mm512_cvtepi32_ps(h0));                   \
        const __m512 v1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(l1), lo_w, _mm512_cvtepi32_ps(h1));                   \
        acc0 = _mm512_fmadd_ps(v0, sc0, acc0);                                                                     \
        acc1 = _mm512_fmadd_ps(v1, sc1, acc1);                                                                     \
    }
        GROUP16X2(q, sidx_a)
        GROUP16X2(q + 1, sidx_b)
#undef GROUP16X2
    }
    *out0 = _mm512_reduce_add_ps(acc0);
    *out1 = _mm512_reduce_add_ps(acc1);
}

// Spec verify (mode MULTI): rows2_dot_i16 for M inputs at once (the tokens of a verify pass that route to the same
// expert). Per 64-weight group the codes are loaded, widened and mapped through the e2m1 table once and the E4M3 block
// scales expanded once; each input adds its two vpdpwssd and one scale FMA, in rows2_dot_i16's per-lane order, so
// every output is bit-identical to a rows2_dot_i16 call with that input. M is a compile-time constant per call site.
#define MT_MAX CPU_MOE_MT_MAX
AVX512_I16_TARGET __attribute__((always_inline)) static inline void rows2_dot_i16_m(
    const uint8_t* c0, const uint8_t* s0, const uint8_t* c1, const uint8_t* s1, const XQ* const* xq, const int M,
    int K, float* out0, float* out1) {
    const __m512i tbl = _mm512_loadu_si512((const void*)k_e2m1_x2_s16);
    const __m512i sidx_a = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    const __m512i sidx_b = _mm512_set_epi32(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4);
    const __mmask64 m4 = 0x1111111111111111ull;
    __m512 acc0[MT_MAX], acc1[MT_MAX];
    for (int j = 0; j < M; ++j) acc0[j] = acc1[j] = _mm512_setzero_ps();
    const int ng = K / 64;
    for (int q = 0; q < ng; q += 2) {
        const __m512i sb0 = _mm512_castsi128_si512(_mm_loadl_epi64((const __m128i*)(s0 + 4 * q)));
        const __m512i sb1 = _mm512_castsi128_si512(_mm_loadl_epi64((const __m128i*)(s1 + 4 * q)));
#define GROUPM(QQ, SIDX)                                                                                           \
    {                                                                                                              \
        const __m512i cw0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(c0 + 32 * (QQ))));            \
        const __m512i cw1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(c1 + 32 * (QQ))));            \
        const __m512i we0 = _mm512_permutexvar_epi16(cw0, tbl), wo0 = _mm512_permutexvar_epi16(_mm512_srli_epi16(cw0, 4), tbl); \
        const __m512i we1 = _mm512_permutexvar_epi16(cw1, tbl), wo1 = _mm512_permutexvar_epi16(_mm512_srli_epi16(cw1, 4), tbl); \
        const __m512 e0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_maskz_permutexvar_epi8(m4, SIDX, sb0), 20)); \
        const __m512 e1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_maskz_permutexvar_epi8(m4, SIDX, sb1), 20)); \
        for (int j = 0; j < M; ++j) {                                                                              \
            const __m512i xe = _mm512_loadu_si512((const void*)(xq[j]->q16 + 64 * (QQ)));                          \
            const __m512i xo = _mm512_loadu_si512((const void*)(xq[j]->q16 + 64 * (QQ) + 32));                     \
            const __m512 sv = _mm512_loadu_ps(xq[j]->sv[QQ]);                                                      \
            const __m512i d0 = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(_mm512_setzero_si512(), we0, xe), wo0, xo); \
            const __m512i d1 = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(_mm512_setzero_si512(), we1, xe), wo1, xo); \
            acc0[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d0), _mm512_mul_ps(e0, sv), acc0[j]);                     \
            acc1[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d1), _mm512_mul_ps(e1, sv), acc1[j]);                     \
        }                                                                                                          \
    }
        GROUPM(q, sidx_a)
        GROUPM(q + 1, sidx_b)
#undef GROUPM
    }
    for (int j = 0; j < M; ++j) {
        out0[j] = _mm512_reduce_add_ps(acc0[j]);
        out1[j] = _mm512_reduce_add_ps(acc1[j]);
    }
}

AVX512_I16_TARGET static void dot_m(const uint8_t* c0, const uint8_t* s0, const uint8_t* c1, const uint8_t* s1,
                                    const XQ* const* xq, int M, int K, float* out0, float* out1) {
    switch (M) {
        case 1: rows2_dot_i16_m(c0, s0, c1, s1, xq, 1, K, out0, out1); break;
        case 2: rows2_dot_i16_m(c0, s0, c1, s1, xq, 2, K, out0, out1); break;
        case 3: rows2_dot_i16_m(c0, s0, c1, s1, xq, 3, K, out0, out1); break;
        case 4: rows2_dot_i16_m(c0, s0, c1, s1, xq, 4, K, out0, out1); break;
        case 5: rows2_dot_i16_m(c0, s0, c1, s1, xq, 5, K, out0, out1); break;
        case 6: rows2_dot_i16_m(c0, s0, c1, s1, xq, 6, K, out0, out1); break;
        case 7: rows2_dot_i16_m(c0, s0, c1, s1, xq, 7, K, out0, out1); break;
        default: rows2_dot_i16_m(c0, s0, c1, s1, xq, 8, K, out0, out1); break;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Visit compute (shared by the threaded runtime, the self-test and the standalone bench)
// ---------------------------------------------------------------------------------------------------------------

typedef struct {
    const uint8_t* w13_codes;  // [E][2304][1152] host pointers, NULL if that bank is not CPU-served
    const uint8_t* w13_scales; // [E][2304][144]
    const uint8_t* w2_codes;   // [E][2304][576]
    const uint8_t* w2_scales;  // [E][2304][72]
    // xp[e] = expert e's first row in each of the four arrays (w13 codes, w13 scales, w2 codes, w2 scales): the bank
    // layout above (cpu_moe_set_layer), or a single expert's own copy (cpu_moe_set_expert: a demoted expert of a
    // VRAM-resident layer). The visit kernels address weights only through xp.
    const uint8_t* xp[128][4];
    float es13[128];
    float es2[128];
} LayerBanks;

typedef struct {
    // per-request scratch (written by the coordinator / phase 1, read by all)
    float xs[HIDDEN];         // fp32 kernel: MoE input, split layout
    float acts[TOPK][INTER];  // fp32 kernel: fp16-rounded activations, split layout per slot
    XQ xq;                    // i16 kernel: MoE input
    XQ aq[TOPK];              // i16 kernel: activations per slot (first INTER entries used)
    int ids[TOPK];
    float wts[TOPK];
    int layer;
    int mode;
    int k0;                   // first slot computed here (slots < k0 are computed by the GPU)
    int ks;                   // > 0: of slot k0 only intermediate rows [ks, INTER) (the GPU does [0, ks))
    int nsl;                  // slots computed here, ascending: sl[0..nsl) (= k0..TOPK-1 unless a slot mask is given)
    int sl[TOPK];
} VisitState;

// Slot list of a visit: the slots in `mask` (ascending) or, with mask == 0, k0..TOPK-1.
static void set_slots(VisitState* vs, uint32_t mask) {
    vs->nsl = 0;
    for (int k = 0; k < TOPK; ++k)
        if (mask ? ((mask >> k) & 1u) : (k >= vs->k0)) vs->sl[vs->nsl++] = k;
}

enum { KERNEL_SCALAR = 0, KERNEL_FP32 = 1, KERNEL_I16 = 2, KERNEL_I16X2 = 3 };
#define KERNEL_IS_INT(k) ((k) == KERNEL_I16 || (k) == KERNEL_I16X2)
static int g_have_avx512 = -1;  // CPU supports the fp32 kernel
static int g_have_vnni = 0;     // CPU supports the i16 kernel
static int g_kernel = KERNEL_SCALAR;
static int g_have_mwaitx = 0;   // AMD user-mode MONITORX/MWAITX
static int g_use_mwaitx = 0;    // SQUID_CPU_MOE_WAIT=mwaitx: idle-wait on the job cache line instead of pause-spinning

static inline float softcap(float v, float cap) { return tanhf(v / cap) * cap; }
static inline float sigmoidf_(float v) { return 1.0f / (1.0f + expf(-v)); }

// Store act for pair index i (0..1151) of slot k into the split layout (fp32 kernel).
static inline void store_act(VisitState* vs, int k, int i, float a) {
    int c = i >> 7, j = i & 127;
    int pos = (j & 1) ? (c * 128 + 64 + (j >> 1)) : (c * 128 + (j >> 1));
    vs->acts[k][pos] = a;
}

static void unsplit(const float* xs, float* x, int K) {
    for (int c = 0; c < K / 128; ++c)
        for (int j = 0; j < 64; ++j) {
            x[c * 128 + 2 * j] = xs[c * 128 + j];
            x[c * 128 + 2 * j + 1] = xs[c * 128 + 64 + j];
        }
}

// Prepare the MoE input for the active kernel.
static void prep_input(VisitState* vs, const float* x) {
    if (KERNEL_IS_INT(g_kernel)) quant_x(x, HIDDEN, &vs->xq);
    else prep_x_split(x, vs->xs, HIDDEN);
}

// Phase 1 for 16-pair blocks [b0, b1) of the flattened (slot, pair) space (8 * 1152 pairs = 576 blocks).
static void phase_up(const LayerBanks* lb, VisitState* vs, int b0, int b1) {
    const size_t row_cb = HIDDEN / 2, row_sb = HIDDEN / 16;
    float xref[HIDDEN];
    if (g_kernel == KERNEL_SCALAR) unsplit(vs->xs, xref, HIDDEN);
    for (int blk = b0; blk < b1; ++blk) {
        float a16[16];
        const int k = vs->sl[(16 * blk) / INTER], i0 = (16 * blk) % INTER;
        const int e = vs->ids[k];
        const float es = lb->es13[e];
        for (int t = 0; t < 16; ++t) {
            const int i = i0 + t;
            const size_t r0 = 2 * (size_t)i;
            const uint8_t* c0 = lb->xp[e][0] + r0 * row_cb;
            const uint8_t* s0 = lb->xp[e][1] + r0 * row_sb;
            float g, u;
            const uint8_t *pfc = NULL, *pfs = NULL;
            if (g_pf_pairs && i + g_pf_pairs < INTER && 16 * blk + t + g_pf_pairs < 16 * b1) {
                pfc = c0 + (size_t)g_pf_pairs * 2 * row_cb;
                pfs = s0 + (size_t)g_pf_pairs * 2 * row_sb;
            }
            if (g_kernel == KERNEL_I16X2) {
                rows2_dot_i16x2(c0, s0, c0 + row_cb, s0 + row_sb, &vs->xq, HIDDEN, &g, &u, pfc, pfs, 0);
            } else if (g_kernel == KERNEL_I16) {
                rows2_dot_i16(c0, s0, c0 + row_cb, s0 + row_sb, &vs->xq, HIDDEN, &g, &u, pfc, pfs, 0);
            } else if (g_kernel == KERNEL_FP32) {
                rows2_dot_avx512(c0, s0, c0 + row_cb, s0 + row_sb, vs->xs, HIDDEN, &g, &u);
            } else {
                g = row_dot_ref(c0, s0, xref, HIDDEN);
                u = row_dot_ref(c0 + row_cb, s0 + row_sb, xref, HIDDEN);
            }
            g *= es;
            u *= es;
            const float a = (softcap(g, GATE_CAP) * sigmoidf_(g)) * softcap(u, UP_CAP);
            a16[t] = f16_to_f32(f32_to_f16(a));  // the GPU stores the activation as fp16
        }
        if (KERNEL_IS_INT(g_kernel)) {
            if (g_quant_scalar) quant_block(&vs->aq[k], i0 / 16, a16);
            else quant_block_v(&vs->aq[k], i0 / 16, _mm512_loadu_ps(a16));
        }
        else for (int t = 0; t < 16; ++t) store_act(vs, k, i0 + t, a16[t]);
    }
}

// Phase 2 for output rows [r0, r1) (r0 even): y[r] = sum_k wts[k] * es2[e_k] * dot(w2[e_k][r], act_k).
static void phase_down(const LayerBanks* lb, const VisitState* vs, int r0, int r1, float* y) {
    const size_t row_cb = INTER / 2, row_sb = INTER / 16;
    for (int r = r0; r < r1; r += 2) {
        const int two = (r + 1 < r1);
        float acc0 = 0.f, acc1 = 0.f;
        for (int j = 0; j < vs->nsl; ++j) {
            const int k = vs->sl[j];
            const int e = vs->ids[k];
            const size_t row = (size_t)r;
            const uint8_t* c0 = lb->xp[e][2] + row * row_cb;
            const uint8_t* s0 = lb->xp[e][3] + row * row_sb;
            const uint8_t* c1 = two ? c0 + row_cb : c0;
            const uint8_t* s1 = two ? s0 + row_sb : s0;
            float d0, d1;
            const uint8_t *pfc = NULL, *pfs = NULL;
            if (g_pf_pairs && r + 2 * g_pf_pairs + 1 < r1) {
                pfc = c0 + (size_t)g_pf_pairs * 2 * row_cb;
                pfs = s0 + (size_t)g_pf_pairs * 2 * row_sb;
            }
            if (g_kernel == KERNEL_I16X2) {
                rows2_dot_i16x2(c0, s0, c1, s1, &vs->aq[k], INTER, &d0, &d1, pfc, pfs, k == vs->k0 ? vs->ks / 64 : 0);
            } else if (g_kernel == KERNEL_I16) {
                rows2_dot_i16(c0, s0, c1, s1, &vs->aq[k], INTER, &d0, &d1, pfc, pfs, k == vs->k0 ? vs->ks / 64 : 0);
            } else if (g_kernel == KERNEL_FP32) {
                rows2_dot_avx512(c0, s0, c1, s1, vs->acts[k], INTER, &d0, &d1);
            } else {
                float x[INTER];
                unsplit(vs->acts[k], x, INTER);
                d0 = row_dot_ref(c0, s0, x, INTER);
                d1 = row_dot_ref(c1, s1, x, INTER);
            }
            const float es = lb->es2[e];
            acc0 = fmaf(vs->wts[k], d0 * es, acc0);
            acc1 = fmaf(vs->wts[k], d1 * es, acc1);
        }
        y[r] = acc0;
        if (two) y[r + 1] = acc1;
    }
}

// Same arithmetic as phase_down (per row: acc = 0, then acc = fmaf(wts[k], dot * es, acc) for k ascending), but with
// the slot loop outside: each thread streams one expert's rows [r0, r1) at a time (2 sequential streams instead of
// 16 interleaved ones). Bit-identical to phase_down. SQUID_CPU_MOE_DOWN=kouter.
static int g_down_kouter = 0;
static void phase_down_kouter(const LayerBanks* lb, const VisitState* vs, int r0, int r1, float* y) {
    const size_t row_cb = INTER / 2, row_sb = INTER / 16;
    float acc[HIDDEN];
    for (int r = r0; r < r1; ++r) acc[r - r0] = 0.f;
    for (int j = 0; j < vs->nsl; ++j) {
        const int k = vs->sl[j];
        const int e = vs->ids[k];
        const float es = lb->es2[e], wk = vs->wts[k];
        for (int r = r0; r < r1; r += 2) {
            const int two = (r + 1 < r1);
            const size_t row = (size_t)r;
            const uint8_t* c0 = lb->xp[e][2] + row * row_cb;
            const uint8_t* s0 = lb->xp[e][3] + row * row_sb;
            const uint8_t* c1 = two ? c0 + row_cb : c0;
            const uint8_t* s1 = two ? s0 + row_sb : s0;
            float d0, d1;
            const uint8_t *pfc = NULL, *pfs = NULL;
            if (g_pf_pairs && r + 2 * g_pf_pairs + 1 < r1) {
                pfc = c0 + (size_t)g_pf_pairs * 2 * row_cb;
                pfs = s0 + (size_t)g_pf_pairs * 2 * row_sb;
            }
            if (g_kernel == KERNEL_I16X2) {
                rows2_dot_i16x2(c0, s0, c1, s1, &vs->aq[k], INTER, &d0, &d1, pfc, pfs, k == vs->k0 ? vs->ks / 64 : 0);
            } else if (g_kernel == KERNEL_I16) {
                rows2_dot_i16(c0, s0, c1, s1, &vs->aq[k], INTER, &d0, &d1, pfc, pfs, k == vs->k0 ? vs->ks / 64 : 0);
            } else if (g_kernel == KERNEL_FP32) {
                rows2_dot_avx512(c0, s0, c1, s1, vs->acts[k], INTER, &d0, &d1);
            } else {
                float x[INTER];
                unsplit(vs->acts[k], x, INTER);
                d0 = row_dot_ref(c0, s0, x, INTER);
                d1 = row_dot_ref(c1, s1, x, INTER);
            }
            acc[r - r0] = fmaf(wk, d0 * es, acc[r - r0]);
            if (two) acc[r + 1 - r0] = fmaf(wk, d1 * es, acc[r + 1 - r0]);
        }
    }
    for (int r = r0; r < r1; ++r) y[r] = acc[r - r0];
}

// Spec verify (mode MULTI, i16 kernel): the CPU slots of n tokens in one visit. Every distinct expert among them is
// read once: phase 1 items are (distinct expert, 16-pair block), each computing that block for all its users (token,
// slot); phase 2 items are output row pairs, each streaming every distinct expert's rows once and summing per token
// in slot order. Per token the arithmetic is the single-token path's (phase_up + phase_down[_kouter]) exactly.
typedef struct {
    int n;                         // tokens in the request
    int nd;                        // distinct experts
    int de[MT_MAX * TOPK];         // expert of distinct entry d
    int nu[MT_MAX * TOPK];         // its users (at most one per token: a token's top-8 are distinct experts)
    uint8_t ut[MT_MAX * TOPK][MT_MAX], uk[MT_MAX * TOPK][MT_MAX];  // user u of d: token, top-k slot
    int nsl[MT_MAX], sl[MT_MAX][TOPK];                             // per token: CPU slots, ascending
    int ids[MT_MAX][TOPK];
    float wts[MT_MAX][TOPK];
    XQ xq[MT_MAX];                 // MoE input per token
    XQ aq[MT_MAX][TOPK];           // activations per (token, slot)
    float* y[MT_MAX];              // result per token
} MultiState;
static MultiState g_mt;
static int g_mt_on = 0;            // the current request is a MULTI one on the shared-expert path
static int g_mt_seq = 0;           // SQUID_CPU_MOE_MULTI=seq: MULTI requests run token by token (no expert sharing)

AVX512_I16_TARGET static void phase_up_multi(const LayerBanks* lb, int b0, int b1) {
    const size_t row_cb = HIDDEN / 2, row_sb = HIDDEN / 16;
    const int nb = INTER / 16;
    for (int blk = b0; blk < b1; ++blk) {
        const int d = blk / nb, i0 = (blk % nb) * 16;
        const int e = g_mt.de[d], M = g_mt.nu[d];
        const float es = lb->es13[e];
        const XQ* xs[MT_MAX];
        for (int u = 0; u < M; ++u) xs[u] = &g_mt.xq[g_mt.ut[d][u]];
        float a16[MT_MAX][16];
        for (int t = 0; t < 16; ++t) {
            const size_t r0 = 2 * (size_t)(i0 + t);
            const uint8_t* c0 = lb->xp[e][0] + r0 * row_cb;
            const uint8_t* s0 = lb->xp[e][1] + r0 * row_sb;
            float g[MT_MAX], v[MT_MAX];
            dot_m(c0, s0, c0 + row_cb, s0 + row_sb, xs, M, HIDDEN, g, v);
            for (int u = 0; u < M; ++u) {
                const float gg = g[u] * es, uu = v[u] * es;
                const float a = (softcap(gg, GATE_CAP) * sigmoidf_(gg)) * softcap(uu, UP_CAP);
                a16[u][t] = f16_to_f32(f32_to_f16(a));
            }
        }
        for (int u = 0; u < M; ++u) {
            XQ* aq = &g_mt.aq[g_mt.ut[d][u]][g_mt.uk[d][u]];
            if (g_quant_scalar) quant_block(aq, i0 / 16, a16[u]);
            else quant_block_v(aq, i0 / 16, _mm512_loadu_ps(a16[u]));
        }
    }
}

AVX512_I16_TARGET static void phase_down_multi(const LayerBanks* lb, int r0, int r1) {
    const size_t row_cb = INTER / 2, row_sb = INTER / 16;
    enum { RC = 32 };
    float dbuf[MT_MAX][TOPK][RC];
    for (int rc = r0; rc < r1; rc += RC) {
        const int re = rc + RC < r1 ? rc + RC : r1;  // r0, r1 even: every step is a full row pair
        for (int d = 0; d < g_mt.nd; ++d) {
            const int e = g_mt.de[d], M = g_mt.nu[d];
            const XQ* xs[MT_MAX];
            for (int u = 0; u < M; ++u) xs[u] = &g_mt.aq[g_mt.ut[d][u]][g_mt.uk[d][u]];
            for (int r = rc; r < re; r += 2) {
                const uint8_t* c0 = lb->xp[e][2] + (size_t)r * row_cb;
                const uint8_t* s0 = lb->xp[e][3] + (size_t)r * row_sb;
                float d0[MT_MAX], d1[MT_MAX];
                dot_m(c0, s0, c0 + row_cb, s0 + row_sb, xs, M, INTER, d0, d1);
                for (int u = 0; u < M; ++u) {
                    dbuf[g_mt.ut[d][u]][g_mt.uk[d][u]][r - rc] = d0[u];
                    dbuf[g_mt.ut[d][u]][g_mt.uk[d][u]][r + 1 - rc] = d1[u];
                }
            }
        }
        for (int t = 0; t < g_mt.n; ++t) {
            if (!g_mt.nsl[t]) continue;
            for (int r = rc; r < re; ++r) {
                float acc = 0.f;
                for (int j = 0; j < g_mt.nsl[t]; ++j) {
                    const int k = g_mt.sl[t][j];
                    acc = fmaf(g_mt.wts[t][k], dbuf[t][k][r - rc] * lb->es2[g_mt.ids[t][k]], acc);
                }
                g_mt.y[t][r] = acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Threaded runtime
// ---------------------------------------------------------------------------------------------------------------
//
// Thread 0 (the coordinator) polls req_seq; threads 1..n-1 wait for a job generation. While a decode is running
// (a request in the last CPU_MOE_HOT_MS, or a cpu_moe_kick()) everything spins, so a request costs no wake-up
// latency. Otherwise workers sleep on a futex and the coordinator polls every 1 ms (10 ms after 10 s idle), so an
// idle server costs almost no CPU.

#include <limits.h>
#include <linux/futex.h>
#include <sys/syscall.h>

#define CPU_MOE_HOT_NS (200ull * 1000 * 1000)

typedef struct {
    pthread_t th;
    int idx;
    int cpu;
} Worker;

static struct {
    CpuMoeShared* sh;              // mapped pinned host buffer (host view)
    LayerBanks layers[MAX_LAYERS];
    VisitState vs;
    int n_threads;
    Worker workers[MAX_THREADS];
    _Atomic uint32_t job_gen;      // bumped by the coordinator per phase (futex word for sleeping workers)
    _Atomic uint32_t phase_done;   // threads finished in the current phase
    _Atomic uint32_t sleepers;     // workers inside futex_wait
    _Atomic uint32_t kick;         // futex word for the sleeping coordinator
    _Atomic uint32_t coord_sleeping;
    _Atomic uint64_t hot_until;    // CLOCK_MONOTONIC ns; spin until then
    _Atomic int stop;
    int phase;                     // 1 = up, 2 = down; published by the job_gen release
    uint64_t requests;
    uint64_t multi_requests;           // of them mode MULTI (spec verify)
    uint64_t busy_ns;
    uint64_t max_ns;
    uint64_t prep_ns, up_ns, down_ns;  // per-phase totals (input prep, phase 1, phase 2)
    double slots_done;                 // slots computed (K-split part counted as a fraction), for the GB/s stats
    int thr_stats;                     // SQUID_CPU_MOE_STATS=1: per-thread start / finish times
    uint64_t phase_t0;
    uint64_t t_start[MAX_THREADS], t_end[MAX_THREADS];
    uint64_t st_wake[2], st_spread[2], st_lag[2];  // per phase: last start - phase start, max - min end, max - mean end
    uint64_t st_dur[2][MAX_THREADS];               // per phase, per thread: total time in its share
    int started;
} g;

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void futex_wait_u32(_Atomic uint32_t* addr, uint32_t expected, uint64_t timeout_ns) {
    struct timespec ts = {(time_t)(timeout_ns / 1000000000ull), (long)(timeout_ns % 1000000000ull)};
    syscall(SYS_futex, (uint32_t*)addr, FUTEX_WAIT_PRIVATE, expected, &ts, NULL, 0);
}

static void futex_wake_u32(_Atomic uint32_t* addr, int n) { syscall(SYS_futex, (uint32_t*)addr, FUTEX_WAKE_PRIVATE, n, NULL, NULL, 0); }

// Wait (briefly) for *addr to change from `seen`: MWAITX on the cache line (wakes on any write, including the
// GPU's PCIe write; 25 us timeout) or a PAUSE.
__attribute__((target("mwaitx"))) static inline void wait_change(const volatile void* addr, uint32_t seen) {
    if (g_use_mwaitx) {
        _mm_monitorx((void*)addr, 0, 0);
        if (*(const volatile uint32_t*)addr == seen) _mm_mwaitx(2, 0xf0, 100000);
    } else {
        _mm_pause();
    }
}

// DRAM keep-alive (SQUID_CPU_MOE_KEEPALIVE=<GB/s>, 0 = off). The SMU of these AMD APUs picks the data-fabric clock
// (FCLK 500/1400/1960 MHz) from recent memory traffic. Between CPU visits the GPU runs its VRAM-resident layers for up
// to ~3.5 ms with no host traffic, FCLK drops to 1400 MHz and the next visits stream ~20% slower (measured: 880 vs
// 721 us/visit in an emulated decode). While decoding, the coordinator therefore reads a private 64 MB buffer at a
// capped rate between requests: ~10 GB/s holds FCLK at 1960 MHz (6 GB/s holds it ~80% of the time).
#define KA_CHUNK 1024
#define KA_BURST (64u << 10)
static uint8_t* g_ka_buf;
static size_t g_ka_bytes, g_ka_pos;
static double g_ka_bpns;          // bytes per ns (= GB/s)
static uint64_t g_ka_t0, g_ka_done, g_ka_total;
static volatile uint64_t g_ka_sink;

static void keepalive_init(void) {
    const char* v = getenv("SQUID_CPU_MOE_KEEPALIVE");
    g_ka_bpns = v ? atof(v) : 0.0;
    if (!(g_ka_bpns > 0)) { g_ka_bpns = 0; return; }
    g_ka_bytes = 64u << 20;
    g_ka_buf = (uint8_t*)aligned_alloc(4096, g_ka_bytes);
    if (!g_ka_buf) { g_ka_bpns = 0; return; }
    memset(g_ka_buf, 1, g_ka_bytes);
}

static inline void keepalive_reset(uint64_t t) { g_ka_t0 = t; g_ka_done = 0; }

// One step of the keep-alive stream: read a 1 KB chunk (one load per 64 B line) if under the rate cap.
static inline void keepalive_step(void) {
    const double allowed = (double)(now_ns() - g_ka_t0) * g_ka_bpns;
    if ((double)g_ka_done + KA_BURST < allowed) g_ka_done = (uint64_t)allowed - KA_BURST;  // no catch-up bursts
    if ((double)g_ka_done >= allowed) { _mm_pause(); return; }
    const uint64_t* p = (const uint64_t*)(g_ka_buf + g_ka_pos);
    uint64_t acc = 0;
    for (int i = 0; i < KA_CHUNK / 8; i += 8) acc += p[i];
    g_ka_sink += acc;
    g_ka_pos += KA_CHUNK;
    if (g_ka_pos >= g_ka_bytes) g_ka_pos = 0;
    g_ka_done += KA_CHUNK;
    g_ka_total += KA_CHUNK;
}

// Work stealing (default 1,16; SQUID_CPU_MOE_STEAL=<up chunk, 16-pair blocks>[,<down chunk, output rows (even)>];
// 0 = the static split). Each thread owns the same contiguous range as in the static split and claims it chunk
// by chunk through an atomic cursor; a thread that runs out claims chunks from the other threads' cursors (t+1, t+2,
// ...). A slow core (e.g. one whose SMT sibling runs the spinning engine thread, ~9% slower) then no longer holds the
// whole visit at the phase barrier. Per-item arithmetic is unchanged and down chunks start at even rows, so the
// output is bit-identical to the static split.
typedef struct {
    _Atomic int next;
    int end;
} __attribute__((aligned(64))) StealRange;
static StealRange g_steal[MAX_THREADS];
static int g_steal_up = 0, g_steal_down = 0;  // chunk sizes (blocks, row pairs); 0 = static split

// Item space of a phase: 16-pair blocks of the visit's slot list (minus the GPU's K-split part of its first slot),
// or output row pairs.
static void phase_items(int phase, int* first, int* total) {
    if (g_mt_on) {
        *first = 0;
        *total = phase == 1 ? g_mt.nd * (INTER / 16) : HIDDEN / 2;
        return;
    }
    if (phase == 1) {
        *first = g.vs.ks / 16;
        *total = g.vs.nsl * (INTER / 16) - *first;
    } else {
        *first = 0;
        *total = HIDDEN / 2;
    }
}

static float* g_yout;  // the current visit's result: sh->y, or a MULTI token's mt_y[t] (token-by-token path)
static void do_items(const LayerBanks* lb, int phase, int i0, int i1) {
    if (g_mt_on) {
        if (phase == 1) phase_up_multi(lb, i0, i1);
        else phase_down_multi(lb, 2 * i0, 2 * i1);
        return;
    }
    if (phase == 1) phase_up(lb, &g.vs, i0, i1);
    else if (g_down_kouter) phase_down_kouter(lb, &g.vs, 2 * i0, 2 * i1, g_yout);
    else phase_down(lb, &g.vs, 2 * i0, 2 * i1, g_yout);
}

// Coordinator, before publishing a stealing phase: each thread's cursor at the start of its static range.
static void steal_setup(int phase) {
    int first, total;
    phase_items(phase, &first, &total);
    const int nt = g.n_threads;
    for (int t = 0; t < nt; ++t) {
        atomic_store_explicit(&g_steal[t].next, first + (int)((long)total * t / nt), memory_order_relaxed);
        g_steal[t].end = first + (int)((long)total * (t + 1) / nt);
    }
}

static void do_phase_share_(int phase, int t, int nt);
static void do_phase_share(int phase, int t, int nt) {
    if (!g.thr_stats) { do_phase_share_(phase, t, nt); return; }
    g.t_start[t] = now_ns();
    do_phase_share_(phase, t, nt);
    g.t_end[t] = now_ns();
}

static void do_phase_share_(int phase, int t, int nt) {
    const LayerBanks* lb = &g.layers[g.vs.layer];
    const int ch = phase == 1 ? g_steal_up : g_steal_down;
    if (ch > 0) {
        for (int v = 0; v < nt; ++v) {
            StealRange* R = &g_steal[(t + v) % nt];
            const int end = R->end;
            for (;;) {
                const int i = atomic_fetch_add_explicit(&R->next, ch, memory_order_relaxed);
                if (i >= end) break;
                do_items(lb, phase, i, i + ch < end ? i + ch : end);
            }
        }
        return;
    }
    int first, total;  // static split: thread t takes the t-th of nt equal contiguous ranges
    phase_items(phase, &first, &total);
    do_items(lb, phase, first + (int)((long)total * t / nt), first + (int)((long)total * (t + 1) / nt));
}

static void pin_to_cpu(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static void* worker_main(void* arg) {
    Worker* w = (Worker*)arg;
    pin_to_cpu(w->cpu);
    _mm_setcsr(_mm_getcsr() & ~0x8040u);  // no DAZ/FTZ: E4M3 subnormal scales pass through f32 denormals
    uint32_t seen = atomic_load(&g.job_gen);
    unsigned spins = 0;
    for (;;) {
        const uint32_t gen = atomic_load_explicit(&g.job_gen, memory_order_acquire);
        if (gen != seen) {
            seen = gen;
            do_phase_share(g.phase, w->idx, g.n_threads);
            atomic_fetch_add_explicit(&g.phase_done, 1, memory_order_acq_rel);
            spins = 0;
            continue;
        }
        if (atomic_load_explicit(&g.stop, memory_order_relaxed)) break;
        if (++spins < (g_use_mwaitx ? 8u : 1024u)) {
            wait_change(&g.job_gen, seen);
            continue;
        }
        spins = 0;
        if (now_ns() < atomic_load(&g.hot_until)) continue;
        atomic_fetch_add(&g.sleepers, 1);
        if (atomic_load(&g.job_gen) == seen && now_ns() >= atomic_load(&g.hot_until))
            futex_wait_u32(&g.job_gen, seen, 100ull * 1000 * 1000);
        atomic_fetch_sub(&g.sleepers, 1);
    }
    return NULL;
}

static void run_phase(int phase) {
    atomic_store_explicit(&g.phase_done, 0, memory_order_relaxed);
    g.phase = phase;
    if ((phase == 1 ? g_steal_up : g_steal_down) > 0) steal_setup(phase);
    if (g.thr_stats) g.phase_t0 = now_ns();
    atomic_fetch_add(&g.job_gen, 1);  // seq_cst (so also a release of the cursors): pairs with the sleepers check below
    if (atomic_load(&g.sleepers)) futex_wake_u32(&g.job_gen, INT_MAX);
    do_phase_share(phase, 0, g.n_threads);
    atomic_fetch_add_explicit(&g.phase_done, 1, memory_order_acq_rel);
    while (atomic_load_explicit(&g.phase_done, memory_order_acquire) < (uint32_t)g.n_threads) _mm_pause();
    if (g.thr_stats) {
        uint64_t smax = 0, emin = UINT64_MAX, emax = 0;
        double esum = 0;
        for (int t = 0; t < g.n_threads; ++t) {
            g.st_dur[phase - 1][t] += g.t_end[t] - g.t_start[t];
            if (g.t_start[t] > smax) smax = g.t_start[t];
            if (g.t_end[t] < emin) emin = g.t_end[t];
            if (g.t_end[t] > emax) emax = g.t_end[t];
            esum += (double)g.t_end[t];
        }
        g.st_wake[phase - 1] += smax - g.phase_t0;
        g.st_spread[phase - 1] += emax - emin;
        g.st_lag[phase - 1] += (uint64_t)((double)emax - esum / g.n_threads);
    }
}

static void handle_multi(uint32_t seq);
static void handle_request(uint32_t seq) {
    if (g.sh->mode == CPU_MOE_MODE_MULTI) { handle_multi(seq); return; }
    const uint64_t t0 = now_ns();
    CpuMoeShared* sh = g.sh;
    VisitState* vs = &g.vs;
    g_yout = (float*)sh->y;
    vs->layer = (int)sh->layer;
    vs->mode = (int)sh->mode;  // only CPU_MOE_MODE_FULL is issued
    const uint32_t mask = sh->slot_mask & 0xffu;
    vs->k0 = mask ? 0 : (sh->slot_begin < TOPK ? (int)sh->slot_begin : TOPK - 1);
    vs->ks = (!mask && KERNEL_IS_INT(g_kernel) && sh->k_split > 0 && sh->k_split < INTER && sh->k_split % 128 == 0) ? (int)sh->k_split : 0;
    set_slots(vs, mask);
    if (vs->layer < 0 || vs->layer >= MAX_LAYERS || !g.layers[vs->layer].w13_codes) vs->layer = 0;  // never fault
    for (int k = 0; k < TOPK; ++k) {
        int e = sh->ids[k];
        vs->ids[k] = (e < 0 || e > 127) ? 0 : e;
        vs->wts[k] = sh->wts[k];
    }
    if (KERNEL_IS_INT(g_kernel) && !g_quant_scalar) {
        quant_x_f16((const uint16_t*)sh->x, HIDDEN, &vs->xq);
    } else {
        float x[HIDDEN];
        for (int i = 0; i < HIDDEN; ++i) x[i] = f16_to_f32(sh->x[i]);
        prep_input(vs, x);
    }
    const uint64_t t1 = now_ns();
    run_phase(1);
    const uint64_t t2 = now_ns();
    run_phase(2);
    const uint64_t t3 = now_ns();
    g.prep_ns += t1 - t0;
    g.up_ns += t2 - t1;
    g.down_ns += t3 - t2;
    g.slots_done += (double)vs->nsl - (double)vs->ks / INTER;
    atomic_thread_fence(memory_order_release);
    *(volatile uint32_t*)&sh->resp_seq = seq;
    const uint64_t dt = now_ns() - t0;
    g.requests++;
    g.busy_ns += dt;
    if (dt > g.max_ns) g.max_ns = dt;
}

// Mode MULTI (spec verify): n tokens, token t's CPU slots in mt_mask[t]; result t in mt_y[t] (tokens with a mask).
static void handle_multi(uint32_t seq) {
    const uint64_t t0 = now_ns();
    CpuMoeShared* sh = g.sh;
    int layer = (int)sh->layer;
    if (layer < 0 || layer >= MAX_LAYERS || !g.layers[layer].w13_codes) layer = 0;  // never fault
    const int n = sh->n_tok < 1 ? 1 : sh->n_tok > MT_MAX ? MT_MAX : (int)sh->n_tok;
    const int fast = g_kernel == KERNEL_I16 && !g_mt_seq;
    double slots = 0;
    g_mt.n = n;
    g_mt.nd = 0;
    for (int t = 0; t < n; ++t) {
        const uint32_t mask = sh->mt_mask[t] & 0xffu;
        g_mt.nsl[t] = 0;
        g_mt.y[t] = sh->mt_y[t];
        for (int k = 0; k < TOPK; ++k) {
            const int e = sh->mt_ids[t][k];
            g_mt.ids[t][k] = (e < 0 || e > 127) ? 0 : e;
            g_mt.wts[t][k] = sh->mt_wts[t][k];
        }
        if (!mask) continue;
        slots += __builtin_popcount(mask);
        if (!fast) {  // token by token through the single-token phases (other kernels, or SQUID_CPU_MOE_MULTI=seq)
            VisitState* vs = &g.vs;
            vs->layer = layer;
            vs->mode = CPU_MOE_MODE_FULL;
            vs->k0 = 0;
            vs->ks = 0;
            set_slots(vs, mask);
            for (int k = 0; k < TOPK; ++k) { vs->ids[k] = g_mt.ids[t][k]; vs->wts[k] = g_mt.wts[t][k]; }
            if (KERNEL_IS_INT(g_kernel) && !g_quant_scalar) {
                quant_x_f16((const uint16_t*)sh->mt_x[t], HIDDEN, &vs->xq);
            } else {
                float x[HIDDEN];
                for (int i = 0; i < HIDDEN; ++i) x[i] = f16_to_f32(sh->mt_x[t][i]);
                prep_input(vs, x);
            }
            g_yout = sh->mt_y[t];
            run_phase(1);
            run_phase(2);
            continue;
        }
        if (g_quant_scalar) {
            float x[HIDDEN];
            for (int i = 0; i < HIDDEN; ++i) x[i] = f16_to_f32(sh->mt_x[t][i]);
            quant_x(x, HIDDEN, &g_mt.xq[t]);
        } else {
            quant_x_f16((const uint16_t*)sh->mt_x[t], HIDDEN, &g_mt.xq[t]);
        }
        for (int k = 0; k < TOPK; ++k) {
            if (!((mask >> k) & 1u)) continue;
            g_mt.sl[t][g_mt.nsl[t]++] = k;
            const int e = g_mt.ids[t][k];
            int d = 0;
            while (d < g_mt.nd && g_mt.de[d] != e) ++d;
            if (d == g_mt.nd) { g_mt.de[d] = e; g_mt.nu[d] = 0; g_mt.nd++; }
            g_mt.ut[d][g_mt.nu[d]] = (uint8_t)t;
            g_mt.uk[d][g_mt.nu[d]] = (uint8_t)k;
            g_mt.nu[d]++;
        }
    }
    g.vs.layer = layer;  // do_phase_share reads the layer from vs
    const uint64_t t1 = now_ns();
    if (fast && g_mt.nd > 0) {
        g_mt_on = 1;
        run_phase(1);
        const uint64_t t2 = now_ns();
        run_phase(2);
        g_mt_on = 0;
        g.up_ns += t2 - t1;
        g.down_ns += now_ns() - t2;
    }
    g_yout = (float*)sh->y;
    g.prep_ns += t1 - t0;
    g.slots_done += slots;
    atomic_thread_fence(memory_order_release);
    *(volatile uint32_t*)&sh->resp_seq = seq;
    const uint64_t dt = now_ns() - t0;
    g.requests++;
    g.multi_requests++;
    g.busy_ns += dt;
    if (dt > g.max_ns) g.max_ns = dt;
}

static void* coordinator_main(void* arg) {
    Worker* w = (Worker*)arg;
    pin_to_cpu(w->cpu);
    _mm_setcsr(_mm_getcsr() & ~0x8040u);
    uint32_t last = *(volatile uint32_t*)&g.sh->req_seq;
    unsigned spins = 0;
    uint64_t cold_since = 0;
    for (;;) {
        const uint32_t s = *(volatile uint32_t*)&g.sh->req_seq;
        if (s != last) {
            atomic_thread_fence(memory_order_acquire);
            last = s;
            atomic_store(&g.hot_until, now_ns() + CPU_MOE_HOT_NS);
            handle_request(s);
            if (g_ka_buf) keepalive_reset(now_ns());
            spins = 0;
            cold_since = 0;
            continue;
        }
        if (atomic_load_explicit(&g.stop, memory_order_relaxed)) break;
        if (++spins < (g_use_mwaitx ? 8u : 256u)) {
            if (g_ka_buf) keepalive_step();
            else wait_change(&g.sh->req_seq, last);
            continue;
        }
        spins = 0;
        const uint64_t t = now_ns();
        if (t < atomic_load(&g.hot_until)) continue;
        if (!cold_since) cold_since = t;
        const uint32_t k = atomic_load(&g.kick);
        atomic_store(&g.coord_sleeping, 1);
        if (*(volatile uint32_t*)&g.sh->req_seq == last && now_ns() >= atomic_load(&g.hot_until))
            futex_wait_u32(&g.kick, k, (t - cold_since < 10ull * 1000 * 1000 * 1000) ? 1000ull * 1000 : 10ull * 1000 * 1000);
        atomic_store(&g.coord_sleeping, 0);
    }
    return NULL;
}

// CPUID + XGETBV directly: __builtin_cpu_supports depends on the compiler runtime (Zig's does not fill it in).
int cpu_moe_available(void) {
    if (g_have_avx512 < 0) {
        unsigned a, b, c, d;
        int ok = 0;
        if (__get_cpuid(1, &a, &b, &c, &d) && (c & (1u << 27)) && (c & (1u << 29)) && (c & (1u << 12))) {  // osxsave, f16c, fma
            unsigned lo, hi;
            __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
            const int os_zmm = (lo & 0xe6u) == 0xe6u;  // SSE, AVX, opmask, ZMM_Hi256, Hi16_ZMM state enabled
            if (os_zmm && __get_cpuid_count(7, 0, &a, &b, &c, &d)) {
                const int f = (b >> 16) & 1, dq = (b >> 17) & 1, bw = (b >> 30) & 1, vl = (b >> 31) & 1, vbmi = (c >> 1) & 1;
                ok = f && dq && bw && vl && vbmi;
                g_have_vnni = ok && ((c >> 11) & 1);
            }
        }
        g_have_avx512 = ok;
        g_kernel = g_have_vnni ? KERNEL_I16 : (ok ? KERNEL_FP32 : KERNEL_SCALAR);
        if (__get_cpuid(0x80000001u, &a, &b, &c, &d)) g_have_mwaitx = (c >> 29) & 1;
        const char* wv = getenv("SQUID_CPU_MOE_WAIT");
        g_use_mwaitx = g_have_mwaitx && wv && !strcmp(wv, "mwaitx");
        const char* kv = getenv("SQUID_CPU_MOE_KERNEL");
        if (kv && !strcmp(kv, "fp32") && ok) g_kernel = KERNEL_FP32;
        if (kv && !strcmp(kv, "i16") && g_have_vnni) g_kernel = KERNEL_I16;
        if (kv && !strcmp(kv, "i16x2") && g_have_vnni) g_kernel = KERNEL_I16X2;
        const char* mv = getenv("SQUID_CPU_MOE_MULTI");
        g_mt_seq = mv && !strcmp(mv, "seq");
        const char* dv = getenv("SQUID_CPU_MOE_DOWN");
        g_down_kouter = dv && !strcmp(dv, "kouter");
        const char* pv = getenv("SQUID_CPU_MOE_PF");
        g_pf_pairs = pv ? atoi(pv) : 0;
        if (g_pf_pairs < 0 || g_pf_pairs > 64) g_pf_pairs = 0;
        const char* hv = getenv("SQUID_CPU_MOE_PF_HINT");
        g_pf_hint = !hv ? 0 : !strcmp(hv, "t1") ? 1 : !strcmp(hv, "t2") ? 2 : !strcmp(hv, "nta") ? 3 : 0;
        const char* qv = getenv("SQUID_CPU_MOE_QUANT");
        g_quant_scalar = qv && !strcmp(qv, "scalar");
        const char* stv = getenv("SQUID_CPU_MOE_STEAL");
        g_steal_up = 1;  // default on (A/B ab11: +2.8%, bit-identical); SQUID_CPU_MOE_STEAL=0 = static split
        g_steal_down = 8;
        if (stv && *stv) {
            char* end;
            const long up = strtol(stv, &end, 10);
            const long dn = (*end == ',') ? strtol(end + 1, NULL, 10) : 16;
            g_steal_up = (up >= 1 && up <= 72) ? (int)up : 0;
            g_steal_down = (g_steal_up && dn >= 2 && dn <= 288 && dn % 2 == 0) ? (int)(dn / 2) : 0;
            if (!g_steal_down) g_steal_up = 0;
        }
        const char* sv = getenv("SQUID_CPU_MOE_STATS");
        g.thr_stats = sv && sv[0] == '1';
        if (kv && !strcmp(kv, "scalar")) g_kernel = KERNEL_SCALAR;
    }
    return g_have_avx512;
}

void cpu_moe_set_layer(int layer, const uint8_t* w13_codes, const uint8_t* w13_scales, const float* es13,
                       const uint8_t* w2_codes, const uint8_t* w2_scales, const float* es2) {
    if (layer < 0 || layer >= MAX_LAYERS) return;
    LayerBanks* lb = &g.layers[layer];
    lb->w13_codes = w13_codes;
    lb->w13_scales = w13_scales;
    lb->w2_codes = w2_codes;
    lb->w2_scales = w2_scales;
    for (int e = 0; e < 128; ++e) {
        lb->xp[e][0] = w13_codes ? w13_codes + (size_t)e * (2 * INTER) * (HIDDEN / 2) : NULL;
        lb->xp[e][1] = w13_scales ? w13_scales + (size_t)e * (2 * INTER) * (HIDDEN / 16) : NULL;
        lb->xp[e][2] = w2_codes ? w2_codes + (size_t)e * HIDDEN * (INTER / 2) : NULL;
        lb->xp[e][3] = w2_scales ? w2_scales + (size_t)e * HIDDEN * (INTER / 16) : NULL;
    }
    if (es13) memcpy(lb->es13, es13, sizeof lb->es13);
    if (es2) memcpy(lb->es2, es2, sizeof lb->es2);
}

void cpu_moe_set_expert(int layer, int e, const uint8_t* w13_codes, const uint8_t* w13_scales, float es13,
                        const uint8_t* w2_codes, const uint8_t* w2_scales, float es2) {
    if (layer < 0 || layer >= MAX_LAYERS || e < 0 || e > 127 || !w13_codes || !w13_scales || !w2_codes || !w2_scales) return;
    LayerBanks* lb = &g.layers[layer];
    if (!lb->w13_codes) {
        // First expert of a layer that has no host bank: every other expert points at this one, so a request naming
        // an expert this side does not hold reads valid memory (wrong numbers, never a fault).
        lb->w13_codes = w13_codes;
        lb->w13_scales = w13_scales;
        lb->w2_codes = w2_codes;
        lb->w2_scales = w2_scales;
        for (int i = 0; i < 128; ++i) {
            lb->xp[i][0] = w13_codes;
            lb->xp[i][1] = w13_scales;
            lb->xp[i][2] = w2_codes;
            lb->xp[i][3] = w2_scales;
            lb->es13[i] = es13;
            lb->es2[i] = es2;
        }
    }
    lb->xp[e][0] = w13_codes;
    lb->xp[e][1] = w13_scales;
    lb->xp[e][2] = w2_codes;
    lb->xp[e][3] = w2_scales;
    lb->es13[e] = es13;
    lb->es2[e] = es2;
}

// cpus: optional list of n_threads CPU ids to pin thread i to (NULL = thread i on CPU i; -1 entries = unpinned).
int cpu_moe_start(CpuMoeShared* shared_host, int n_threads, const int* cpus) {
    cpu_moe_available();
    if (n_threads < 1) n_threads = 1;
    if (n_threads > MAX_THREADS) n_threads = MAX_THREADS;
    g.sh = shared_host;
    g.n_threads = n_threads;
    keepalive_init();
    if (g_ka_buf) {
        printf("[cpu-moe] DRAM keep-alive between visits while decoding: %.1f GB/s (64 MB buffer)\n", g_ka_bpns);
        fflush(stdout);
    }
    if (g_down_kouter) {
        printf("[cpu-moe] down phase: slot-outer order\n");
        fflush(stdout);
    }
    if (g_steal_up) {
        printf("[cpu-moe] work stealing: chunks of %d up blocks (%d pairs), %d down rows\n", g_steal_up, 16 * g_steal_up, 2 * g_steal_down);
        fflush(stdout);
    }
    if (g_pf_pairs) {
        static const char* hn[4] = {"t0", "t1", "t2", "nta"};
        printf("[cpu-moe] software prefetch: %d row pairs ahead, hint %s\n", g_pf_pairs, hn[g_pf_hint]);
        fflush(stdout);
    }
    atomic_store(&g.stop, 0);
    atomic_store(&g.hot_until, 0);
    for (int i = 0; i < n_threads; ++i) {
        g.workers[i].idx = i;
        g.workers[i].cpu = cpus ? cpus[i] : i;
    }
    // SQUID_CPU_MOE_CPUS=a,b,c,...: thread i runs on the i-th listed CPU (overrides the caller's list).
    const char* cl = getenv("SQUID_CPU_MOE_CPUS");
    if (cl && *cl) {
        const char* q = cl;
        for (int i = 0; i < n_threads && *q; ++i) {
            char* end;
            const long c = strtol(q, &end, 10);
            if (end == q) break;
            g.workers[i].cpu = (int)c;
            q = (*end == ',') ? end + 1 : end;
        }
        printf("[cpu-moe] thread CPUs:");
        for (int i = 0; i < n_threads; ++i) printf(" %d", g.workers[i].cpu);
        printf("\n");
        fflush(stdout);
    }
    for (int i = 1; i < n_threads; ++i)
        if (pthread_create(&g.workers[i].th, NULL, worker_main, &g.workers[i]) != 0) return -1;
    if (pthread_create(&g.workers[0].th, NULL, coordinator_main, &g.workers[0]) != 0) return -1;
    g.started = 1;
    return 0;
}

// Called by the engine before decode steps: spin for the next CPU_MOE_HOT_NS and wake any sleepers now, so the
// first request of a generation does not pay a wake-up.
void cpu_moe_kick(void) {
    atomic_store(&g.hot_until, now_ns() + CPU_MOE_HOT_NS);
    if (atomic_load(&g.coord_sleeping)) {
        atomic_fetch_add(&g.kick, 1);
        futex_wake_u32(&g.kick, 1);
    }
    if (atomic_load(&g.sleepers)) futex_wake_u32(&g.job_gen, INT_MAX);
}

void cpu_moe_stop(void) {
    if (!g.started) return;
    atomic_store(&g.stop, 1);
    atomic_fetch_add(&g.kick, 1);
    futex_wake_u32(&g.kick, 1);
    atomic_fetch_add(&g.job_gen, 0);
    futex_wake_u32(&g.job_gen, INT_MAX);
    for (int i = 0; i < g.n_threads; ++i) pthread_join(g.workers[i].th, NULL);
    g.started = 0;
    if (g.requests) {
        const double n = (double)g.requests, kk = g.slots_done / n;  // mean slots per visit
        const double up_b = kk * 2.0 * INTER * (HIDDEN / 2 + HIDDEN / 16), dn_b = kk * HIDDEN * (INTER / 2 + INTER / 16);
        printf("[cpu-moe] per visit: prep %.1f us | up %.1f us (%.1f GB/s) | down %.1f us (%.1f GB/s)\n",
               g.prep_ns / n * 1e-3, g.up_ns / n * 1e-3, up_b / (g.up_ns / n), g.down_ns / n * 1e-3, dn_b / (g.down_ns / n));
        if (g.thr_stats)
            printf("[cpu-moe] threads per visit: up last-start %.1f us, end spread %.1f us, max - mean end %.1f us | "
                   "down last-start %.1f us, end spread %.1f us, max - mean end %.1f us\n",
                   g.st_wake[0] / n * 1e-3, g.st_spread[0] / n * 1e-3, g.st_lag[0] / n * 1e-3,
                   g.st_wake[1] / n * 1e-3, g.st_spread[1] / n * 1e-3, g.st_lag[1] / n * 1e-3);
        if (g.thr_stats) {
            for (int ph = 0; ph < 2; ++ph) {
                printf("[cpu-moe] %s per thread (us):", ph ? "down" : "up");
                for (int t = 0; t < g.n_threads; ++t) printf(" %.1f", g.st_dur[ph][t] / n * 1e-3);
                printf("\n");
            }
        }
        fflush(stdout);
    }
    if (g_ka_buf) {
        printf("[cpu-moe] keep-alive streamed %.2f GB\n", g_ka_total * 1e-9);
        fflush(stdout);
        free(g_ka_buf);
        g_ka_buf = NULL;
    }
}

void cpu_moe_stats(uint64_t* requests, uint64_t* busy_ns, uint64_t* max_ns) {
    *requests = g.requests;
    *busy_ns = g.busy_ns;
    *max_ns = g.max_ns;
}

size_t cpu_moe_shared_bytes(void) { return sizeof(CpuMoeShared); }

void cpu_moe_pin_current_thread(int cpu) { pin_to_cpu(cpu); }

int cpu_moe_write_file(const char* path, const void* data, size_t bytes) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    const size_t n = fwrite(data, 1, bytes, f);
    fclose(f);
    return n == bytes ? 0 : -1;
}

// Reads up to `max` decimal token ids separated by commas or whitespace; returns how many were read.
int cpu_moe_read_tokens(const char* path, int32_t* out, int max) {
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    int n = 0, c, have = 0;
    long v = 0;
    while (n < max && (c = fgetc(f)) != EOF) {
        if (c >= '0' && c <= '9') {
            v = v * 10 + (c - '0');
            have = 1;
        } else if (have) {
            out[n++] = (int32_t)v;
            v = 0;
            have = 0;
        }
    }
    if (have && n < max) out[n++] = (int32_t)v;
    fclose(f);
    return n;
}

// One teacher-forced position: rec = {forced id, argmax id (as int bits), log p(forced), log p(argmax), top1 - top2}.
void cpu_moe_score_logits(const float* logits, int vocab, int32_t forced, float rec[5]) {
    float m1 = -INFINITY, m2 = -INFINITY;
    int am = 0;
    for (int j = 0; j < vocab; ++j) {
        const float v = logits[j];
        if (v > m1) {
            m2 = m1;
            m1 = v;
            am = j;
        } else if (v > m2) {
            m2 = v;
        }
    }
    double s = 0.0;
    for (int j = 0; j < vocab; ++j) s += exp((double)logits[j] - (double)m1);
    const double lse = (double)m1 + log(s);
    const int32_t fi = (forced >= 0 && forced < vocab) ? forced : am;
    memcpy(&rec[0], &forced, 4);
    memcpy(&rec[1], &am, 4);
    rec[2] = (float)((double)logits[fi] - lse);
    rec[3] = (float)((double)m1 - lse);
    rec[4] = m1 - m2;
}

const char* cpu_moe_kernel_name(void) {
    cpu_moe_available();
    static const char* names[2][4] = {{"scalar", "fp32-avx512", "int16-vnni", "int16x2-vnni"},
                                      {"scalar, mwaitx", "fp32-avx512, mwaitx", "int16-vnni, mwaitx", "int16x2-vnni, mwaitx"}};
    return names[g_use_mwaitx ? 1 : 0][g_kernel];
}

// Single-threaded reference of a full visit (no runtime needed): used by the self-test.
void cpu_moe_reference(int layer, const uint16_t* x16, const int32_t* ids, const float* wts, float* y_out) {
    VisitState* vs = (VisitState*)aligned_alloc(64, sizeof(VisitState));
    vs->layer = layer;
    vs->k0 = 0;
    vs->ks = 0;
    set_slots(vs, 0);
    for (int k = 0; k < TOPK; ++k) { vs->ids[k] = ids[k]; vs->wts[k] = wts[k]; }
    float x[HIDDEN];
    for (int i = 0; i < HIDDEN; ++i) x[i] = f16_to_f32(x16[i]);
    prep_input(vs, x);
    phase_up(&g.layers[layer], vs, 0, TOPK * INTER / 16);
    phase_down(&g.layers[layer], vs, 0, HIDDEN, y_out);
    free(vs);
}

#ifdef CPU_MOE_STANDALONE_BENCH
#include <sys/mman.h>
// Standalone check + bench on synthetic weights:  cc -O3 -DCPU_MOE_STANDALONE_BENCH cpu_moe.c -lm -lpthread
static double secs(void) { return now_ns() * 1e-9; }
static uint8_t* g_ka;
static size_t g_ka_bytes, g_ka_pos;
static __m512i g_ka_acc;
// Spin until t_end; with KEEP_STREAM=n, stream n bytes of a 256 MB buffer between clock checks (DRAM keep-alive).
static void gpu_time(double t_end, size_t stream_bytes) __attribute__((target("avx512f")));
static double g_keep_gbps;  // KEEP_GBPS: rate-limit the keep-alive stream (0 = as fast as one thread can)
static void gpu_time(double t_end, size_t stream_bytes) {
    const double t_start = secs();
    double streamed = 0;
    while (secs() < t_end) {
        if (stream_bytes && g_ka && g_keep_gbps > 0 && streamed > (secs() - t_start) * g_keep_gbps * 1e9) {
            _mm_pause();
            continue;
        }
        streamed += stream_bytes;
        if (stream_bytes && g_ka) {
            for (size_t i = 0; i < stream_bytes; i += 64) {
                g_ka_acc = _mm512_xor_si512(g_ka_acc, _mm512_load_si512((const void*)(g_ka + g_ka_pos)));
                g_ka_pos = (g_ka_pos + 64) % g_ka_bytes;
            }
        } else {
            _mm_pause();
        }
    }
}
int main(int argc, char** argv) {
    int nt = argc > 1 ? atoi(argv[1]) : 8;
    cpu_moe_available();
    printf("kernel %s, threads %d\n", cpu_moe_kernel_name(), nt);
    const size_t E = 128;
    const size_t w13c = E * 2 * INTER * (HIDDEN / 2), w13s = E * 2 * INTER * (HIDDEN / 16);
    const size_t w2c = E * HIDDEN * (INTER / 2), w2s = E * HIDDEN * (INTER / 16);
    const int NL = 4;  // 4 layers so the working set (2.3 GB) is far beyond L3
    const size_t mem_bytes = NL * (w13c + w13s + w2c + w2s);
    uint8_t* mem = (uint8_t*)mmap(NULL, mem_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    const int nohuge = getenv("NOHUGE") != NULL;
    madvise(mem, mem_bytes, nohuge ? MADV_NOHUGEPAGE : MADV_HUGEPAGE);
    printf("pages: %s\n", nohuge ? "4K (MADV_NOHUGEPAGE)" : "THP (MADV_HUGEPAGE)");
    srand(1);
    for (size_t i = 0; i < NL * (w13c + w13s + w2c + w2s); ++i) mem[i] = (uint8_t)rand();
    // keep scales sane: e4m3 in [2^-4, 2^2]
    for (int l = 0; l < NL; ++l) {
        uint8_t* base = mem + l * (w13c + w13s + w2c + w2s);
        for (size_t i = 0; i < w13s; ++i) base[w13c + i] = 0x20 + (rand() % 0x28);
        for (size_t i = 0; i < w2s; ++i) base[w13c + w13s + w2c + i] = 0x20 + (rand() % 0x28);
        float es[128];
        for (int e = 0; e < 128; ++e) es[e] = 0.01f + 0.001f * e;
        cpu_moe_set_layer(l, base, base + w13c, es, base + w13c + w13s, base + w13c + w13s + w2c, es);
    }
    // correctness of the AVX-512 row kernel vs scalar reference
    {
        float x[HIDDEN], xs[HIDDEN];
        for (int i = 0; i < HIDDEN; ++i) x[i] = (rand() / (float)RAND_MAX - 0.5f) * 4.f;
        prep_x_split(x, xs, HIDDEN);
        static XQ xq;
        quant_x(x, HIDDEN, &xq);
        double maxrel = 0, maxrel16 = 0, se = 0, sr = 0, se2 = 0, sf = 0;
        for (int r = 0; r < 200; ++r) {
            const uint8_t* c = mem + (size_t)r * 1152 * 7;
            const uint8_t* s = mem + w13c + (size_t)r * 144 * 7;
            float a, b, a16, b16, a2, b2;
            rows2_dot_avx512(c, s, c + 1152, s + 144, xs, HIDDEN, &a, &b);
            if (g_have_vnni) rows2_dot_i16(c, s, c + 1152, s + 144, &xq, HIDDEN, &a16, &b16, NULL, NULL, 0);
            else a16 = a, b16 = b;
            if (g_have_vnni) rows2_dot_i16x2(c, s, c + 1152, s + 144, &xq, HIDDEN, &a2, &b2, NULL, NULL, 0);
            else a2 = a, b2 = b;
            float ra = row_dot_ref(c, s, x, HIDDEN), rb = row_dot_ref(c + 1152, s + 144, x, HIDDEN);
            double d = fabs(a - ra) / (fabs(ra) + 1e-3), d2 = fabs(b - rb) / (fabs(rb) + 1e-3);
            if (d > maxrel) maxrel = d;
            if (d2 > maxrel) maxrel = d2;
            d = fabs(a16 - ra) / (fabs(ra) + 1e-3), d2 = fabs(b16 - rb) / (fabs(rb) + 1e-3);
            if (d > maxrel16) maxrel16 = d;
            if (d2 > maxrel16) maxrel16 = d2;
            se += (a16 - ra) * (a16 - ra) + (b16 - rb) * (b16 - rb);
            se2 += (a2 - ra) * (a2 - ra) + (b2 - rb) * (b2 - rb);
            sf += (a - ra) * (a - ra) + (b - rb) * (b - rb);
            sr += ra * ra + rb * rb;
        }
        printf("row kernels vs scalar reference (rel L2): fp32 %.3g | int16 %.3g | int16x2 %.3g   (fp32 max rel err %.3g, int16 %.3g)\n",
               sqrt(sf / sr), sqrt(se / sr), sqrt(se2 / sr), maxrel, maxrel16);
        // vectorized quantization must match the scalar one bit for bit (random magnitudes incl. zeros, subnormals,
        // fp16 extremes, values at the clamp)
        static XQ qa, qb;
        long bad = 0;
        for (int trial = 0; trial < 2000; ++trial) {
            uint16_t xh[HIDDEN];
            float xf32[HIDDEN];
            for (int i = 0; i < HIDDEN; ++i) {
                uint16_t h = (uint16_t)rand();
                if ((h & 0x7c00) == 0x7c00) h &= 0x7bff;  // no inf / NaN
                if (trial % 7 == 0 && (i & 15) < 4) h = 0;
                if (trial % 11 == 0) h &= 0x83ff;  // subnormals only
                if (trial % 13 == 0 && (i & 15) == 0) h = 0x7bff;  // 65504
                xh[i] = h;
                xf32[i] = f16_to_f32(h);
            }
            if (trial % 17 == 0) for (int i = 16; i < 32; ++i) xh[i] = 0, xf32[i] = 0.f;  // all-zero block
            memset(&qa, 0, sizeof qa); memset(&qb, 0, sizeof qb);
            quant_x(xf32, HIDDEN, &qa);
            quant_x_f16(xh, HIDDEN, &qb);
            if (memcmp(&qa, &qb, sizeof qa)) ++bad;
            for (int bb = 0; bb < HIDDEN / 16; ++bb) quant_block_v(&qb, bb, _mm512_loadu_ps(xf32 + 16 * bb));  // activation path
            if (memcmp(&qa, &qb, sizeof qa)) ++bad;
        }
        printf("vector quant vs scalar quant: %s (%ld mismatching of 4000)\n", bad ? "MISMATCH" : "bit-identical", bad);
    }
    // threaded runtime driven by a fake GPU (this thread)
    CpuMoeShared* sh = (CpuMoeShared*)aligned_alloc(4096, sizeof(CpuMoeShared));
    memset(sh, 0, sizeof *sh);
    // SLOT_BEGIN=n: the CPU computes slots [n, 8) only (GPU + CPU split); the reference check then covers those.
    sh->slot_begin = getenv("SLOT_BEGIN") ? (uint32_t)atoi(getenv("SLOT_BEGIN")) : 0;
    if (sh->slot_begin) printf("CPU computes slots %u..7 only\n", sh->slot_begin);
    // SLOT_MASK=m (1-255): the CPU computes exactly the slots in bit mask m (decode expert cache mode).
    sh->slot_mask = getenv("SLOT_MASK") ? (uint32_t)strtoul(getenv("SLOT_MASK"), NULL, 0) & 0xffu : 0;
    if (sh->slot_mask) printf("CPU computes the slots in mask 0x%02x only\n", sh->slot_mask);
    cpu_moe_start(sh, nt, NULL);
    cpu_moe_kick();
    usleep(20000);
    float yref[HIDDEN];
    double best = 1e9, tot = 0;
    int iters = 200;
    // GAP_US: idle time between visits (the GPU's own work between CPU visits in the engine).
    const double gap_us = getenv("GAP_US") ? atof(getenv("GAP_US")) : 0;
    if (gap_us > 0) printf("gap between visits: %.0f us\n", gap_us);
    if (getenv("ITERS")) iters = atoi(getenv("ITERS"));
    const double keep_ns = getenv("KEEP_NS") ? atof(getenv("KEEP_NS")) : 0;
    const size_t ka_bytes = 256u << 20;
    uint8_t* ka = keep_ns > 0 ? (uint8_t*)malloc(ka_bytes) : NULL;
    if (ka) { memset(ka, 1, ka_bytes); printf("keep-alive: one 64 B line every %.0f ns during gaps (%.2f GB/s)\n", keep_ns, 64.0 / keep_ns); }
    size_t ka_pos = 0;
    unsigned ka_sum = 0;
    const char* flag = getenv("FLAG_FILE");
    if (flag) { FILE* ff = fopen(flag, "w"); if (ff) fclose(ff); }
    for (int it = 0; it < iters; ++it) {
        int layer = it % NL;
        for (int k = 0; k < TOPK; ++k) { sh->ids[k] = (it * 37 + k * 13) % 128; sh->wts[k] = 0.125f; }
        for (int i = 0; i < HIDDEN; ++i) sh->x[i] = f32_to_f16((float)((i * 7 + it) % 23) / 11.0f - 1.0f);
        sh->layer = layer;
        sh->mode = CPU_MOE_MODE_FULL;
        double t0 = secs();
        atomic_thread_fence(memory_order_release);
        *(volatile uint32_t*)&sh->req_seq = it + 1;
        while (*(volatile uint32_t*)&sh->resp_seq != (uint32_t)(it + 1)) _mm_pause();
        double dt = secs() - t0;
        if (it >= 10) { tot += dt; if (dt < best) best = dt; }
        if (gap_us > 0) {
            const double tg = secs() + gap_us * 1e-6;
            if (keep_ns > 0) {  // light DRAM traffic during the gap: one 64 B line every keep_ns
                double tn = secs();
                while (secs() < tg) {
                    if (secs() >= tn) { ka_sum += *(volatile uint8_t*)(ka + ka_pos); ka_pos = (ka_pos + 64) % ka_bytes; tn += keep_ns * 1e-9; }
                    _mm_pause();
                }
            } else {
                while (secs() < tg) _mm_pause();
            }
        }
        if (it == 5) {
            const int kern = g_kernel;
            g_kernel = KERNEL_SCALAR;
            float wref[TOPK];
            for (int k = 0; k < TOPK; ++k)
                wref[k] = (sh->slot_mask ? !((sh->slot_mask >> k) & 1u) : (k < (int)sh->slot_begin)) ? 0.f : sh->wts[k];
            cpu_moe_reference(layer, sh->x, sh->ids, wref, yref);
            g_kernel = kern;
            double maxd = 0, maxv = 0;
            for (int i = 0; i < HIDDEN; ++i) { double d = fabs(sh->y[i] - yref[i]); if (d > maxd) maxd = d; if (fabs(yref[i]) > maxv) maxv = fabs(yref[i]); }
            double se = 0, sr = 0;
            for (int i = 0; i < HIDDEN; ++i) { se += (sh->y[i] - yref[i]) * (sh->y[i] - yref[i]); sr += yref[i] * yref[i]; }
            uint32_t ck = 0;
            for (int i = 0; i < HIDDEN; ++i) { uint32_t u; memcpy(&u, &sh->y[i], 4); ck = ck * 31u + u; }
            printf("threaded visit vs scalar reference: max abs diff %.3g (max |y| %.3g), rel L2 %.3g | y checksum %08x\n", maxd, maxv, sqrt(se / sr), ck);
        }
    }
    // MULTI (spec verify) vs single-token requests: bit-identical per token, and the time of one n-token request vs n
    // single ones. MT_SHARE = how many of a token's CPU slots reuse an expert of token 0 (default 1 of 4).
    {
        const int share = getenv("MT_SHARE") ? atoi(getenv("MT_SHARE")) : 1;
        const int ncpu = getenv("MT_SLOTS") ? atoi(getenv("MT_SLOTS")) : 4;
        uint32_t seq = 50000;
        static float ysingle[MT_MAX][HIDDEN];
        for (int n = 2; n <= MT_MAX; n *= 2) {
            double tm = 1e9, ts = 1e9;
            long bad = 0;
            for (int rep = 0; rep < 40; ++rep) {
                const int layer = rep % NL;
                int ids[MT_MAX][TOPK];
                uint32_t mask[MT_MAX];
                for (int tt = 0; tt < n; ++tt) {
                    // token tt: distinct ids; slots 0..ncpu-1 are CPU slots; the first `share` reuse token 0's
                    for (int k = 0; k < TOPK; ++k) ids[tt][k] = (rep * 53 + tt * 29 + k * 13) % 128;
                    if (tt > 0) for (int k = 0; k < share && k < ncpu; ++k) ids[tt][k] = ids[0][k];
                    for (int k = 0; k < TOPK; ++k) for (int j = 0; j < k; ++j) if (ids[tt][j] == ids[tt][k]) ids[tt][k] = (ids[tt][k] + 64 + k) % 128;
                    mask[tt] = (1u << ncpu) - 1u;
                    if (tt == 1 && rep % 3 == 0) mask[tt] = 0;  // a token with every slot cached
                    for (int k = 0; k < TOPK; ++k) { sh->mt_ids[tt][k] = ids[tt][k]; sh->mt_wts[tt][k] = 0.1f + 0.01f * k; }
                    for (int i = 0; i < HIDDEN; ++i) sh->mt_x[tt][i] = f32_to_f16((float)((i * 7 + tt * 5 + rep) % 23) / 11.0f - 1.0f);
                    sh->mt_mask[tt] = mask[tt];
                }
                sh->layer = layer;
                sh->n_tok = n;
                sh->mode = CPU_MOE_MODE_MULTI;
                double t0 = secs();
                atomic_thread_fence(memory_order_release);
                *(volatile uint32_t*)&sh->req_seq = ++seq;
                while (*(volatile uint32_t*)&sh->resp_seq != seq) _mm_pause();
                const double dm = secs() - t0;
                // the same tokens as single requests: once warm on the same layer (results to compare), then timed
                // cold on another layer with the ids shifted (the warm pass would find the multi pass's experts in L3)
                double dsg = 0;
                for (int pass = 0; pass < 2; ++pass) {
                    sh->layer = pass ? (layer + 2) % NL : layer;
                    t0 = secs();
                    for (int tt = 0; tt < n; ++tt) {
                        if (!mask[tt]) continue;
                        for (int k = 0; k < TOPK; ++k) { sh->ids[k] = pass ? (ids[tt][k] + 41) % 128 : ids[tt][k]; sh->wts[k] = sh->mt_wts[tt][k]; }
                        memcpy(sh->x, sh->mt_x[tt], sizeof sh->x);
                        sh->slot_mask = mask[tt];
                        sh->slot_begin = 0;
                        sh->mode = CPU_MOE_MODE_FULL;
                        atomic_thread_fence(memory_order_release);
                        *(volatile uint32_t*)&sh->req_seq = ++seq;
                        while (*(volatile uint32_t*)&sh->resp_seq != seq) _mm_pause();
                        if (!pass) memcpy(ysingle[tt], sh->y, sizeof sh->y);
                    }
                    dsg = secs() - t0;
                }
                if (rep == 4) tm = ts = 0;
                if (rep >= 4) { tm += dm / 36; ts += dsg / 36; }  // means (best-of would pick the masked-token reps)
                for (int tt = 0; tt < n; ++tt)
                    if (mask[tt]) bad += memcmp(ysingle[tt], sh->mt_y[tt], sizeof ysingle[tt]) != 0;
            }
            sh->slot_mask = 0;
            printf("MULTI n=%d (%d CPU slots/token, %d shared with token 0): %s vs single requests (%ld token results differ) | "
                   "mean %.0f us vs %.0f us for %d single requests (%.2fx)\n",
                   n, ncpu, share, bad ? "DIFFERENT" : "bit-identical", bad, tm * 1e6, ts * 1e6, n, tm / ts);
        }
    }
    // PATTERN: emulate whole decode tokens. One char per visit: H = CPU visit, V = GPU-only (VRAM) visit; every visit
    // is preceded by ATTN_US of GPU-only time and V visits add VRAM_US more.
    const char* pat = getenv("PATTERN");
    if (pat) {
        const double attn_us = getenv("ATTN_US") ? atof(getenv("ATTN_US")) : 165;
        const double vram_us = getenv("VRAM_US") ? atof(getenv("VRAM_US")) : 207;
        const int ntok = getenv("TOKENS") ? atoi(getenv("TOKENS")) : 100;
        const int nv = (int)strlen(pat);
        const size_t keep_stream = getenv("KEEP_STREAM") ? (size_t)atol(getenv("KEEP_STREAM")) : 0;
        if (keep_stream) {
            g_ka_bytes = 256u << 20;
            g_ka = (uint8_t*)aligned_alloc(64, g_ka_bytes);
            memset(g_ka, 1, g_ka_bytes);
            g_keep_gbps = getenv("KEEP_GBPS") ? atof(getenv("KEEP_GBPS")) : 0;
            printf("keep-alive: streaming %zu B between clock checks during GPU-only time, rate cap %.1f GB/s\n", keep_stream, g_keep_gbps);
        }
        double t_tok0 = 0, cpu_tot = 0, cpu_max = 0;
        int ncpu = 0;
        uint32_t seq = 100000;
        for (int t = 0; t < ntok + 3; ++t) {
            if (t == 3) { t_tok0 = secs(); cpu_tot = 0; ncpu = 0; cpu_max = 0; }
            for (int v = 0; v < nv; ++v) {
                gpu_time(secs() + (attn_us + (pat[v] == 'V' ? vram_us : 0)) * 1e-6, keep_stream);
                if (pat[v] != 'H') continue;
                for (int k = 0; k < TOPK; ++k) { sh->ids[k] = (t * 37 + v * 11 + k * 13) % 128; sh->wts[k] = 0.125f; }
                sh->layer = v % NL;
                double t0 = secs();
                atomic_thread_fence(memory_order_release);
                *(volatile uint32_t*)&sh->req_seq = ++seq;
                while (*(volatile uint32_t*)&sh->resp_seq != seq) _mm_pause();
                const double dt = secs() - t0;
                cpu_tot += dt; ncpu++; if (dt > cpu_max) cpu_max = dt;
            }
        }
        const double tok = (secs() - t_tok0) / ntok;
        if (g_ka) { volatile int sink = _mm512_reduce_add_epi32(g_ka_acc); (void)sink; }
        printf("pattern %s: token %.2f ms (%.2f tok/s) | CPU visit avg %.0f us max %.0f us over %d\n", pat, tok * 1e3, 1.0 / tok,
               cpu_tot / ncpu * 1e6, cpu_max * 1e6, ncpu);
    }
    if (flag) remove(flag);
    if (ka) { free(ka); if (ka_sum == 0xdeadbeef) printf("!\n"); }
    double avg = tot / (iters - 10);
    const double bytes = TOPK * (2.0 * INTER * (HIDDEN / 2 + HIDDEN / 16) + HIDDEN * (INTER / 2 + INTER / 16));
    printf("visit: avg %.3f ms, best %.3f ms -> %.1f GB/s avg (%.2f MB/visit)\n", avg * 1e3, best * 1e3, bytes / avg / 1e9, bytes / 1e6);
    cpu_moe_stop();
    return 0;
}
#endif
