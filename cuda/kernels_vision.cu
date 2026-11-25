// Inkling 2 vision encoder: a 23-layer ViT (1024 wide, 16 heads of 64) over 20x20 RGB patches with a CLS token,
// a learned 32x32 position grid (bilinearly resized to the image's patch grid) plus 2D axial RoPE, then a 2x2 patch
// merge and a GELU projector into the 2304-wide text embedding space. Follows llama.cpp's mtmd inkling2v graph.
//
// It runs only when a request carries an image, inside the caller's scratch (the engine's prefill buffers). The
// weights stay in pinned host RAM and stream over PCIe one layer ahead of the compute (two slots, a copy stream),
// so the encoder holds no VRAM between images. GEMMs are FP16 tensor-core with FP32 accumulation; the residual
// stream, LayerNorm, RoPE, softmax and attention accumulation are FP32.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>

namespace squid {
void* prefill_cublas_handle(cudaStream_t stream);
}

namespace {

constexpr int VD = 1024;
constexpr int VHEADS = 16;
constexpr int VHD = 64;
constexpr int VFF = 4096;
constexpr int VLAYERS = 23;
constexpr int VPATCH_IN = 20 * 20 * 3;
constexpr int VGRID = 32;
constexpr int VMERGED = 4 * VD;
constexpr int VOUT = 2304;
constexpr float VLN_EPS = 1e-5f;

constexpr uint64_t align256(uint64_t x) { return (x + 255) & ~uint64_t(255); }

// Host block layouts (bytes). Must match src/vision.zig.
constexpr uint64_t L_QKV_W = 0;
constexpr uint64_t L_O_W = L_QKV_W + uint64_t(3 * VD) * VD * 2;
constexpr uint64_t L_FC_W = L_O_W + uint64_t(VD) * VD * 2;
constexpr uint64_t L_PR_W = L_FC_W + uint64_t(VFF) * VD * 2;
constexpr uint64_t L_VEC = L_PR_W + uint64_t(VD) * VFF * 2;  // fp32 vectors follow
// fp32 vector offsets (in floats) inside the L_VEC section
constexpr int LV_LN1_W = 0, LV_LN1_B = VD, LV_LN2_W = 2 * VD, LV_LN2_B = 3 * VD, LV_LS1 = 4 * VD, LV_LS2 = 5 * VD;
constexpr int LV_QKV_B = 6 * VD, LV_O_B = 9 * VD, LV_FC_B = 10 * VD, LV_PR_B = 10 * VD + VFF;
constexpr int LV_COUNT = 11 * VD + VFF;
constexpr uint64_t LAYER_BYTES = L_VEC + uint64_t(LV_COUNT) * 4;

constexpr uint64_t G_PATCH_W = 0;                                     // fp16 [1024][1200]
constexpr uint64_t G_POS = G_PATCH_W + uint64_t(VD) * VPATCH_IN * 2;  // fp32 [1024 positions][1024]
constexpr uint64_t G_CLS = G_POS + uint64_t(VGRID * VGRID) * VD * 4;  // fp32 cls, cls_pos, lnpre_w, lnpre_b
constexpr uint64_t GLOB_BYTES = G_CLS + uint64_t(4) * VD * 4;

constexpr uint64_t P_IN_W = 0;                                    // fp16 [4096][4096]
constexpr uint64_t P_OUT_W = P_IN_W + uint64_t(VMERGED) * VMERGED * 2;  // fp16 [2304][4096]
constexpr uint64_t PROJ_BYTES = P_OUT_W + uint64_t(VOUT) * VMERGED * 2;

constexpr uint64_t SLOT_BYTES = align256(LAYER_BYTES > PROJ_BYTES / 2 ? LAYER_BYTES : (PROJ_BYTES + 1) / 2);
constexpr uint64_t ROPE_FLOATS_MAX = 4 * 16 * 1024;  // cos/sin for w and h positions, up to 1023 patches a side

struct Scratch {
    uint8_t* slot[2];
    uint8_t* glob;
    float* rope;
    float* x;       // [n, 1024] residual stream
    __half* h16;    // [n, 1024] LayerNorm out (GEMM input); later the merged tokens [T, 4096]
    __half* a16;    // [n, 1024] attention out
    float* qkv;     // [n, 3072]; later the projector output [T, 2304]
    float* y;       // [n, 1024] projection out
    float* fc32;    // [n, 4096]; first holds the fp16 patches [P, 1200]
    __half* fc16;   // [n, 4096]
    uint64_t total;
};

Scratch layout(uint8_t* base, int n) {
    Scratch s{};
    uint64_t off = 0;
    auto take = [&](uint64_t bytes) { uint8_t* p = base ? base + off : nullptr; off += align256(bytes); return p; };
    s.slot[0] = take(SLOT_BYTES);
    s.slot[1] = take(SLOT_BYTES);
    s.glob = take(GLOB_BYTES);
    s.rope = reinterpret_cast<float*>(take(ROPE_FLOATS_MAX * 4));
    s.x = reinterpret_cast<float*>(take(uint64_t(n) * VD * 4));
    s.h16 = reinterpret_cast<__half*>(take(uint64_t(n) * VD * 2));
    s.a16 = reinterpret_cast<__half*>(take(uint64_t(n) * VD * 2));
    s.qkv = reinterpret_cast<float*>(take(uint64_t(n) * 3 * VD * 4));
    s.y = reinterpret_cast<float*>(take(uint64_t(n) * VD * 4));
    s.fc32 = reinterpret_cast<float*>(take(uint64_t(n) * VFF * 4));
    s.fc16 = reinterpret_cast<__half*>(take(uint64_t(n) * VFF * 2));
    s.total = off;
    return s;
}

__device__ __forceinline__ float block_sum_256(float v, float* red) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float t = (l < 8) ? red[l] : 0.0f;
    #pragma unroll
    for (int o = 4; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    return __shfl_sync(0xffffffffu, t, 0);
}

// One row per block, 256 threads x 4 values. In place is fine (each thread rewrites only what it read).
template <typename OutT>
__global__ void __launch_bounds__(256) v_layernorm_kernel(const float* x, const float* __restrict__ w, const float* __restrict__ b, OutT* out) {
    __shared__ float red[8];
    const size_t r = blockIdx.x;
    const int t = threadIdx.x;
    const float4 v = reinterpret_cast<const float4*>(x + r * VD)[t];
    const float mean = block_sum_256(v.x + v.y + v.z + v.w, red) * (1.0f / VD);
    const float d0 = v.x - mean, d1 = v.y - mean, d2 = v.z - mean, d3 = v.w - mean;
    const float var = block_sum_256(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3, red) * (1.0f / VD);
    const float rs = rsqrtf(var + VLN_EPS);
    const float4 ww = reinterpret_cast<const float4*>(w)[t];
    const float4 bb = reinterpret_cast<const float4*>(b)[t];
    const float o0 = d0 * rs * ww.x + bb.x, o1 = d1 * rs * ww.y + bb.y, o2 = d2 * rs * ww.z + bb.z, o3 = d3 * rs * ww.w + bb.w;
    if constexpr (sizeof(OutT) == 4) {
        reinterpret_cast<float4*>(out + r * VD)[t] = make_float4(o0, o1, o2, o3);
    } else {
        __half2* o = reinterpret_cast<__half2*>(out + r * VD) + 2 * t;
        o[0] = __floats2half2_rn(o0, o1);
        o[1] = __floats2half2_rn(o2, o3);
    }
}

// x rows 1..P hold the patch embeddings: add the position grid (bilinear, ggml semantics: half-pixel centres,
// clamped corners, no antialias; untouched when the patch grid is 32x32). Row 0 becomes CLS + its position.
__global__ void __launch_bounds__(256) v_embed_kernel(float* x, const float* __restrict__ pos, const float* __restrict__ cls,
                                                      const float* __restrict__ cls_pos, int gw, int gh) {
    const int r = blockIdx.x;
    float* xr = x + size_t(r) * VD;
    if (r == 0) {
        for (int d = threadIdx.x; d < VD; d += 256) xr[d] = cls[d] + cls_pos[d];
        return;
    }
    const int p = r - 1, px = p % gw, py = p / gw;
    if (gw == VGRID && gh == VGRID) {
        const float* pr = pos + size_t(py * VGRID + px) * VD;
        for (int d = threadIdx.x; d < VD; d += 256) xr[d] += pr[d];
        return;
    }
    const float fx = (px + 0.5f) * (float(VGRID) / float(gw)) - 0.5f;
    const float fy = (py + 0.5f) * (float(VGRID) / float(gh)) - 0.5f;
    int x0 = int(floorf(fx)), y0 = int(floorf(fy));
    const float dx = fminf(fmaxf(fx - floorf(fx), 0.0f), 1.0f);
    const float dy = fminf(fmaxf(fy - floorf(fy), 0.0f), 1.0f);
    int x1 = min(max(x0 + 1, 0), VGRID - 1), y1 = min(max(y0 + 1, 0), VGRID - 1);
    x0 = min(max(x0, 0), VGRID - 1);
    y0 = min(max(y0, 0), VGRID - 1);
    const float* a00 = pos + size_t(y0 * VGRID + x0) * VD;
    const float* a01 = pos + size_t(y0 * VGRID + x1) * VD;
    const float* a10 = pos + size_t(y1 * VGRID + x0) * VD;
    const float* a11 = pos + size_t(y1 * VGRID + x1) * VD;
    const float w00 = (1 - dx) * (1 - dy), w01 = dx * (1 - dy), w10 = (1 - dx) * dy, w11 = dx * dy;
    for (int d = threadIdx.x; d < VD; d += 256) xr[d] += a00[d] * w00 + a01[d] * w01 + a10[d] * w10 + a11[d] * w11;
}

// qkv [n, 3072] += bias; q and k get 2D RoPE per head: dims [0,32) by the column position (CLS 0, patches x+1),
// dims [32,64) by the row position, adjacent pairs, 16 frequencies (tables precomputed in double on the host).
__global__ void __launch_bounds__(256) v_qkv_rope_kernel(float* qkv, const float* __restrict__ bias, const float* __restrict__ rope, int gw, int gh) {
    const int r = blockIdx.x;
    const int pw = (r == 0) ? 0 : ((r - 1) % gw) + 1;
    const int ph = (r == 0) ? 0 : ((r - 1) / gw) + 1;
    const float* cw = rope;
    const float* sw = cw + (gw + 1) * 16;
    const float* ch = sw + (gw + 1) * 16;
    const float* sh = ch + (gh + 1) * 16;
    float* row = qkv + size_t(r) * 3 * VD;
    // 1024 pairs over q and k (512 each), 4 per thread
    for (int pi = threadIdx.x; pi < 1024; pi += 256) {
        const int base = (pi < 512) ? 0 : VD;
        const int pp = pi & 511;
        const int head = pp >> 5, j = pp & 31;
        const int dim = head * VHD + (j < 16 ? 2 * j : 32 + 2 * (j - 16));
        const int fi = j & 15;
        const float c = (j < 16) ? cw[pw * 16 + fi] : ch[ph * 16 + fi];
        const float s = (j < 16) ? sw[pw * 16 + fi] : sh[ph * 16 + fi];
        const float x0 = row[base + dim] + bias[base + dim];
        const float x1 = row[base + dim + 1] + bias[base + dim + 1];
        row[base + dim] = x0 * c - x1 * s;
        row[base + dim + 1] = x0 * s + x1 * c;
    }
    for (int d = threadIdx.x; d < VD; d += 256) row[2 * VD + d] += bias[2 * VD + d];
}

// Full (bidirectional) attention, FP32, online softmax. One query per thread, 64 queries x one head per block;
// K/V tiles of 32 keys staged in shared memory and read as broadcasts.
constexpr int ATT_BQ = 64;
constexpr int ATT_BK = 32;
__global__ void __launch_bounds__(ATT_BQ) v_attn_kernel(const float* __restrict__ qkv, __half* __restrict__ out, int n) {
    __shared__ __align__(16) float Ks[ATT_BK][VHD];
    __shared__ __align__(16) float Vs[ATT_BK][VHD];
    const int h = blockIdx.y;
    const int t = threadIdx.x;
    const int qi = blockIdx.x * ATT_BQ + t;
    const bool valid = qi < n;
    float q[VHD], acc[VHD];
    if (valid) {
        const float4* qp = reinterpret_cast<const float4*>(qkv + size_t(qi) * 3 * VD + h * VHD);
        #pragma unroll
        for (int i = 0; i < VHD / 4; ++i) {
            const float4 v = qp[i];
            q[4 * i] = v.x * 0.125f;
            q[4 * i + 1] = v.y * 0.125f;
            q[4 * i + 2] = v.z * 0.125f;
            q[4 * i + 3] = v.w * 0.125f;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < VHD; ++i) q[i] = 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < VHD; ++i) acc[i] = 0.0f;
    float m = -1e30f, l = 0.0f;
    for (int k0 = 0; k0 < n; k0 += ATT_BK) {
        __syncthreads();
        for (int idx = t; idx < ATT_BK * (VHD / 4); idx += ATT_BQ) {
            const int rr = idx >> 4, c = idx & 15;
            const int kr = k0 + rr;
            float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), vv = kv;
            if (kr < n) {
                const float* base = qkv + size_t(kr) * 3 * VD + h * VHD + c * 4;
                kv = *reinterpret_cast<const float4*>(base + VD);
                vv = *reinterpret_cast<const float4*>(base + 2 * VD);
            }
            *reinterpret_cast<float4*>(&Ks[rr][c * 4]) = kv;
            *reinterpret_cast<float4*>(&Vs[rr][c * 4]) = vv;
        }
        __syncthreads();
        const int kmax = min(ATT_BK, n - k0);
        #pragma unroll 1
        for (int j0 = 0; j0 < kmax; j0 += 16) {
            float s[16];
            float mx = m;
            #pragma unroll
            for (int jj = 0; jj < 16; ++jj) {
                const float4* kr = reinterpret_cast<const float4*>(Ks[j0 + jj]);
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                #pragma unroll
                for (int i = 0; i < VHD / 4; ++i) {
                    const float4 kv = kr[i];
                    a0 = fmaf(q[4 * i], kv.x, a0);
                    a1 = fmaf(q[4 * i + 1], kv.y, a1);
                    a2 = fmaf(q[4 * i + 2], kv.z, a2);
                    a3 = fmaf(q[4 * i + 3], kv.w, a3);
                }
                s[jj] = (j0 + jj < kmax) ? ((a0 + a1) + (a2 + a3)) : -1e30f;
                mx = fmaxf(mx, s[jj]);
            }
            const float corr = __expf(m - mx);
            l *= corr;
            #pragma unroll
            for (int i = 0; i < VHD; ++i) acc[i] *= corr;
            #pragma unroll
            for (int jj = 0; jj < 16; ++jj) {
                const float p = __expf(s[jj] - mx);
                l += p;
                const float4* vr = reinterpret_cast<const float4*>(Vs[j0 + jj]);
                #pragma unroll
                for (int i = 0; i < VHD / 4; ++i) {
                    const float4 vv = vr[i];
                    acc[4 * i] = fmaf(p, vv.x, acc[4 * i]);
                    acc[4 * i + 1] = fmaf(p, vv.y, acc[4 * i + 1]);
                    acc[4 * i + 2] = fmaf(p, vv.z, acc[4 * i + 2]);
                    acc[4 * i + 3] = fmaf(p, vv.w, acc[4 * i + 3]);
                }
            }
            m = mx;
        }
    }
    if (valid) {
        const float inv = 1.0f / l;
        __half2* o = reinterpret_cast<__half2*>(out + size_t(qi) * VD + h * VHD);
        #pragma unroll
        for (int i = 0; i < VHD / 2; ++i) o[i] = __floats2half2_rn(acc[2 * i] * inv, acc[2 * i + 1] * inv);
    }
}

// x += gamma * (y + bias)
__global__ void v_resid_kernel(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ bias,
                               const float* __restrict__ gamma, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int d = int(i & (VD - 1));
    x[i] += gamma[d] * (y[i] + bias[d]);
}

// out = GELU_erf(in + bias) as fp16 (bias optional)
__global__ void v_gelu_kernel(const float* __restrict__ in, const float* __restrict__ bias, __half* __restrict__ out, size_t count, int cols) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float v = in[i] + (bias ? bias[i % cols] : 0.0f);
    out[i] = __float2half_rn(0.5f * v * (1.0f + erff(v * 0.70710678118654752f)));
}

// Drop CLS and merge 2x2 patches, row-major over the merged grid: [TL, TR, BL, BR] concatenated (fp16).
__global__ void __launch_bounds__(256) v_merge_kernel(const float* __restrict__ x, __half* __restrict__ out, int gw) {
    const int tok = blockIdx.x;
    const int mw = gw / 2;
    const int i = tok / mw, j = tok % mw;
    __half* o = out + size_t(tok) * VMERGED;
    for (int e = threadIdx.x; e < VMERGED; e += 256) {
        const int q = e >> 10, d = e & (VD - 1);
        const int src = 1 + (2 * i + (q >> 1)) * gw + (2 * j + (q & 1));
        o[e] = __float2half_rn(x[size_t(src) * VD + d]);
    }
}

void vgemm(cublasHandle_t h, const void* W, const __half* X, float* Y, int out, int n, int in, int ldy) {
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, out, n, in, &alpha, W, CUDA_R_16F, in, X, CUDA_R_16F, in,
                                           &beta, Y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) fprintf(stderr, "[vision] cublasGemmEx failed: %d (out=%d n=%d in=%d)\n", int(st), out, n, in);
}

inline unsigned blocks_for(size_t count, int bs) { return unsigned((count + bs - 1) / bs); }

cudaStream_t g_copy = nullptr;
cudaEvent_t g_start = nullptr, g_up[2] = {nullptr, nullptr}, g_free[2] = {nullptr, nullptr};

}  // namespace

extern "C" {

struct SquidVisionArgs {
    const void* glob;     // pinned host, GLOB_BYTES
    const void* layers;   // pinned host, 23 x LAYER_BYTES
    const void* proj;     // pinned host, PROJ_BYTES
    const void* patches;  // pinned host fp16 [gw*gh, 1200]
    const float* rope;    // pinned host fp32 [cw | sw | ch | sh], (gw+1)*16 and (gh+1)*16 each
    float* out;           // pinned host fp32 [tokens, 2304]
    float* dbg;           // optional pinned host fp32: lnpre [n,1024] | layer 0 [n,1024] | encoder out [n,1024]
    void* scratch;
    uint64_t scratch_bytes;
    int32_t gw, gh;       // patch grid, both even
};

uint64_t squid_vision_layer_bytes(void) { return LAYER_BYTES; }
uint64_t squid_vision_glob_bytes(void) { return GLOB_BYTES; }
uint64_t squid_vision_proj_bytes(void) { return PROJ_BYTES; }
uint64_t squid_vision_scratch_bytes(int32_t n_rows) { return layout(nullptr, n_rows).total; }

int32_t squid_vision_forward(const SquidVisionArgs* a, void* stream_v) {
    cudaStream_t st = static_cast<cudaStream_t>(stream_v);
    const int gw = a->gw, gh = a->gh;
    const int P = gw * gh, n = P + 1, T = P / 4;
    if (gw < 2 || gh < 2 || (gw & 1) || (gh & 1) || gw > 1022 || gh > 1022) return -1;
    const Scratch s = layout(static_cast<uint8_t*>(a->scratch), n);
    if (s.total > a->scratch_bytes) return -2;
    if (!g_copy) {
        if (cudaStreamCreateWithFlags(&g_copy, cudaStreamNonBlocking) != cudaSuccess) return -3;
        cudaEventCreateWithFlags(&g_start, cudaEventDisableTiming);
        for (int i = 0; i < 2; ++i) {
            cudaEventCreateWithFlags(&g_up[i], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&g_free[i], cudaEventDisableTiming);
        }
    }
    cublasHandle_t cb = static_cast<cublasHandle_t>(squid::prefill_cublas_handle(st));
    const uint8_t* layers = static_cast<const uint8_t*>(a->layers);

    // The copy stream may only touch the scratch after everything already queued on `st` (prefill, cache uploads).
    cudaEventRecord(g_start, st);
    cudaStreamWaitEvent(g_copy, g_start, 0);
    cudaMemcpyAsync(s.slot[0], layers, LAYER_BYTES, cudaMemcpyHostToDevice, g_copy);
    cudaEventRecord(g_up[0], g_copy);

    cudaMemcpyAsync(s.glob, a->glob, GLOB_BYTES, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(s.rope, a->rope, size_t(gw + gh + 2) * 32 * 4, cudaMemcpyHostToDevice, st);
    __half* patches16 = reinterpret_cast<__half*>(s.fc32);
    cudaMemcpyAsync(patches16, a->patches, size_t(P) * VPATCH_IN * 2, cudaMemcpyHostToDevice, st);

    const float* gpos = reinterpret_cast<const float*>(s.glob + G_POS);
    const float* gvec = reinterpret_cast<const float*>(s.glob + G_CLS);
    vgemm(cb, s.glob + G_PATCH_W, patches16, s.x + VD, VD, P, VPATCH_IN, VD);
    v_embed_kernel<<<n, 256, 0, st>>>(s.x, gpos, gvec, gvec + VD, gw, gh);
    v_layernorm_kernel<float><<<n, 256, 0, st>>>(s.x, gvec + 2 * VD, gvec + 3 * VD, s.x);
    if (a->dbg) cudaMemcpyAsync(a->dbg, s.x, size_t(n) * VD * 4, cudaMemcpyDeviceToHost, st);

    const size_t nd = size_t(n) * VD;
    for (int l = 0; l < VLAYERS; ++l) {
        const int k = l & 1;
        if (l + 1 < VLAYERS) {
            if (l >= 1) cudaStreamWaitEvent(g_copy, g_free[k ^ 1], 0);
            cudaMemcpyAsync(s.slot[k ^ 1], layers + size_t(l + 1) * LAYER_BYTES, LAYER_BYTES, cudaMemcpyHostToDevice, g_copy);
            cudaEventRecord(g_up[k ^ 1], g_copy);
        }
        cudaStreamWaitEvent(st, g_up[k], 0);
        const uint8_t* lw = s.slot[k];
        const float* vec = reinterpret_cast<const float*>(lw + L_VEC);

        v_layernorm_kernel<__half><<<n, 256, 0, st>>>(s.x, vec + LV_LN1_W, vec + LV_LN1_B, s.h16);
        vgemm(cb, lw + L_QKV_W, s.h16, s.qkv, 3 * VD, n, VD, 3 * VD);
        v_qkv_rope_kernel<<<n, 256, 0, st>>>(s.qkv, vec + LV_QKV_B, s.rope, gw, gh);
        v_attn_kernel<<<dim3((n + ATT_BQ - 1) / ATT_BQ, VHEADS), ATT_BQ, 0, st>>>(s.qkv, s.a16, n);
        vgemm(cb, lw + L_O_W, s.a16, s.y, VD, n, VD, VD);
        v_resid_kernel<<<blocks_for(nd, 256), 256, 0, st>>>(s.x, s.y, vec + LV_O_B, vec + LV_LS1, nd);

        v_layernorm_kernel<__half><<<n, 256, 0, st>>>(s.x, vec + LV_LN2_W, vec + LV_LN2_B, s.h16);
        vgemm(cb, lw + L_FC_W, s.h16, s.fc32, VFF, n, VD, VFF);
        v_gelu_kernel<<<blocks_for(size_t(n) * VFF, 256), 256, 0, st>>>(s.fc32, vec + LV_FC_B, s.fc16, size_t(n) * VFF, VFF);
        vgemm(cb, lw + L_PR_W, s.fc16, s.y, VD, n, VFF, VD);
        v_resid_kernel<<<blocks_for(nd, 256), 256, 0, st>>>(s.x, s.y, vec + LV_PR_B, vec + LV_LS2, nd);
        cudaEventRecord(g_free[k], st);
        if (l == 0 && a->dbg) cudaMemcpyAsync(a->dbg + nd, s.x, nd * 4, cudaMemcpyDeviceToHost, st);
    }
    if (a->dbg) cudaMemcpyAsync(a->dbg + 2 * nd, s.x, nd * 4, cudaMemcpyDeviceToHost, st);

    // Projector: its 50 MiB span both slots, both free once the last layer is done (stream order on `st`).
    cudaMemcpyAsync(s.slot[0], a->proj, PROJ_BYTES, cudaMemcpyHostToDevice, st);
    v_merge_kernel<<<T, 256, 0, st>>>(s.x, s.h16, gw);
    vgemm(cb, s.slot[0] + P_IN_W, s.h16, s.fc32, VMERGED, T, VMERGED, VMERGED);
    v_gelu_kernel<<<blocks_for(size_t(T) * VMERGED, 256), 256, 0, st>>>(s.fc32, nullptr, s.fc16, size_t(T) * VMERGED, VMERGED);
    vgemm(cb, s.slot[0] + P_OUT_W, s.fc16, s.qkv, VOUT, T, VMERGED, VOUT);
    cudaMemcpyAsync(a->out, s.qkv, size_t(T) * VOUT * 4, cudaMemcpyDeviceToHost, st);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[vision] CUDA error: %s\n", cudaGetErrorString(e));
        return -4;
    }
    return 0;
}

}  // extern "C"
