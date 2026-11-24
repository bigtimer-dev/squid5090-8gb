// GPU side of the CPU expert path (see cpu/cpu_moe.c): for a B = 1 decode visit whose expert banks live in host
// RAM, hand the MoE input, top-8 ids and routing weights to the CPU through a mapped pinned buffer, wait for its
// [HIDDEN] f32 result and write it where moe_down_tc_kernel would have. One block; it runs inside the decode CUDA
// graph, so the request sequence number comes from a device counter that advances on every execution.
#include "../cpu/cpu_moe.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr int kHidden = 2304;
constexpr int kThreads = 256;
constexpr uint64_t kTimeoutNs = 10ull * 1000 * 1000 * 1000;

__device__ __forceinline__ uint64_t global_timer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ uint32_t ld_acquire_sys_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_release_sys_u32(uint32_t* p, uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__global__ void __launch_bounds__(kThreads) cpu_moe_exchange_kernel(
    const __half* __restrict__ x_h, const int32_t* __restrict__ ids, const float* __restrict__ wts,
    CpuMoeShared* sh, uint32_t* seq_ctr, int layer, float* __restrict__ out, int32_t* err) {
    __shared__ uint32_t s_seq;
    const int tid = threadIdx.x;

    // Request payload: x (fp16, 4.6 KB) + ids + weights.
    const uint4* xs = reinterpret_cast<const uint4*>(x_h);
    uint4* xd = reinterpret_cast<uint4*>(sh->x);
    for (int i = tid; i < kHidden / 8; i += kThreads) xd[i] = xs[i];
    if (tid < 8) {
        sh->ids[tid] = ids[tid];
        sh->wts[tid] = wts[tid];
    }
    if (tid == 0) {
        sh->layer = static_cast<uint32_t>(layer);
        sh->mode = CPU_MOE_MODE_FULL;
        sh->slot_begin = 0;
        sh->k_split = 0;
        sh->slot_mask = 0;
        const uint32_t s = *seq_ctr + 1;
        *seq_ctr = s;
        s_seq = s;
    }
    __threadfence_system();
    __syncthreads();

    if (tid == 0) {
        const uint32_t s = s_seq;
        __threadfence_system();
        st_release_sys_u32(&sh->req_seq, s);
        const uint64_t t0 = global_timer_ns();
        while (ld_acquire_sys_u32(&sh->resp_seq) != s) {
            if (global_timer_ns() - t0 > kTimeoutNs) {
                atomicExch(err, 1);
                break;
            }
        }
    }
    __syncthreads();

    // Response: y (f32, 9.2 KB). ld.cv: never reuse a cached copy of system memory.
    const float4* ys = reinterpret_cast<const float4*>(sh->y);
    float4* od = reinterpret_cast<float4*>(out);
    for (int i = tid; i < kHidden / 4; i += kThreads) od[i] = __ldcv(ys + i);
}

// GPU + CPU split of one visit (SQUID_CPU_MOE_GPU_SLOTS=n): post the request for slots [n, 8) and return at once;
// the GPU then runs the zero-copy expert kernels for slots [0, n) into `out` while the CPU works, and
// cpu_moe_wait_add_kernel waits for the CPU and adds its partial sum.
__global__ void __launch_bounds__(kThreads) cpu_moe_post_kernel(
    const __half* __restrict__ x_h, const int32_t* __restrict__ ids, const float* __restrict__ wts,
    CpuMoeShared* sh, uint32_t* seq_ctr, int layer, int slot_begin, int k_split) {
    __shared__ uint32_t s_seq;
    const int tid = threadIdx.x;
    const uint4* xs = reinterpret_cast<const uint4*>(x_h);
    uint4* xd = reinterpret_cast<uint4*>(sh->x);
    for (int i = tid; i < kHidden / 8; i += kThreads) xd[i] = xs[i];
    if (tid < 8) {
        sh->ids[tid] = ids[tid];
        sh->wts[tid] = wts[tid];
    }
    if (tid == 0) {
        sh->layer = static_cast<uint32_t>(layer);
        sh->mode = CPU_MOE_MODE_FULL;
        sh->slot_begin = static_cast<uint32_t>(slot_begin);
        sh->k_split = static_cast<uint32_t>(k_split);
        sh->slot_mask = 0;
        const uint32_t s = *seq_ctr + 1;
        *seq_ctr = s;
        s_seq = s;
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        st_release_sys_u32(&sh->req_seq, s_seq);
    }
}

// Decode expert cache (SQUID_DECODE_CACHE): cmap_l[e] = VRAM cache slot of this layer's expert e, or -1. The slots
// whose expert is cached go to the GPU list glist ([0] = n, [1 + j] = cache slot, [9 + j] = top-k slot, ascending);
// the CPU gets the others as a slot mask. glist[17] = number of CPU slots (0: every slot is cached, nothing posted).
__global__ void __launch_bounds__(kThreads) cpu_moe_post_cache_kernel(
    const __half* __restrict__ x_h, const int32_t* __restrict__ ids, const float* __restrict__ wts,
    CpuMoeShared* sh, uint32_t* seq_ctr, int layer, const int32_t* __restrict__ cmap_l, int32_t* glist,
    uint32_t* hist_l, int pcie_h) {
    constexpr int kHostTag = 1 << 16;  // = MOE_HOST_TAG (kernels_gemv.cu)
    __shared__ uint32_t s_cpu_mask;
    const int tid = threadIdx.x;
    if (tid < 32) {
        const int e = (tid < 8) ? ids[tid] : 0;
        int c = (tid < 8) ? cmap_l[e] : -1;
        uint32_t gmask = __ballot_sync(0xffffffffu, c >= 0) & 0xffu;
        // At most pcie_h slots cached: the GPU also streams the first uncached expert from the host bank over PCIe
        // (as the split path's GPU slot), so the CPU's DRAM share shrinks by one expert.
        if (__popc(gmask) <= pcie_h && gmask != 0xffu) {
            const int s0 = __ffs(~gmask & 0xffu) - 1;
            if (tid == s0) c = kHostTag | e;
            gmask |= 1u << s0;
        }
        if (c >= 0) {
            const int j = __popc(gmask & ((1u << tid) - 1u));
            glist[1 + j] = c;
            glist[9 + j] = tid;
        }
        if (tid < 8 && hist_l) atomicAdd(hist_l + e, 1u);
        if (tid == 0) {
            const uint32_t cpu_mask = ~gmask & 0xffu;
            glist[0] = __popc(gmask);
            glist[17] = __popc(cpu_mask);
            s_cpu_mask = cpu_mask;
        }
    }
    __syncthreads();
    const uint32_t cpu_mask = s_cpu_mask;
    if (!cpu_mask) return;
    const uint4* xs = reinterpret_cast<const uint4*>(x_h);
    uint4* xd = reinterpret_cast<uint4*>(sh->x);
    for (int i = tid; i < kHidden / 8; i += kThreads) xd[i] = xs[i];
    if (tid < 8) {
        sh->ids[tid] = ids[tid];
        sh->wts[tid] = wts[tid];
    }
    __shared__ uint32_t s_seq;
    if (tid == 0) {
        sh->layer = static_cast<uint32_t>(layer);
        sh->mode = CPU_MOE_MODE_FULL;
        sh->slot_begin = 0;
        sh->k_split = 0;
        sh->slot_mask = cpu_mask;
        const uint32_t s = *seq_ctr + 1;
        *seq_ctr = s;
        s_seq = s;
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        st_release_sys_u32(&sh->req_seq, s_seq);
    }
}

__global__ void __launch_bounds__(kThreads) cpu_moe_wait_add_kernel(
    CpuMoeShared* sh, const uint32_t* seq_ctr, float* __restrict__ out, int32_t* err, const int32_t* posted) {
    if (posted && *posted == 0) return;  // decode expert cache: every slot was cached, no CPU request
    if (threadIdx.x == 0) {
        const uint32_t s = *seq_ctr;
        const uint64_t t0 = global_timer_ns();
        while (ld_acquire_sys_u32(&sh->resp_seq) != s) {
            if (global_timer_ns() - t0 > kTimeoutNs) {
                atomicExch(err, 1);
                break;
            }
        }
    }
    __syncthreads();
    const float4* ys = reinterpret_cast<const float4*>(sh->y);
    float4* od = reinterpret_cast<float4*>(out);
    for (int i = threadIdx.x; i < kHidden / 4; i += kThreads) {
        const float4 c = __ldcv(ys + i);
        float4 g = od[i];
        g.x += c.x; g.y += c.y; g.z += c.z; g.w += c.w;
        od[i] = g;
    }
}

// SQUID_CPU_MOE_L2PF=1|2: while the CPU computes its part of a split visit, the GPU streams one slot over PCIe and
// then spins in cpu_moe_wait_add_kernel; its VRAM bandwidth is idle. Read the next visit's dense weights (QKVRG + o_proj
// NVFP4, ~15 MB) once so they sit in the 32 MB L2 when their kernels run. Mode 1: plain loads (normal L2 priority);
// mode 2: L2::evict_last cache policy. No arithmetic changes.
struct L2Ranges {
    const uint8_t* p[8];
    uint32_t bytes[8];
    int n;
};

template <int MODE>
__global__ void __launch_bounds__(256) l2_prefetch_kernel(L2Ranges r, uint32_t* sink, const int32_t* gate) {
    // decode expert cache: gate = CPU slots of this visit. With 0 there is nothing to overlap with; with 1 the CPU
    // finishes about when the GPU's 7 cached slots do, so the prefetch would extend the visit.
    if (gate && *gate < 2) return;
    uint64_t pol = 0;
    if constexpr (MODE == 2) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x, nth = gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (int k = 0; k < r.n; ++k) {
        const uint4* base = reinterpret_cast<const uint4*>(r.p[k]);
        const uint32_t n16 = r.bytes[k] / 16;
        #pragma unroll 4
        for (uint32_t i = tid; i < n16; i += nth) {
            uint4 v;
            if constexpr (MODE == 2)
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                             : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(base + i), "l"(pol));
            else
                asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(base + i));
            acc ^= v.x;
        }
    }
    if (acc == 0x9e3779b9u && sink) *sink = acc;  // practically never; keeps the loads observable
}

// SQUID_ROUTE_TRACE: append this visit's top-8 expert ids to a mapped host trace [token][visit][8]. The visit
// with is_last != 0 advances the token counter.
__global__ void route_trace_kernel(const int32_t* __restrict__ ids, int32_t* trace, uint32_t* tok_ctr, int visit,
                                   int visits, int n_max, int is_last) {
    const uint32_t row = *tok_ctr;
    if (row < static_cast<uint32_t>(n_max) && threadIdx.x < 8)
        trace[(static_cast<size_t>(row) * visits + visit) * 8 + threadIdx.x] = ids[threadIdx.x];
    __syncthreads();
    if (is_last && threadIdx.x == 0) *tok_ctr = row + 1;
}

// SQUID_BLOCKING_SYNC=1: the host thread sleeps in cudaStreamSynchronize instead of spinning on a core (one
// wake-up per decoded token). Must be set before the CUDA context exists, so it runs as a static constructor.
__attribute__((constructor)) void squid_sync_mode_init() {
    const char* v = getenv("SQUID_BLOCKING_SYNC");
    if (v && v[0] == '1') cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
}

}  // namespace

extern "C" void squid_launch_cpu_moe_exchange(
    const void* x_h, const int32_t* ids, const float* wts, void* shared_dev, uint32_t* seq_ctr, int32_t layer,
    float* out, int32_t* err, void* stream) {
    cpu_moe_exchange_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __half*>(x_h), ids, wts, static_cast<CpuMoeShared*>(shared_dev), seq_ctr, layer, out, err);
}

extern "C" void squid_launch_route_trace(const int32_t* ids, void* trace_dev, uint32_t* tok_ctr, int32_t visit,
                                         int32_t visits, int32_t n_max, int32_t is_last, void* stream) {
    route_trace_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        ids, static_cast<int32_t*>(trace_dev), tok_ctr, visit, visits, n_max, is_last);
}

extern "C" void squid_launch_cpu_moe_post(
    const void* x_h, const int32_t* ids, const float* wts, void* shared_dev, uint32_t* seq_ctr, int32_t layer,
    int32_t slot_begin, int32_t k_split, void* stream) {
    cpu_moe_post_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __half*>(x_h), ids, wts, static_cast<CpuMoeShared*>(shared_dev), seq_ctr, layer, slot_begin,
        k_split);
}

extern "C" void squid_launch_l2_prefetch_if(const void* const* ptrs, const uint32_t* bytes, int32_t n, int32_t mode,
                                            const int32_t* gate, void* stream) {
    L2Ranges r{};
    r.n = n < 8 ? n : 8;
    for (int i = 0; i < r.n; ++i) {
        r.p[i] = static_cast<const uint8_t*>(ptrs[i]);
        r.bytes[i] = bytes[i];
    }
    if (mode == 2) l2_prefetch_kernel<2><<<96, 256, 0, static_cast<cudaStream_t>(stream)>>>(r, nullptr, gate);
    else l2_prefetch_kernel<1><<<96, 256, 0, static_cast<cudaStream_t>(stream)>>>(r, nullptr, gate);
}

extern "C" void squid_launch_l2_prefetch(const void* const* ptrs, const uint32_t* bytes, int32_t n, int32_t mode,
                                         void* stream) {
    squid_launch_l2_prefetch_if(ptrs, bytes, n, mode, nullptr, stream);
}

extern "C" void squid_launch_cpu_moe_wait_add(void* shared_dev, const uint32_t* seq_ctr, float* out, int32_t* err,
                                              void* stream) {
    cpu_moe_wait_add_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<CpuMoeShared*>(shared_dev), seq_ctr, out, err, nullptr);
}

extern "C" void squid_launch_cpu_moe_post_cache(
    const void* x_h, const int32_t* ids, const float* wts, void* shared_dev, uint32_t* seq_ctr, int32_t layer,
    const int32_t* cmap_l, int32_t* glist, uint32_t* hist_l, int32_t pcie_h, void* stream) {
    cpu_moe_post_cache_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __half*>(x_h), ids, wts, static_cast<CpuMoeShared*>(shared_dev), seq_ctr, layer, cmap_l,
        glist, hist_l, pcie_h);
}

extern "C" void squid_launch_cpu_moe_wait_add_if(void* shared_dev, const uint32_t* seq_ctr, float* out, int32_t* err,
                                                 const int32_t* posted, void* stream) {
    cpu_moe_wait_add_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<CpuMoeShared*>(shared_dev), seq_ctr, out, err, posted);
}
