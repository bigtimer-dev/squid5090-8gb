// Explicit extern "C" ABI declarations matching zig/src/cuda_abi.cu bit-for-bit.
// Zero @cImport dependency — fully compatible with Zig 0.17.1 and TigerStyle fixed-width types.

const constants = @import("constants.zig");

pub const CSampleConfig = extern struct {
    inv_temp: f32 = 0.0,
    top_p: f32 = 0.95,
    top_k: i32 = 0,
    pad: u32 = 0,
    seed: u64 = 0,
};

pub const CLinearWeight = extern struct {
    kind: u8 = 0,
    rows: i32 = 0,
    cols: i32 = 0,
    data: ?*anyopaque = null,
    scales: ?[*]u8 = null,
    row_scales: ?[*]f32 = null,
    global_scale: f32 = 1.0,
};

pub const CExpertBank = extern struct {
    is_nvfp4: u8 = 0,
    n_experts: i32 = 0,
    rows: i32 = 0,
    cols: i32 = 0,
    data: ?[*]u8 = null,
    scales: ?[*]u8 = null,
    expert_scale: ?[*]f32 = null,
};

pub const CControllerWeights = extern struct {
    down_w: ?*anyopaque = null,
    forget_w: ?*anyopaque = null,
    gain_w: ?*anyopaque = null,
    down_nvfp4: CLinearWeight = .{},
    forget_nvfp4: CLinearWeight = .{},
    gain_nvfp4: CLinearWeight = .{},
};

pub const CDsparkLayerWeights = extern struct {
    is_local: u8 = 0,
    input_norm: ?*anyopaque = null,
    post_norm: ?*anyopaque = null,
    dflare_norm: ?*anyopaque = null,
    q_norm: ?*anyopaque = null,
    k_norm: ?*anyopaque = null,
    k_norm_ctx: ?*anyopaque = null,
    w_qkv: CLinearWeight = .{},
    w_o: CLinearWeight = .{},
    w_gu: CLinearWeight = .{},
    w_down: CLinearWeight = .{},
    w_kv_ctx: CLinearWeight = .{},
};

pub const CDsparkWeights = extern struct {
    loaded: u8 = 0,
    fuse_alpha: ?[*]f32 = null,
    mask_embed: ?*anyopaque = null,
    norm_w: ?*anyopaque = null,
    markov_dflare_norm: ?*anyopaque = null,
    markov_w1: ?*anyopaque = null,
    rnn_w_sp: ?*anyopaque = null,
    rnn_w_h: ?*anyopaque = null,
    rnn_w_f: ?*anyopaque = null,
    rnn_bias: ?*anyopaque = null,
    hidden_out_w: ?*anyopaque = null,
    hidden_out_b: ?*anyopaque = null,
    draft_vocab_size: i32 = 0,
    draft_vocab_ids: ?[*]i32 = null,
    unembed_draft: CLinearWeight = .{},
    fold_w: CLinearWeight = .{},
    fold_b: ?[*]f32 = null,
    layers: [constants.mtp_layer_count]CDsparkLayerWeights = .{ .{}, .{}, .{} },
};

// --- CUDA Runtime Primitives ---

pub extern "c" fn squid_cuda_malloc(ptr: *?*anyopaque, bytes: usize) i32;
pub extern "c" fn squid_cuda_free(ptr: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_malloc_host_pinned(ptr: *?*anyopaque, bytes: usize, mapped: u8) i32;
pub extern "c" fn squid_cuda_free_host(ptr: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_mem_get_info(free_bytes: *usize, total_bytes: *usize) i32;
pub extern "c" fn squid_cuda_malloc_host_thp(ptr: *?*anyopaque, bytes: usize, thp_bytes: *usize) i32;
pub extern "c" fn squid_cuda_free_host_thp(ptr: ?*anyopaque, bytes: usize) i32;
pub extern "c" fn squid_cuda_host_get_device_pointer(dev_ptr: *?*anyopaque, host_ptr: *anyopaque) i32;
pub extern "c" fn squid_cuda_memcpy_h2d(dst: *anyopaque, src: *const anyopaque, bytes: usize) i32;
pub extern "c" fn squid_cuda_memcpy_d2h(dst: *anyopaque, src: *const anyopaque, bytes: usize) i32;
pub extern "c" fn squid_cuda_memcpy_h2d_async(dst: *anyopaque, src: *const anyopaque, bytes: usize, stream: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_memcpy_d2h_async(dst: *anyopaque, src: *const anyopaque, bytes: usize, stream: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_memcpy_d2d_async(dst: *anyopaque, src: *const anyopaque, bytes: usize, stream: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_memcpy_default_async(dst: *anyopaque, src: *const anyopaque, bytes: usize, stream: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_memset_async(dst: *anyopaque, value: i32, bytes: usize, stream: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_device_sync() i32;
pub extern "c" fn squid_cuda_stream_create(out_stream: *?*anyopaque) i32;
pub extern "c" fn squid_cuda_stream_destroy(stream: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_stream_sync(stream: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_event_create(out_event: *?*anyopaque, disable_timing: u8) i32;
pub extern "c" fn squid_cuda_event_destroy(ev: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_event_record(ev: ?*anyopaque, stream: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_event_sync(ev: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_event_query(ev: ?*anyopaque) i32; // 0 done, 1 not ready, else error
pub extern "c" fn squid_cuda_event_elapsed_ms(out_ms: *f32, ev0: ?*anyopaque, ev1: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_stream_wait_event(stream: ?*anyopaque, ev: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_graph_begin(stream: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_graph_end_instantiate(stream: ?*anyopaque, out_exec: *?*anyopaque) i32;
pub extern "c" fn squid_cuda_graph_launch(exec: ?*anyopaque, stream: ?*anyopaque) i32;
pub extern "c" fn squid_cuda_graph_destroy(exec: ?*anyopaque) i32;

// --- Weight Packing Kernels ---

pub extern "c" fn squid_pack_bf16_to_nvfp4(
    src_bf16_dev: *const anyopaque,
    dst_data: *anyopaque,
    dst_scales: [*]u8,
    d_amax_scratch: *f32,
    rows: i32,
    cols: i32,
) f32;

pub extern "c" fn squid_pack_bf16_to_fp8(
    src_bf16_dev: *const anyopaque,
    dst_data: *anyopaque,
    dst_row_scales: [*]f32,
    rows: i32,
    cols: i32,
) void;

pub extern "c" fn squid_convert_mxfp8_to_nvfp4_experts(
    src_w: [*]const u8,
    src_s: [*]const u8,
    dst_w: [*]u8,
    dst_s: [*]u8,
    dst_expert_scale: [*]f32,
    n_experts: i32,
    rows: i32,
    cols: i32,
) void;

pub extern "c" fn squid_split_dspark_rnn(
    src_bf16: *const anyopaque,
    w_sp: *anyopaque,
    w_h: *anyopaque,
    w_f: *anyopaque,
) void;

pub extern "c" fn squid_compute_dspark_fold(
    unembed_bf16: *const anyopaque,
    hidden_out_w: *const anyopaque,
    hidden_out_b: *const anyopaque,
    fold_b: [*]f32,
    fold_bf16_out: *anyopaque,
    vocab_count: i32,
    hidden_cols: i32,
    markov_rank: i32,
) void;

// --- Inference Kernel Trampolines ---

pub extern "c" fn squid_launch_embed_norm(
    embed: *const anyopaque,
    embed_norm: *const anyopaque,
    tokens: [*]const i32,
    out: [*]f32,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_dir_project(
    y: [*]f32,
    dir: [*]const f32,
    n_vecs: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_rmsnorm(
    x: [*]const f32,
    weight: ?*const anyopaque,
    out: [*]f32,
    batch_size: i32,
    width: i32,
    group_size: i32,
    eps: f32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_extract_block(
    blocks: [*]const f32,
    max_blocks: i32,
    slot_idx: i32,
    dst: [*]f32,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_fused_depth_norm(
    hidden: [*]const f32,
    partial_out: ?[*]f32,
    blocks: [*]const f32,
    n_blocks: i32,
    queries: ?[*]const f32,
    n0: ?*const anyopaque,
    n1: ?*const anyopaque,
    n2: ?*const anyopaque,
    o0: ?[*]f32,
    o1: ?[*]f32,
    o2: ?[*]f32,
    oh0: ?*anyopaque,
    oh1: ?*anyopaque,
    oh2: ?*anyopaque,
    n_out: i32,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_controller_mega_nvfp4(
    xs: [*]const [*]f32,
    saved: [*]const f32,
    prior_last: [*]const f32,
    cp_queries: [*]const ?[*]const f32,
    cws_in: [*]const ?*const CControllerWeights,
    post_norms: [*]const ?*const anyopaque,
    outs: [*]const [*]f32,
    outs_h: ?[*]const ?*anyopaque,
    n_streams: i32,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_controller_merge_batched(
    xs: [*]const [*]f32,
    saved: [*]const f32,
    prior_last: [*]const f32,
    cp_queries: [*]const ?[*]const f32,
    cws_in: [*]const ?*const CControllerWeights,
    post_norms: [*]const ?*const anyopaque,
    outs: [*]const [*]f32,
    outs_h: ?[*]const ?*anyopaque,
    n_streams: i32,
    scratch_msg: [*]f32,
    scratch_cat: [*]f32,
    scratch_d: [*]f32,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_qkvrg_fused(
    w_qrgs: *const CLinearWeight,
    w_k: *const CLinearWeight,
    w_v: *const CLinearWeight,
    in_qh: *const anyopaque,
    in_kh: *const anyopaque,
    in_vh: *const anyopaque,
    qrgs_out: [*]f32,
    k_out: [*]f32,
    v_out: [*]f32,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_linear(
    w: *const CLinearWeight,
    x: [*]const f32,
    y: [*]f32,
    batch_size: i32,
    add_res: u8,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_linear_h(
    w: *const CLinearWeight,
    x_h: *const anyopaque,
    y: [*]f32,
    batch_size: i32,
    add_res: u8,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_mega_global_sconv_prep_store(
    qrgs: [*]f32,
    k_proj: [*]const f32,
    v_proj: [*]const f32,
    in_k: [*]const f32,
    in_v: [*]const f32,
    k_sw: *const anyopaque,
    v_sw: *const anyopaque,
    k_sp: *const anyopaque,
    v_sp: *const anyopaque,
    k_ring: [*]f32,
    v_ring: [*]f32,
    q_cs: *const anyopaque,
    r_cs: *const anyopaque,
    k_cache: *anyopaque,
    v_cache: *anyopaque,
    k_scale: ?*anyopaque,
    v_scale: ?*anyopaque,
    kv_fmt: u8,
    kv_cap: i32,
    batch_size: i32,
    d_pos: *const i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_mega_local_sconv_prep_store(
    qrgs: [*]f32,
    k_proj: [*]const f32,
    v_proj: [*]const f32,
    k_sw: *const anyopaque,
    v_sw: *const anyopaque,
    k_ring: [*]f32,
    v_ring: [*]f32,
    q_cs: *const anyopaque,
    r_cs: *const anyopaque,
    k_cache: *anyopaque,
    v_cache: *anyopaque,
    k_scale: ?*anyopaque,
    v_scale: ?*anyopaque,
    kv_fmt: u8,
    kv_cap: i32,
    batch_size: i32,
    d_pos: *const i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_relative_attention_gated(
    qrgs: [*]const f32,
    rel_proj: *const anyopaque,
    k_cache: *const anyopaque,
    v_cache: *const anyopaque,
    k_scale: ?*const anyopaque,
    v_scale: ?*const anyopaque,
    kv_fmt: u8,
    split_buf: ?[*]f32,
    out_gated: ?[*]f32,
    out_gated_h: ?*anyopaque,
    kv_cap: i32,
    window: i32,
    batch_size: i32,
    d_pos: *const i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_mega_attn_post_mlp_in(
    o_proj: [*]const f32,
    attn_sw: *const anyopaque,
    attn_ring: [*]f32,
    hidden: [*]const f32,
    partial_prev: [*]const f32,
    blocks: [*]f32,
    block_count_before: i32,
    is_boundary: u8,
    residual_out: [*]f32,
    mlp_query: [*]const f32,
    mlp_norm_w: ?*const anyopaque,
    mlp_out: [*]f32,
    mlp_out_h: ?*anyopaque,
    batch_size: i32,
    d_pos: *const i32,
    abl_r: ?[*]const f32,
    abl_d: ?[*]const f32,
    abl_p: ?[*]const f32,
    udir: ?[*]const f32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_mega_attn_post_mlp_in_x(
    w_o: ?*const CLinearWeight, // != null: o_proj computed here from x_h (B = 1 fused); null: already computed
    x_h: ?*const anyopaque,
    o_proj: [*]f32,
    attn_sw: *const anyopaque,
    attn_ring: [*]f32,
    hidden: [*]const f32,
    partial_prev: [*]const f32,
    blocks: [*]f32,
    block_count_before: i32,
    is_boundary: u8,
    residual_out: [*]f32,
    mlp_query: [*]const f32,
    mlp_norm_w: ?*const anyopaque,
    mlp_out: [*]f32,
    mlp_out_h: ?*anyopaque,
    batch_size: i32,
    d_pos: *const i32,
    abl_r: ?[*]const f32,
    abl_d: ?[*]const f32,
    abl_p: ?[*]const f32,
    udir: ?[*]const f32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_route_fused_b1(
    m_h: *const anyopaque,
    router_nvfp4: *const CLinearWeight,
    router_bias: [*]const f32,
    scratch_logits: [*]f32,
    out_ids: [*]i32,
    out_weights: [*]f32,
    done_ctr: [*]u32,
    cpu_sh: ?*anyopaque,
    seq_ctr: ?[*]u32,
    layer: i32,
    cmap_l: ?[*]const i32,
    glist: ?[*]i32,
    hist_l: ?[*]u32,
    pcie_h: i32,
    stream: ?*anyopaque,
) void;
pub extern "c" fn squid_launch_route_topk_nvfp4(
    m_h: *const anyopaque,
    router_nvfp4: *const CLinearWeight,
    router_bias: [*]const f32,
    scratch_logits: [*]f32,
    out_ids: [*]i32,
    out_weights: [*]f32,
    groups: ?*anyopaque,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_route_topk(
    m: [*]const f32,
    router_w: *const anyopaque,
    router_bias: [*]const f32,
    scratch_logits: [*]f32,
    out_ids: [*]i32,
    out_weights: [*]f32,
    groups: ?*anyopaque,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_moe_gate_up_h(
    w13: *const CExpertBank,
    x_h: *const anyopaque,
    route_ids: [*]const i32,
    groups: *const anyopaque,
    out_act_h: *anyopaque,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_moe_down_h(
    w2: *const CExpertBank,
    act_h: *const anyopaque,
    route_ids: [*]const i32,
    route_w: [*]const f32,
    groups: *const anyopaque,
    out_down: [*]f32,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

// CPU expert path for B = 1 decode (cpu/cpu_moe.c + cuda/kernels_cpumoe.cu).
pub extern "c" fn squid_launch_cpu_moe_post(x_h: *const anyopaque, ids: [*]const i32, wts: [*]const f32, shared_dev: *anyopaque, seq_ctr: [*]u32, layer: i32, slot_begin: i32, k_split: i32, stream: ?*anyopaque) void;
pub extern "c" fn squid_launch_l2_prefetch_if(ptrs: [*]const ?*const anyopaque, bytes: [*]const u32, n: i32, mode: i32, gate: ?[*]const i32, stream: ?*anyopaque) void;
pub extern "c" fn squid_launch_l2_prefetch(ptrs: [*]const ?*const anyopaque, bytes: [*]const u32, n: i32, mode: i32, stream: ?*anyopaque) void;
pub extern "c" fn squid_launch_cpu_moe_post_cache_multi(x_h: *const anyopaque, ids: [*]const i32, wts: [*]const f32, shared_dev: *anyopaque, seq_ctr: [*]u32, layer: i32, cmap_l: [*]const i32, glist: [*]i32, hist_l: ?[*]u32, pcie_h: i32, n: i32, stream: ?*anyopaque) void;
pub extern "c" fn squid_launch_cpu_moe_wait_add_multi(shared_dev: *anyopaque, seq_ctr: [*]const u32, out: [*]f32, err: [*]i32, glist: [*]const i32, n: i32, stream: ?*anyopaque) void;
pub extern "c" fn squid_launch_cpu_moe_wait_add(shared_dev: *anyopaque, seq_ctr: [*]const u32, out: [*]f32, err: [*]i32, stream: ?*anyopaque) void;
pub extern "c" fn squid_launch_moe_slots_list_h(w13: *const CExpertBank, w2: *const CExpertBank, w13b: *const CExpertBank, w2b: *const CExpertBank, w13h: *const CExpertBank, w2h: *const CExpertBank, x_h: *const anyopaque, route_w: [*]const f32, act_h: *anyopaque, out_down: [*]f32, glist: [*]const i32, stream: ?*anyopaque) void;
pub extern "c" fn squid_launch_moe_slots_multi_h(w13: *const CExpertBank, w2: *const CExpertBank, w13b: *const CExpertBank, w2b: *const CExpertBank, w13h: *const CExpertBank, w2h: *const CExpertBank, x_h: *const anyopaque, route_w: [*]const f32, act_h: *anyopaque, out_down: [*]f32, gcode: [*]const i32, n: i32, stream: ?*anyopaque) void;
pub extern "c" fn squid_launch_cpu_moe_post_cache(x_h: *const anyopaque, ids: [*]const i32, wts: [*]const f32, shared_dev: *anyopaque, seq_ctr: [*]u32, layer: i32, cmap_l: [*]const i32, glist: [*]i32, hist_l: ?[*]u32, pcie_h: i32, stream: ?*anyopaque) void;
pub extern "c" fn squid_launch_cpu_moe_wait_add_if(shared_dev: *anyopaque, seq_ctr: [*]const u32, out: [*]f32, err: [*]i32, posted: [*]const i32, stream: ?*anyopaque) void;
pub extern "c" fn squid_set_moe_tail_slots(w13: [*]const ?*const anyopaque, w13s: [*]const ?*const anyopaque, w2: [*]const ?*const anyopaque, w2s: [*]const ?*const anyopaque, n: i32, wf_off: i32) i32;
pub extern "c" fn squid_launch_moe_slots_h(w13: *const CExpertBank, w2: *const CExpertBank, x_h: *const anyopaque, route_ids: [*]const i32, route_w: [*]const f32, act_h: *anyopaque, out_down: [*]f32, kslots: i32, k_split: i32, stream: ?*anyopaque) void;
pub extern "c" fn squid_launch_cpu_moe_exchange(
    x_h: *const anyopaque,
    ids: [*]const i32,
    wts: [*]const f32,
    shared_dev: *anyopaque,
    seq_ctr: [*]u32,
    layer: i32,
    out: [*]f32,
    err: [*]i32,
    stream: ?*anyopaque,
) void;
pub extern "c" fn cpu_moe_available() c_int;
pub extern "c" fn cpu_moe_set_layer(
    layer: c_int,
    w13_codes: [*]const u8,
    w13_scales: [*]const u8,
    es13: [*]const f32,
    w2_codes: [*]const u8,
    w2_scales: [*]const u8,
    es2: [*]const f32,
) void;
pub extern "c" fn cpu_moe_set_expert(
    layer: c_int,
    e: c_int,
    w13_codes: [*]const u8,
    w13_scales: [*]const u8,
    es13: f32,
    w2_codes: [*]const u8,
    w2_scales: [*]const u8,
    es2: f32,
) void;
pub extern "c" fn cpu_moe_start(shared_host: *anyopaque, n_threads: c_int, cpus: ?[*]const c_int) c_int;
pub extern "c" fn cpu_moe_kick() void;
pub extern "c" fn cpu_moe_stop() void;
pub extern "c" fn cpu_moe_stats(requests: *u64, busy_ns: *u64, max_ns: *u64) void;
pub extern "c" fn cpu_moe_shared_bytes() usize;
pub extern "c" fn cpu_moe_kernel_name() [*:0]const u8;
pub extern "c" fn cpu_moe_pin_current_thread(cpu: c_int) void;
pub extern "c" fn cpu_moe_write_file(path: [*:0]const u8, data: *const anyopaque, bytes: usize) c_int;
pub extern "c" fn cpu_moe_read_tokens(path: [*:0]const u8, out: [*]i32, max: c_int) c_int;
pub extern "c" fn cpu_moe_score_logits(logits: [*]const f32, vocab: c_int, forced: i32, rec: [*]f32) void;
pub extern "c" fn squid_launch_route_trace(ids: [*]const i32, trace_dev: *anyopaque, tok_ctr: [*]u32, visit: i32, visits: i32, n_max: i32, is_last: i32, stream: ?*anyopaque) void;

pub extern "c" fn squid_launch_mega_mlp_post_res(
    moe_down_or_slots: [*]const f32,
    moe_is_slotted: u8,
    mlp_sw: *const anyopaque,
    mlp_ring: [*]f32,
    residual: [*]const f32,
    hidden_out: [*]f32,
    batch_size: i32,
    d_pos: *const i32,
    abl_r: ?[*]const f32,
    abl_d: ?[*]const f32,
    abl_p: ?[*]const f32,
    udir: ?[*]const f32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_mega_mlp_post_next_depth_norm(
    moe_down_or_slots: [*]const f32,
    moe_is_slotted: u8,
    prev_mlp_sw: *const anyopaque,
    prev_mlp_ring: [*]f32,
    residual: [*]const f32,
    hidden_out: [*]f32,
    pre_tap_out: ?[*]f32,
    post_tap_out: ?[*]f32,
    partial_out: ?[*]f32,
    blocks: [*]const f32,
    n_blocks: i32,
    queries: [*]const f32,
    n0: ?*const anyopaque,
    n1: ?*const anyopaque,
    n2: ?*const anyopaque,
    o0: ?[*]f32,
    o1: ?[*]f32,
    o2: ?[*]f32,
    oh0: ?*anyopaque,
    oh1: ?*anyopaque,
    oh2: ?*anyopaque,
    n_out: i32,
    batch_size: i32,
    d_pos: *const i32,
    abl_r: ?[*]const f32,
    abl_d: ?[*]const f32,
    abl_p: ?[*]const f32,
    udir: ?[*]const f32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_lm_head_argmax(
    unembed: *const CLinearWeight,
    x: [*]const f32,
    out_tokens: [*]i32,
    out_logits: ?[*]f32,
    scratch_vals: [*]f32,
    scratch_idxs: [*]i32,
    sample_cfg: ?*const CSampleConfig,
    d_pos: ?*const i32,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_dspark_commit(
    dw_in: *const CDsparkWeights,
    taps: [*]const f32,
    ctx_k: [*]const ?*anyopaque,
    ctx_v: [*]const ?*anyopaque,
    ctx_ks: [*]const ?*anyopaque,
    ctx_vs: [*]const ?*anyopaque,
    ctx_cap: [*]const i32,
    global_kv_fmt: u8,
    f_ring: [*]f32,
    scratch_mix: [*]f32,
    scratch_kv: [*]f32,
    batch_size: i32,
    d_pos: *const i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_dspark_draft(
    embed: *const anyopaque,
    embed_norm: *const anyopaque,
    dw_in: *const CDsparkWeights,
    d_anchor_token: *const i32,
    d_pos: *const i32,
    num_drafts: i32,
    ctx_k: [*]const ?*anyopaque,
    ctx_v: [*]const ?*anyopaque,
    ctx_ks: [*]const ?*const anyopaque,
    ctx_vs: [*]const ?*const anyopaque,
    ctx_cap: [*]const i32,
    global_kv_fmt: u8,
    split_buf: [*]f32,
    f_ring: [*]const f32,
    workspace: [*]f32,
    out_draft_tokens: [*]i32,
    out_draft_conf: ?[*]f32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_adaptive_graph_create(
    d_anchor: *const i32,
    d_drafts: [*]const i32,
    d_draft_conf: [*]const f32,
    threshold: f32,
    d_tokens: [*]i32,
    d_chosen_width: *i32,
    d_accepted: [*]const i32,
    h_accepted: [*]i32,
    d_num_acc: *const i32,
    h_num_acc: *i32,
    h_chosen_width: *i32,
    stream: ?*anyopaque,
    out_graph: *?*anyopaque,
) i32;
pub extern "c" fn squid_adaptive_graph_branch_begin(graph: ?*anyopaque, branch_idx: i32, stream: ?*anyopaque) i32;
pub extern "c" fn squid_adaptive_graph_branch_end(stream: ?*anyopaque) i32;
pub extern "c" fn squid_adaptive_graph_instantiate(graph: ?*anyopaque) i32;
pub extern "c" fn squid_adaptive_graph_launch(graph: ?*anyopaque, stream: ?*anyopaque) i32;
pub extern "c" fn squid_adaptive_graph_destroy(graph: ?*anyopaque) i32;

pub extern "c" fn squid_launch_spec_verify(
    target_preds: [*]const i32,
    draft_tokens: [*]const i32,
    batch_size: i32,
    out_accepted: [*]i32,
    out_num_accepted: *i32,
    d_pos: *i32,
    d_next_anchor: *i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_inc_pos_and_copy_tok(
    d_pos: *i32,
    d_tok_in: *i32,
    d_tok_out: *const i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_inc_pos_by(
    d_pos: *i32,
    delta: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_assemble_verify_tokens(
    d_anchor: *const i32,
    d_drafts: [*]const i32,
    d_tokens: [*]i32,
    num_drafts: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_sizeof_expert_groups() usize;

pub extern "c" fn squid_launch_prefill_warmup_cublas(stream: ?*anyopaque) void;

pub extern "c" fn squid_launch_prefill_f32_to_f16(
    src: [*]const f32,
    dst_h: *anyopaque,
    count: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_prefill_gemm_h(
    w: *const CLinearWeight,
    x_h: *const anyopaque,
    y: [*]f32,
    w_f16_scratch: *anyopaque,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_prefill_sconv_post_res(
    src_or_slots: [*]const f32,
    is_slotted: i32,
    sconv_w: *const anyopaque,
    ring: [*]f32,
    residual: [*]const f32,
    out_res: [*]f32,
    pre_tap_out: ?[*]f32,
    blocks: ?[*]f32,
    partial: ?[*]const f32,
    block_idx: i32,
    is_boundary: i32,
    batch_size: i32,
    d_pos: *const i32,
    abl_r: ?[*]const f32,
    abl_d: ?[*]const f32,
    abl_p: ?[*]const f32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_prefill_copy_taps(
    src: [*]const f32,
    dst_taps: [*]f32,
    tap_slot: i32,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_prefill_local_sconv_prep_store(
    qrgs: [*]f32,
    k_proj: [*]f32,
    v_proj: [*]f32,
    k_sw: *const anyopaque,
    v_sw: *const anyopaque,
    k_ring: [*]f32,
    v_ring: [*]f32,
    q_cs: *const anyopaque,
    r_cs: *const anyopaque,
    k_cache: *anyopaque,
    v_cache: *anyopaque,
    k_scale: ?*anyopaque,
    v_scale: ?*anyopaque,
    kv_fmt: u8,
    kv_cap: i32,
    batch_size: i32,
    d_pos: *const i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_prefill_global_sconv_prep_store(
    qrgs: [*]f32,
    k_proj: [*]f32,
    v_proj: [*]f32,
    in_k_h: *const anyopaque,
    in_v_h: *const anyopaque,
    k_sw: *const anyopaque,
    v_sw: *const anyopaque,
    k_sp: *const anyopaque,
    v_sp: *const anyopaque,
    k_ring: [*]f32,
    v_ring: [*]f32,
    q_cs: *const anyopaque,
    r_cs: *const anyopaque,
    k_cache: *anyopaque,
    v_cache: *anyopaque,
    k_scale: ?*anyopaque,
    v_scale: ?*anyopaque,
    kv_fmt: u8,
    kv_cap: i32,
    scratch_taps: [*]f32,
    w_f16_scratch: *anyopaque,
    batch_size: i32,
    d_pos: *const i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_prefill_flash_attn_tc(
    qrgs: [*]const f32,
    rel_proj: *const anyopaque,
    k_cache: *const anyopaque,
    v_cache: *const anyopaque,
    k_scale: ?*const anyopaque,
    v_scale: ?*const anyopaque,
    kv_fmt: u8,
    out_gated_h: *anyopaque,
    kv_cap: i32,
    window: i32,
    batch_size: i32,
    d_pos: *const i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_prefill_controller_one(
    x: [*]const f32,
    saved: [*]const f32,
    prior_last: [*]const f32,
    cp_q: [*]const f32,
    cw: *const CControllerWeights,
    post_norm: *const anyopaque,
    out: [*]f32,
    out_h: ?*anyopaque,
    scratch_msg: [*]f32,
    scratch_cat_h: *anyopaque,
    scratch_d: [*]f32,
    scratch_d_h: *anyopaque,
    scratch_f: [*]f32,
    scratch_g: [*]f32,
    w_f16_scratch: *anyopaque,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

/// Prefill token-tile default by expert placement: 128-row tiles at every chunk size when any bank is on the host.
pub extern "c" fn squid_prefill_tiles_set_host_banks(banks_host: i32) void;
pub extern "c" fn squid_prefill_stage_init() u64;
pub extern "c" fn squid_prefill_stage_borrow(ptrs: [*]const u64, sizes: [*]const u64, n: i32, b: i32, stream: ?*anyopaque) i32;
pub extern "c" fn squid_prefill_stage_return() void;

pub extern "c" fn squid_launch_prefill_route_and_sort(
    router_logits: [*]const f32,
    router_bias: [*]const f32,
    route_ids: [*]i32,
    route_w: [*]f32,
    expert_offsets: [*]i32,
    sorted_tok: [*]i32,
    sorted_slot: [*]i32,
    sorted_w: [*]f32,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_prefill_moe_gate_up_h(
    w13: *const CExpertBank,
    x_h: *const anyopaque,
    expert_offsets: [*]const i32,
    sorted_tok: [*]const i32,
    sorted_slot: [*]const i32,
    out_act_h: *anyopaque,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

pub extern "c" fn squid_launch_prefill_moe_down_h(
    w2: *const CExpertBank,
    act_h: *const anyopaque,
    expert_offsets: [*]const i32,
    sorted_tok: [*]const i32,
    sorted_slot: [*]const i32,
    sorted_w: [*]const f32,
    out_slots: [*]f32,
    batch_size: i32,
    stream: ?*anyopaque,
) void;

// --- Vision encoder (cuda/kernels_vision.cu) and image decoding (cpu/image_decode.c) ---
pub const SquidVisionArgs = extern struct {
    glob: ?*const anyopaque,
    layers: ?*const anyopaque,
    proj: ?*const anyopaque,
    patches: ?*const anyopaque,
    rope: ?[*]const f32,
    out: ?[*]f32,
    dbg: ?[*]f32,
    scratch: ?*anyopaque,
    scratch_bytes: u64,
    gw: i32,
    gh: i32,
};
pub extern "c" fn squid_vision_layer_bytes() u64;
pub extern "c" fn squid_vision_glob_bytes() u64;
pub extern "c" fn squid_vision_proj_bytes() u64;
pub extern "c" fn squid_vision_scratch_bytes(n_rows: i32) u64;
pub extern "c" fn squid_vision_forward(args: *const SquidVisionArgs, stream: ?*anyopaque) i32;
pub extern "c" fn squid_image_probe(buf: [*]const u8, len: usize, w: *c_int, h: *c_int) c_int;
pub extern "c" fn squid_image_decode_rgb(buf: [*]const u8, len: usize, w: *c_int, h: *c_int) ?[*]u8;
pub extern "c" fn squid_image_free(p: ?[*]u8) void;
