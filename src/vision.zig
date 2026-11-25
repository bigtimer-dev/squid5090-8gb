// Image understanding: the Inkling 2 vision encoder (cuda/kernels_vision.cu) and everything around it on the host.
//
// - The server probes each image of a request (PNG, JPEG, WebP), plans its size and emits N placeholder tokens.
// - The engine asks for the embeddings of the images it has to prefill (not those inside a reused prefix): they
//   come from a small cache keyed by the image bytes, or the encoder runs in the prefill buffers.
// - The encoder weights (610 MiB) load from the model file on the first image and stay in pinned host RAM.
//
// Preprocessing follows the reference processor: the image at its own size, padded bottom-right to a multiple of 40
// pixels with -1 (after (x - 127.5) / 127.5). One token per 40x40 pixels. Images above the token cap
// (SQUID_IMAGE_MAX_TOKENS, default 1600, about 2.5 MP) are first scaled down by area averaging (the reference does
// not resize; llama.cpp does the same to bound the token count).

const std = @import("std");
const cuda = @import("cuda.zig");
const safetensors = @import("safetensors.zig");
const tiger = @import("tiger.zig");

pub const embed_dim: usize = 2304;
pub const placeholder_token: i32 = 200054; // <|unused_200054|>
pub const content_image_token: i32 = 200005; // <|content_image|>
pub const images_per_request_max: usize = 64;
pub const max_tokens_default: u32 = 1600;

const vd: usize = 1024;
const vff: usize = 4096;
const vlayers: usize = 23;
const patch: usize = 20;
const patch_in: usize = patch * patch * 3;
const cell: u32 = 40; // patch x merge

// Host block layouts: must match cuda/kernels_vision.cu (checked against it at load).
const l_qkv_w: usize = 0;
const l_o_w: usize = l_qkv_w + 3 * vd * vd * 2;
const l_fc_w: usize = l_o_w + vd * vd * 2;
const l_pr_w: usize = l_fc_w + vff * vd * 2;
const l_vec: usize = l_pr_w + vd * vff * 2;
const lv_ln1_w: usize = 0;
const lv_ln1_b: usize = vd;
const lv_ln2_w: usize = 2 * vd;
const lv_ln2_b: usize = 3 * vd;
const lv_ls1: usize = 4 * vd;
const lv_ls2: usize = 5 * vd;
const lv_qkv_b: usize = 6 * vd;
const lv_o_b: usize = 9 * vd;
const lv_fc_b: usize = 10 * vd;
const lv_pr_b: usize = 10 * vd + vff;
const lv_count: usize = 11 * vd + vff;
const layer_bytes: usize = l_vec + lv_count * 4;
const g_patch_w: usize = 0;
const g_pos: usize = g_patch_w + vd * patch_in * 2;
const g_cls: usize = g_pos + 1024 * vd * 4;
const glob_bytes: usize = g_cls + 4 * vd * 4;
const p_in_w: usize = 0;
const p_out_w: usize = p_in_w + 4 * vd * 4 * vd * 2;
const proj_bytes: usize = p_out_w + embed_dim * 4 * vd * 2;

/// One image of a request, as the server found it.
pub const ImageRef = struct {
    hash: u64,
    bytes: []const u8, // the encoded file, valid for the request
    w: u32,
    h: u32,
    tw: u32, // size the encoder sees (after the token cap's resize; == w, h without one)
    th: u32,
    gw: u32, // patch grid of the padded image (even)
    gh: u32,
    tokens: u32,
    start: u32 = 0, // first placeholder position in the prompt
};

pub const SessionImage = struct {
    start: u32,
    tokens: u32,
    hash: u64,
};

pub const Plan = struct { tw: u32, th: u32, gw: u32, gh: u32, tokens: u32 };

pub fn plan(w: u32, h: u32, max_tokens: u32) Plan {
    tiger.assert(w > 0 and h > 0 and max_tokens >= 1);
    var tw: u32 = w;
    var th: u32 = h;
    const cw0: u64 = (w + cell - 1) / cell;
    const ch0: u64 = (h + cell - 1) / cell;
    if (cw0 * ch0 > max_tokens) {
        const s: f64 = @sqrt(@as(f64, @floatFromInt(max_tokens)) * @as(f64, cell * cell) / (@as(f64, @floatFromInt(w)) * @as(f64, @floatFromInt(h))));
        var a: u32 = @max(1, @as(u32, @intFromFloat(@floor(@as(f64, @floatFromInt(w)) * s / @as(f64, cell)))));
        var b: u32 = @max(1, @as(u32, @intFromFloat(@floor(@as(f64, @floatFromInt(h)) * s / @as(f64, cell)))));
        while (@as(u64, a) * b > max_tokens) {
            if (a >= b) a -= 1 else b -= 1;
        }
        tw = a * cell;
        th = b * cell;
    }
    const cw: u32 = (tw + cell - 1) / cell;
    const ch: u32 = (th + cell - 1) / cell;
    return .{ .tw = tw, .th = th, .gw = 2 * cw, .gh = 2 * ch, .tokens = cw * ch };
}

pub const ProbeResult = union(enum) {
    ok: struct { w: u32, h: u32 },
    unsupported,
    broken,
};

pub fn probe(bytes: []const u8) ProbeResult {
    var w: c_int = 0;
    var h: c_int = 0;
    const fmt = cuda.squid_image_probe(bytes.ptr, bytes.len, &w, &h);
    if (fmt == 0) return .unsupported;
    if (fmt < 0) return .broken;
    return .{ .ok = .{ .w = @intCast(w), .h = @intCast(h) } };
}

/// Base64 (standard or URL-safe alphabet) into `out`; skips whitespace, JSON's `\/` backslashes and padding.
pub fn base64_decode(src: []const u8, out: []u8) ?usize {
    var acc: u32 = 0;
    var bits: u5 = 0;
    var n: usize = 0;
    for (src) |c| {
        const v: u32 = switch (c) {
            'A'...'Z' => c - 'A',
            'a'...'z' => c - 'a' + 26,
            '0'...'9' => c - '0' + 52,
            '+', '-' => 62,
            '/', '_' => 63,
            '=', '\\', ' ', '\n', '\r', '\t' => continue,
            else => return null,
        };
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            if (n >= out.len) return null;
            out[n] = @truncate(acc >> bits);
            n += 1;
            acc &= (@as(u32, 1) << bits) - 1;
        }
    }
    return n;
}

const CacheEntry = struct {
    hash: u64 = 0,
    tokens: u32 = 0,
    cap: u32 = 0,
    data: ?[*]f32 = null,
    stamp: u64 = 0,
    request: u64 = 0,
};
const cache_keep: usize = 8;

pub const Vision = struct {
    model_dir: []const u8 = "",
    max_tokens: u32 = max_tokens_default,
    host: ?[*]u8 = null, // pinned: glob | 23 layers | proj
    stage: ?[*]u8 = null, // pinned: fp16 patches | rope tables
    stage_bytes: usize = 0,
    load_failed: bool = false,
    cache: [images_per_request_max]CacheEntry = @splat(.{}),
    clock: u64 = 0,
    request: u64 = 0,
    dbg: ?[*]f32 = null, // SQUID_VISION_DUMP
    encodes: u64 = 0,
    cache_hits: u64 = 0,

    pub fn init(self: *Vision, model_dir: []const u8, scratch_bytes: u64) void {
        self.model_dir = model_dir;
        var mt: u32 = max_tokens_default;
        if (tiger.getenv("SQUID_IMAGE_MAX_TOKENS")) |v| mt = std.fmt.parseInt(u32, std.mem.span(v), 10) catch mt;
        // the prefill buffers bound the rows the encoder can hold
        var fit: u32 = 1;
        while (fit < 16384 and cuda.squid_vision_scratch_bytes(@intCast(4 * (fit + 1) + 1)) <= scratch_bytes) fit += 1;
        self.max_tokens = @max(1, @min(mt, fit));
        tiger.print_stdout("[vision] images on: up to {d} tokens each (SQUID_IMAGE_MAX_TOKENS; the prefill buffers fit {d}); encoder weights load on the first image\n", .{ self.max_tokens, fit });
    }

    /// A new request: the cache entries it uses stay put while it runs.
    pub fn begin_request(self: *Vision) void {
        self.trim();
        self.request += 1;
    }

    fn lookup(self: *Vision, hash: u64, tokens: u32) ?[*]const f32 {
        for (&self.cache) |*e| {
            if (e.data != null and e.tokens == tokens and e.hash == hash and e.tokens > 0) {
                self.clock += 1;
                e.stamp = self.clock;
                e.request = self.request;
                return e.data.?;
            }
        }
        return null;
    }

    /// A cache slot for `tokens` embeddings, never one the current request uses: an empty one while fewer than
    /// `cache_keep` are allocated, else the least recently used.
    fn take_slot(self: *Vision, tokens: u32) ?*CacheEntry {
        var allocated: usize = 0;
        for (&self.cache) |*e| {
            if (e.data != null) allocated += 1;
        }
        var victim: ?*CacheEntry = null;
        for (&self.cache) |*e| {
            if (e.data != null and e.request == self.request) continue;
            if (allocated < cache_keep and e.data == null) {
                victim = e;
                break;
            }
            if (e.data == null) {
                if (victim == null) victim = e;
                continue;
            }
            if (victim == null or victim.?.data == null or e.stamp < victim.?.stamp) victim = e;
        }
        const e = victim orelse return null;
        if (e.data != null and e.cap < tokens) {
            _ = cuda.squid_cuda_free_host(@ptrCast(e.data.?));
            e.data = null;
            e.cap = 0;
        }
        if (e.data == null) {
            var p: ?*anyopaque = null;
            if (cuda.squid_cuda_malloc_host_pinned(&p, @as(usize, tokens) * embed_dim * 4, 0) != 0 or p == null) return null;
            e.data = @ptrCast(@alignCast(p.?));
            e.cap = tokens;
        }
        e.hash = 0;
        e.tokens = 0;
        return e;
    }

    /// Free the least recently used entries beyond `cache_keep` (a request with many images allocates more).
    fn trim(self: *Vision) void {
        while (true) {
            var allocated: usize = 0;
            var lru: ?*CacheEntry = null;
            for (&self.cache) |*e| {
                if (e.data == null) continue;
                allocated += 1;
                if (lru == null or e.stamp < lru.?.stamp) lru = e;
            }
            if (allocated <= cache_keep) return;
            _ = cuda.squid_cuda_free_host(@ptrCast(lru.?.data.?));
            lru.?.* = .{};
        }
    }

    fn ensure_loaded(self: *Vision) bool {
        if (self.host != null) return true;
        if (self.load_failed) return false;
        self.load_failed = !self.load();
        return !self.load_failed;
    }

    pub fn ready(self: *Vision) bool {
        return self.ensure_loaded();
    }

    fn load(self: *Vision) bool {
        const t0 = tiger.clock_monotonic_ns();
        if (cuda.squid_vision_layer_bytes() != layer_bytes or cuda.squid_vision_glob_bytes() != glob_bytes or cuda.squid_vision_proj_bytes() != proj_bytes) {
            tiger.print_stdout("[vision] host layout mismatch with the CUDA encoder; images off\n", .{});
            return false;
        }
        var path_buf: [1024]u8 = undefined;
        const path_z = tiger.format_path_z(&path_buf, self.model_dir, "/model-00001-of-00001.safetensors");
        const sf = std.heap.page_allocator.create(safetensors.SafetensorsFile) catch return false;
        defer std.heap.page_allocator.destroy(sf);
        sf.* = .{};
        if (!sf.open_file(path_z)) {
            tiger.print_stdout("[vision] cannot open {s}; images off\n", .{path_z});
            return false;
        }
        defer sf.close_file();
        if (sf.lookup_optional("model.visual.proj_out.weight") == null) {
            tiger.print_stdout("[vision] the model file has no vision encoder (model.visual.*); images off\n", .{});
            return false;
        }

        const total: usize = glob_bytes + vlayers * layer_bytes + proj_bytes;
        var hp: ?*anyopaque = null;
        if (cuda.squid_cuda_malloc_host_pinned(&hp, total, 0) != 0 or hp == null) {
            tiger.print_stdout("[vision] cannot pin {d} MiB of host RAM for the encoder; images off\n", .{total >> 20});
            return false;
        }
        const host: [*]u8 = @ptrCast(hp.?);
        const tmp = std.heap.page_allocator.alloc(u8, 4 * vd * 4 * vd * 2) catch {
            _ = cuda.squid_cuda_free_host(hp);
            return false;
        };
        defer std.heap.page_allocator.free(tmp);

        var ld: Loader = .{ .sf = sf, .tmp = tmp };
        const e = "model.visual.backbone.embeddings.";
        ld.put_f16(host + g_patch_w, e ++ "patch_embedding.weight", vd * patch_in);
        ld.put_f32(host + g_pos, e ++ "position_embedding", 1024 * vd);
        ld.put_f32(host + g_cls, e ++ "class_embedding", vd);
        ld.put_f32(host + g_cls + vd * 4, e ++ "cls_position_embedding", vd);
        ld.put_f32(host + g_cls + 2 * vd * 4, "model.visual.backbone.ln_pre.weight", vd);
        ld.put_f32(host + g_cls + 3 * vd * 4, "model.visual.backbone.ln_pre.bias", vd);
        var name_buf: [160]u8 = undefined;
        for (0..vlayers) |l| {
            const lb: [*]u8 = host + glob_bytes + l * layer_bytes;
            const Item = struct { off: usize, suffix: []const u8, n: usize, half: bool };
            const items = [_]Item{
                .{ .off = l_qkv_w, .suffix = "self_attn.q_proj.weight", .n = vd * vd, .half = true },
                .{ .off = l_qkv_w + vd * vd * 2, .suffix = "self_attn.k_proj.weight", .n = vd * vd, .half = true },
                .{ .off = l_qkv_w + 2 * vd * vd * 2, .suffix = "self_attn.v_proj.weight", .n = vd * vd, .half = true },
                .{ .off = l_o_w, .suffix = "self_attn.out_proj.weight", .n = vd * vd, .half = true },
                .{ .off = l_fc_w, .suffix = "mlp.c_fc.weight", .n = vff * vd, .half = true },
                .{ .off = l_pr_w, .suffix = "mlp.c_proj.weight", .n = vd * vff, .half = true },
                .{ .off = l_vec + lv_ln1_w * 4, .suffix = "ln_1.weight", .n = vd, .half = false },
                .{ .off = l_vec + lv_ln1_b * 4, .suffix = "ln_1.bias", .n = vd, .half = false },
                .{ .off = l_vec + lv_ln2_w * 4, .suffix = "ln_2.weight", .n = vd, .half = false },
                .{ .off = l_vec + lv_ln2_b * 4, .suffix = "ln_2.bias", .n = vd, .half = false },
                .{ .off = l_vec + lv_ls1 * 4, .suffix = "ls_1.gamma", .n = vd, .half = false },
                .{ .off = l_vec + lv_ls2 * 4, .suffix = "ls_2.gamma", .n = vd, .half = false },
                .{ .off = l_vec + lv_qkv_b * 4, .suffix = "self_attn.q_proj.bias", .n = vd, .half = false },
                .{ .off = l_vec + (lv_qkv_b + vd) * 4, .suffix = "self_attn.k_proj.bias", .n = vd, .half = false },
                .{ .off = l_vec + (lv_qkv_b + 2 * vd) * 4, .suffix = "self_attn.v_proj.bias", .n = vd, .half = false },
                .{ .off = l_vec + lv_o_b * 4, .suffix = "self_attn.out_proj.bias", .n = vd, .half = false },
                .{ .off = l_vec + lv_fc_b * 4, .suffix = "mlp.c_fc.bias", .n = vff, .half = false },
                .{ .off = l_vec + lv_pr_b * 4, .suffix = "mlp.c_proj.bias", .n = vd, .half = false },
            };
            for (items) |it| {
                const name = std.fmt.bufPrint(&name_buf, "model.visual.backbone.layers.{d}.{s}", .{ l, it.suffix }) catch unreachable;
                if (it.half) ld.put_f16(lb + it.off, name, it.n) else ld.put_f32(lb + it.off, name, it.n);
            }
        }
        const pb: [*]u8 = host + glob_bytes + vlayers * layer_bytes;
        ld.put_f16(pb + p_in_w, "model.visual.proj_in.weight", 4 * vd * 4 * vd);
        ld.put_f16(pb + p_out_w, "model.visual.proj_out.weight", embed_dim * 4 * vd);
        if (!ld.ok) {
            _ = cuda.squid_cuda_free_host(hp);
            tiger.print_stdout("[vision] the encoder weights do not match the expected shapes; images off\n", .{});
            return false;
        }

        // Staging: fp16 patches for the largest image + RoPE tables (+ the debug dump when asked).
        const p_max: usize = 4 * @as(usize, self.max_tokens);
        self.stage_bytes = p_max * patch_in * 2 + 4 * 16 * 1024 * 4;
        var sp: ?*anyopaque = null;
        if (cuda.squid_cuda_malloc_host_pinned(&sp, self.stage_bytes, 0) != 0 or sp == null) {
            _ = cuda.squid_cuda_free_host(hp);
            return false;
        }
        self.stage = @ptrCast(sp.?);
        if (tiger.getenv("SQUID_VISION_DUMP") != null) {
            var dp: ?*anyopaque = null;
            if (cuda.squid_cuda_malloc_host_pinned(&dp, 3 * (p_max + 1) * vd * 4, 0) == 0) self.dbg = @ptrCast(@alignCast(dp.?));
        }
        self.host = host;
        const ms = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - t0)) / 1e6;
        tiger.print_stdout("[vision] encoder weights loaded: {d} MiB pinned host RAM in {d:.0} ms{s}\n", .{ total >> 20, ms, if (ld.f16_overflow) " (WARNING: some weights overflow fp16)" else "" });
        return true;
    }

    /// Embeddings [tokens, 2304] of `img` (pinned host memory, valid while this request runs), running the
    /// encoder in `scratch` on `stream` when they are not cached. Null on failure.
    pub fn embeddings(self: *Vision, img: *const ImageRef, scratch: *anyopaque, scratch_bytes: u64, stream: ?*anyopaque) ?[*]const f32 {
        if (self.lookup(img.hash, img.tokens)) |p| {
            self.cache_hits += 1;
            return p;
        }
        if (!self.ensure_loaded()) return null;
        const t0 = tiger.clock_monotonic_ns();
        var w: c_int = 0;
        var h: c_int = 0;
        const rgb = cuda.squid_image_decode_rgb(img.bytes.ptr, img.bytes.len, &w, &h) orelse {
            tiger.print_stdout("[vision] image decode failed\n", .{});
            return null;
        };
        defer cuda.squid_image_free(rgb);
        if (w != img.w or h != img.h or img.tokens > self.max_tokens) return null;
        const t1 = tiger.clock_monotonic_ns();

        const stage = self.stage.?;
        const patches: [*]f16 = @ptrCast(@alignCast(stage));
        const P: usize = @as(usize, img.gw) * img.gh;
        const rope: [*]f32 = @ptrCast(@alignCast(stage + 4 * @as(usize, self.max_tokens) * patch_in * 2));
        if (img.tw == img.w and img.th == img.h) {
            fill_patches_u8(rgb[0 .. @as(usize, img.w) * img.h * 3], img.w, img.h, img.gw, img.gh, patches[0 .. P * patch_in]);
        } else {
            const rs = resize_area(rgb[0 .. @as(usize, img.w) * img.h * 3], img.w, img.h, img.tw, img.th) orelse return null;
            defer std.heap.page_allocator.free(rs);
            fill_patches_f32(rs, img.tw, img.th, img.gw, img.gh, patches[0 .. P * patch_in]);
        }
        fill_rope(rope, img.gw, img.gh);
        const t2 = tiger.clock_monotonic_ns();

        const slot = self.take_slot(img.tokens) orelse return null;
        const host = self.host.?;
        const args: cuda.SquidVisionArgs = .{
            .glob = @ptrCast(host),
            .layers = @ptrCast(host + glob_bytes),
            .proj = @ptrCast(host + glob_bytes + vlayers * layer_bytes),
            .patches = @ptrCast(patches),
            .rope = rope,
            .out = slot.data.?,
            .dbg = self.dbg,
            .scratch = scratch,
            .scratch_bytes = scratch_bytes,
            .gw = @intCast(img.gw),
            .gh = @intCast(img.gh),
        };
        const rc = cuda.squid_vision_forward(&args, stream);
        if (rc != 0 or cuda.squid_cuda_stream_sync(stream) != 0) {
            tiger.print_stdout("[vision] encoder failed ({d})\n", .{rc});
            return null;
        }
        const t3 = tiger.clock_monotonic_ns();
        self.clock += 1;
        slot.* = .{ .hash = img.hash, .tokens = img.tokens, .cap = slot.cap, .data = slot.data, .stamp = self.clock, .request = self.request };
        self.encodes += 1;
        tiger.print_stdout("[vision] image {d}x{d}{s} -> {d} tokens: decode {d:.0} ms, preprocess {d:.0} ms, encoder {d:.0} ms\n", .{
            img.w,                                                     img.h,
            if (img.tw != img.w or img.th != img.h) " (resized)" else "", img.tokens,
            @as(f64, @floatFromInt(t1 - t0)) / 1e6,                     @as(f64, @floatFromInt(t2 - t1)) / 1e6,
            @as(f64, @floatFromInt(t3 - t2)) / 1e6,
        });
        if (self.dbg) |d| dump_debug(d, slot.data.?, img);
        return slot.data.?;
    }
};

const Loader = struct {
    sf: *safetensors.SafetensorsFile,
    tmp: []u8,
    ok: bool = true,
    f16_overflow: bool = false,

    fn bf16(self: *Loader, name: []const u8, n: usize) ?[]const u16 {
        const info = self.sf.lookup_optional(name) orelse {
            tiger.print_stdout("[vision] missing tensor {s}\n", .{name});
            self.ok = false;
            return null;
        };
        if (info.byte_length != n * 2 or n * 2 > self.tmp.len) {
            tiger.print_stdout("[vision] tensor {s}: {d} bytes, expected {d}\n", .{ name, info.byte_length, n * 2 });
            self.ok = false;
            return null;
        }
        self.sf.copy_to_host(info, self.tmp.ptr, n * 2);
        const p: [*]const u16 = @ptrCast(@alignCast(self.tmp.ptr));
        return p[0..n];
    }

    fn put_f16(self: *Loader, dst: [*]u8, name: []const u8, n: usize) void {
        const src = self.bf16(name, n) orelse return;
        const out: [*]f16 = @ptrCast(@alignCast(dst));
        const V = 16;
        var i: usize = 0;
        var big: bool = false;
        while (i + V <= n) : (i += V) {
            const b: @Vector(V, u16) = src[i..][0..V].*;
            const b32: @Vector(V, u32) = b;
            const sh: @Vector(V, u5) = @splat(16);
            const f: @Vector(V, f32) = @bitCast(b32 << sh);
            big = big or @reduce(.Max, @abs(f)) > 65504.0;
            const hv: @Vector(V, f16) = @floatCast(f);
            out[i..][0..V].* = hv;
        }
        while (i < n) : (i += 1) {
            const f: f32 = @bitCast(@as(u32, src[i]) << 16);
            big = big or @abs(f) > 65504.0;
            out[i] = @floatCast(f);
        }
        if (big) self.f16_overflow = true;
    }

    fn put_f32(self: *Loader, dst: [*]u8, name: []const u8, n: usize) void {
        const src = self.bf16(name, n) orelse return;
        const out: [*]f32 = @ptrCast(@alignCast(dst));
        for (0..n) |i| out[i] = @bitCast(@as(u32, src[i]) << 16);
    }
};

// Patches [gh*gw][20*20*3] (HWC inside a patch, row-major over the grid), normalized, padding -1.
fn fill_patches_u8(rgb: []const u8, w: u32, h: u32, gw: u32, gh: u32, out: []f16) void {
    var p: usize = 0;
    for (0..gh) |gy| {
        for (0..gw) |gx| {
            const o = out[p * patch_in ..][0..patch_in];
            for (0..patch) |py| {
                const y = gy * patch + py;
                for (0..patch) |px| {
                    const x = gx * patch + px;
                    const k = (py * patch + px) * 3;
                    if (y < h and x < w) {
                        const s = (y * w + x) * 3;
                        inline for (0..3) |c| o[k + c] = @floatCast((@as(f32, @floatFromInt(rgb[s + c])) - 127.5) / 127.5);
                    } else {
                        inline for (0..3) |c| o[k + c] = -1.0;
                    }
                }
            }
            p += 1;
        }
    }
}

fn fill_patches_f32(rgb: []const f32, w: u32, h: u32, gw: u32, gh: u32, out: []f16) void {
    var p: usize = 0;
    for (0..gh) |gy| {
        for (0..gw) |gx| {
            const o = out[p * patch_in ..][0..patch_in];
            for (0..patch) |py| {
                const y = gy * patch + py;
                for (0..patch) |px| {
                    const x = gx * patch + px;
                    const k = (py * patch + px) * 3;
                    if (y < h and x < w) {
                        const s = (y * w + x) * 3;
                        inline for (0..3) |c| o[k + c] = @floatCast((rgb[s + c] - 127.5) / 127.5);
                    } else {
                        inline for (0..3) |c| o[k + c] = -1.0;
                    }
                }
            }
            p += 1;
        }
    }
}

// cos/sin of the 2D RoPE angles, computed in double: [cos_w (gw+1)x16 | sin_w | cos_h (gh+1)x16 | sin_h].
fn fill_rope(out: [*]f32, gw: u32, gh: u32) void {
    var inv: [16]f64 = undefined;
    for (0..16) |i| inv[i] = std.math.pow(f64, 10000.0, -@as(f64, @floatFromInt(2 * i)) / 32.0);
    var o: usize = 0;
    inline for (0..2) |axis| {
        const n: usize = (if (axis == 0) gw else gh) + 1;
        for (0..n) |p| {
            for (0..16) |i| out[o + p * 16 + i] = @floatCast(@cos(@as(f64, @floatFromInt(p)) * inv[i]));
        }
        o += n * 16;
        for (0..n) |p| {
            for (0..16) |i| out[o + p * 16 + i] = @floatCast(@sin(@as(f64, @floatFromInt(p)) * inv[i]));
        }
        o += n * 16;
    }
}

// Area-average downscale of RGB8 [h][w][3] to f32 [th][tw][3] (separable box filter with fractional edges).
fn resize_area(rgb: []const u8, w: u32, h: u32, tw: u32, th: u32) ?[]f32 {
    const pa = std.heap.page_allocator;
    const mid = pa.alloc(f32, @as(usize, tw) * h * 3) catch return null;
    defer pa.free(mid);
    const out = pa.alloc(f32, @as(usize, tw) * th * 3) catch return null;
    const sx: f64 = @as(f64, @floatFromInt(w)) / @as(f64, @floatFromInt(tw));
    const sy: f64 = @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(th));
    for (0..h) |y| {
        for (0..tw) |ox| {
            const a: f64 = @as(f64, @floatFromInt(ox)) * sx;
            const b: f64 = a + sx;
            var acc: [3]f64 = .{ 0, 0, 0 };
            var x: usize = @intFromFloat(@floor(a));
            while (x < w and @as(f64, @floatFromInt(x)) < b) : (x += 1) {
                const xf: f64 = @floatFromInt(x);
                const wt = @min(b, xf + 1) - @max(a, xf);
                if (wt <= 0) continue;
                const s = (y * w + x) * 3;
                inline for (0..3) |c| acc[c] += wt * @as(f64, @floatFromInt(rgb[s + c]));
            }
            inline for (0..3) |c| mid[(y * tw + ox) * 3 + c] = @floatCast(acc[c] / sx);
        }
    }
    for (0..th) |oy| {
        const a: f64 = @as(f64, @floatFromInt(oy)) * sy;
        const b: f64 = a + sy;
        for (0..tw) |ox| {
            var acc: [3]f64 = .{ 0, 0, 0 };
            var y: usize = @intFromFloat(@floor(a));
            while (y < h and @as(f64, @floatFromInt(y)) < b) : (y += 1) {
                const yf: f64 = @floatFromInt(y);
                const wt = @min(b, yf + 1) - @max(a, yf);
                if (wt <= 0) continue;
                inline for (0..3) |c| acc[c] += wt * mid[(y * tw + ox) * 3 + c];
            }
            inline for (0..3) |c| out[(oy * tw + ox) * 3 + c] = @floatCast(acc[c] / sy);
        }
    }
    return out;
}

// SQUID_VISION_DUMP=<prefix>: write <prefix>.lnpre/.l0/.enc ([n,1024] f32) and .emb ([tokens,2304] f32).
fn dump_debug(dbg: [*]const f32, emb: [*]const f32, img: *const ImageRef) void {
    const prefix = std.mem.span(tiger.getenv("SQUID_VISION_DUMP").?);
    const n: usize = @as(usize, img.gw) * img.gh + 1;
    const parts = [_]struct { ext: []const u8, data: []const f32 }{
        .{ .ext = ".lnpre", .data = dbg[0 .. n * vd] },
        .{ .ext = ".l0", .data = dbg[n * vd .. 2 * n * vd] },
        .{ .ext = ".enc", .data = dbg[2 * n * vd .. 3 * n * vd] },
        .{ .ext = ".emb", .data = emb[0 .. @as(usize, img.tokens) * embed_dim] },
    };
    for (parts) |part| {
        var buf: [1024]u8 = undefined;
        const path = std.fmt.bufPrintSentinel(&buf, "{s}{s}", .{ prefix, part.ext }, 0) catch continue;
        const fd = tiger.open(path, 0o1101, @as(c_int, 0o644)); // O_WRONLY | O_CREAT | O_TRUNC
        if (fd < 0) continue;
        tiger.write_fd_all(fd, std.mem.sliceAsBytes(part.data));
        _ = tiger.close(fd);
    }
}
