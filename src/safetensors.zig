// TigerStyle POSIX mmap Safetensors loader.
// Static fixed-capacity tensor directory (2048 tensors, 4096-slot FNV-1a open-addressed table).
// Zero recursion, zero heap allocations, bounded loops, paired offset & size assertions.

const std = @import("std");
const tiger = @import("tiger.zig");
const cuda = @import("cuda.zig");

pub const tensor_count_max: u32 = 2048;
pub const hash_slot_count: u32 = 4096;
pub const header_bytes_max: u64 = 16 * 1024 * 1024;

comptime {
    if ((hash_slot_count & (hash_slot_count - 1)) != 0) {
        @compileError("hash_slot_count must be a power of two");
    }
    if (hash_slot_count < tensor_count_max * 2) {
        @compileError("hash_slot_count must provide <= 50% load factor");
    }
}

pub const TensorInfo = struct {
    name_offset: u32 = 0,
    name_length: u16 = 0,
    data_offset: u64 = 0,
    byte_length: u64 = 0,
};

pub const SafetensorsFile = struct {
    file_descriptor: i32 = -1,
    // Big-tensor fast path: O_DIRECT fd + 64 MiB pinned bounce buffer (large sequential reads instead of
    // 128 KB mmap fault readahead, which crawls through dm-crypt). Only set up for files > 256 MiB.
    direct_fd: i32 = -1,
    bounce: ?[*]u8 = null,
    mapped_ptr: ?[*]const u8 = null,
    file_bytes: u64 = 0,
    data_base_offset: u64 = 0,
    header_bytes: u64 = 0,
    tensor_count: u32 = 0,
    tensors: [tensor_count_max]TensorInfo = @splat(.{}),
    hash_table: [hash_slot_count]u16 = @splat(empty_slot),

    const empty_slot: u16 = 0xffff;

    pub fn open_file(self: *SafetensorsFile, path_z: [*:0]const u8) bool {
        tiger.assert(self.mapped_ptr == null);
        tiger.assert(self.file_descriptor < 0);

        const fd: i32 = tiger.open(path_z, tiger.O_RDONLY);
        if (fd < 0) return false;

        const size_i64: i64 = tiger.lseek(fd, 0, tiger.SEEK_END);
        if (size_i64 <= 8) {
            _ = tiger.close(fd);
            return false;
        }
        const file_bytes: u64 = @intCast(size_i64);
        const mapped_raw: *anyopaque = tiger.mmap(null, @intCast(file_bytes), tiger.PROT_READ, tiger.MAP_PRIVATE, fd, 0);
        if (@intFromPtr(mapped_raw) == tiger.MAP_FAILED_ADDR) {
            _ = tiger.close(fd);
            return false;
        }

        const base: [*]const u8 = @ptrCast(mapped_raw);
        const header_len: u64 = std.mem.readInt(u64, base[0..8], .little);
        tiger.assert(header_len > 0);
        tiger.assert(header_len <= header_bytes_max);
        tiger.assert(8 + header_len <= file_bytes);

        self.file_descriptor = fd;
        self.mapped_ptr = base;
        self.file_bytes = file_bytes;
        self.header_bytes = header_len;
        self.data_base_offset = 8 + header_len;
        self.tensor_count = 0;
        @memset(&self.hash_table, empty_slot);

        self.parse_header(base[8 .. 8 + @as(usize, @intCast(header_len))]);
        tiger.assert(self.tensor_count > 0);

        if (file_bytes > 256 * 1024 * 1024) {
            const dfd: i32 = tiger.open(path_z, tiger.O_RDONLY | tiger.O_DIRECT);
            if (dfd >= 0) {
                var b: ?*anyopaque = null;
                if (cuda.squid_cuda_malloc_host_pinned(&b, bounce_bytes, 0) == 0 and b != null) {
                    self.direct_fd = dfd;
                    self.bounce = @ptrCast(b.?);
                } else {
                    _ = tiger.close(dfd);
                }
            }
        }
        return true;
    }

    const bounce_bytes: usize = 64 * 1024 * 1024;
    const dio_align: u64 = 4096;
    const chunk_bytes_max: u64 = bounce_bytes - 2 * dio_align;

    /// Read [file_off, file_off + n) through O_DIRECT into the pinned bounce buffer; returns exactly those bytes.
    fn read_chunk(self: *const SafetensorsFile, file_off: u64, n: u64) []const u8 {
        tiger.assert(n > 0 and n <= chunk_bytes_max);
        const a0: u64 = file_off & ~(dio_align - 1);
        const end: u64 = file_off + n;
        const a1: u64 = tiger.align_up_u64(end, dio_align);
        const span: usize = @intCast(a1 - a0);
        const buf: [*]u8 = self.bounce.?;
        var got: usize = 0;
        while (got < span) {
            const r: isize = tiger.pread(self.direct_fd, buf + got, span - got, @intCast(a0 + got));
            if (r <= 0) break; // short read only at EOF (last partial block)
            got += @intCast(r);
        }
        tiger.assert(a0 + got >= end);
        const skip: usize = @intCast(file_off - a0);
        return buf[skip .. skip + @as(usize, @intCast(n))];
    }

    /// Copy the first `len` bytes of a tensor into host memory.
    pub fn copy_to_host(self: *const SafetensorsFile, info: *const TensorInfo, dst: [*]u8, len: u64) void {
        tiger.assert(len <= info.byte_length);
        if (self.bounce == null) {
            @memcpy(dst[0..@intCast(len)], self.data_slice(info)[0..@intCast(len)]);
            return;
        }
        const start: u64 = self.data_base_offset + info.data_offset;
        var done: u64 = 0;
        while (done < len) {
            const n: u64 = @min(len - done, chunk_bytes_max);
            const src = self.read_chunk(start + done, n);
            @memcpy(dst[@intCast(done)..][0..@intCast(n)], src);
            done += n;
        }
    }

    /// Copy the first `len` bytes of a tensor into device memory (DMA from the pinned bounce buffer).
    pub fn copy_to_device(self: *const SafetensorsFile, info: *const TensorInfo, dst_dev: *anyopaque, len: u64) void {
        self.copy_range_to_device(info, 0, len, dst_dev);
    }

    /// Copy tensor bytes [rel, rel + len) into device memory.
    pub fn copy_range_to_device(self: *const SafetensorsFile, info: *const TensorInfo, rel: u64, len: u64, dst_dev: *anyopaque) void {
        tiger.assert(len > 0 and rel + len <= info.byte_length);
        if (self.bounce == null) {
            const src: [*]const u8 = @ptrCast(self.data_ptr(info));
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(dst_dev, @ptrCast(src + @as(usize, @intCast(rel))), @intCast(len)));
            return;
        }
        const start: u64 = self.data_base_offset + info.data_offset + rel;
        const dst: [*]u8 = @ptrCast(dst_dev);
        var done: u64 = 0;
        while (done < len) {
            const n: u64 = @min(len - done, chunk_bytes_max);
            const src = self.read_chunk(start + done, n);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(dst + @as(usize, @intCast(done))), @ptrCast(src.ptr), @intCast(n)));
            done += n;
        }
    }

    pub fn close_file(self: *SafetensorsFile) void {
        if (self.mapped_ptr) |base| {
            tiger.assert(self.file_bytes > 0);
            _ = tiger.munmap(@ptrCast(@constCast(base)), @intCast(self.file_bytes));
            self.mapped_ptr = null;
        }
        if (self.file_descriptor >= 0) {
            _ = tiger.close(self.file_descriptor);
            self.file_descriptor = -1;
        }
        if (self.direct_fd >= 0) {
            _ = tiger.close(self.direct_fd);
            self.direct_fd = -1;
        }
        if (self.bounce) |b| {
            _ = cuda.squid_cuda_free_host(@ptrCast(b));
            self.bounce = null;
        }
        self.file_bytes = 0;
        self.tensor_count = 0;
    }

    pub fn lookup_optional(self: *const SafetensorsFile, name: []const u8) ?*const TensorInfo {
        tiger.assert(self.mapped_ptr != null);
        tiger.assert(name.len > 0);
        const header_slice: []const u8 = self.mapped_ptr.?[8 .. 8 + @as(usize, @intCast(self.header_bytes))];
        const hash: u64 = tiger.hash_bytes_u64(name);
        const mask: u32 = hash_slot_count - 1;
        var slot: u32 = @intCast(hash & mask);

        var probe: u32 = 0;
        while (probe < hash_slot_count) : (probe += 1) {
            const index: u16 = self.hash_table[slot];
            if (index == empty_slot) break;
            tiger.assert(index < self.tensor_count);
            const info: *const TensorInfo = &self.tensors[index];
            const entry_name = header_slice[info.name_offset .. info.name_offset + info.name_length];
            if (std.mem.eql(u8, entry_name, name)) {
                return info;
            }
            slot = (slot + 1) & mask;
        }
        return null;
    }

    pub fn lookup(self: *const SafetensorsFile, name: []const u8) *const TensorInfo {
        if (self.lookup_optional(name)) |info| return info;
        tiger.print_stderr("[safetensors] Missing tensor: {s}\n", .{name});
        @panic("Safetensors tensor not found");
    }

    pub fn data_slice(self: *const SafetensorsFile, info: *const TensorInfo) []const u8 {
        tiger.assert(self.mapped_ptr != null);
        const start: u64 = self.data_base_offset + info.data_offset;
        const end: u64 = start + info.byte_length;
        tiger.assert(end >= start);
        tiger.assert(end <= self.file_bytes);
        return self.mapped_ptr.?[@intCast(start)..@intCast(end)];
    }

    pub fn data_ptr(self: *const SafetensorsFile, info: *const TensorInfo) *const anyopaque {
        const bytes = self.data_slice(info);
        tiger.assert(bytes.len == info.byte_length);
        return @ptrCast(bytes.ptr);
    }

    fn insert_tensor(self: *SafetensorsFile, header: []const u8, info: TensorInfo) void {
        tiger.assert(self.tensor_count < tensor_count_max);
        tiger.assert(info.name_length > 0);
        const index: u16 = @intCast(self.tensor_count);
        self.tensors[index] = info;
        self.tensor_count += 1;

        const name = header[info.name_offset .. info.name_offset + info.name_length];
        const mask: u32 = hash_slot_count - 1;
        var slot: u32 = @intCast(tiger.hash_bytes_u64(name) & mask);
        var probe: u32 = 0;
        while (probe < hash_slot_count) : (probe += 1) {
            if (self.hash_table[slot] == empty_slot) {
                self.hash_table[slot] = index;
                return;
            }
            slot = (slot + 1) & mask;
        }
        @panic("Safetensors hash table full");
    }

    fn parse_header(self: *SafetensorsFile, header: []const u8) void {
        tiger.assert(header.len >= 2);
        var pos: usize = 0;
        skip_ws(header, &pos);
        tiger.assert(pos < header.len and header[pos] == '{');
        pos += 1;

        var entry_iteration: u32 = 0;
        const max_entries: u32 = tensor_count_max + 16;
        while (pos < header.len and entry_iteration < max_entries) : (entry_iteration += 1) {
            skip_ws(header, &pos);
            if (pos < header.len and header[pos] == '}') break;

            const key_span = parse_raw_json_string(header, &pos);
            skip_ws(header, &pos);
            tiger.assert(pos < header.len and header[pos] == ':');
            pos += 1;
            skip_ws(header, &pos);

            const key_name = header[key_span.start .. key_span.start + key_span.length];
            if (std.mem.eql(u8, key_name, "__metadata__")) {
                skip_json_value(header, &pos);
            } else {
                tiger.assert(pos < header.len and header[pos] == '{');
                pos += 1;
                var start_off: u64 = 0;
                var end_off: u64 = 0;
                var saw_offsets: bool = false;

                var field_iter: u32 = 0;
                while (pos < header.len and field_iter < 32) : (field_iter += 1) {
                    skip_ws(header, &pos);
                    if (pos < header.len and header[pos] == '}') {
                        pos += 1;
                        break;
                    }
                    const fk_span = parse_raw_json_string(header, &pos);
                    const fk_name = header[fk_span.start .. fk_span.start + fk_span.length];
                    skip_ws(header, &pos);
                    tiger.assert(pos < header.len and header[pos] == ':');
                    pos += 1;
                    skip_ws(header, &pos);

                    if (std.mem.eql(u8, fk_name, "data_offsets")) {
                        tiger.assert(pos < header.len and header[pos] == '[');
                        pos += 1;
                        start_off = parse_u64(header, &pos);
                        skip_ws(header, &pos);
                        tiger.assert(pos < header.len and header[pos] == ',');
                        pos += 1;
                        end_off = parse_u64(header, &pos);
                        skip_ws(header, &pos);
                        tiger.assert(pos < header.len and header[pos] == ']');
                        pos += 1;
                        saw_offsets = true;
                    } else {
                        skip_json_value(header, &pos);
                    }
                    skip_ws(header, &pos);
                    if (pos < header.len and header[pos] == ',') pos += 1;
                }

                tiger.assert(saw_offsets);
                tiger.assert(end_off >= start_off);
                tiger.assert(self.data_base_offset + end_off <= self.file_bytes);

                self.insert_tensor(header, .{
                    .name_offset = @intCast(key_span.start),
                    .name_length = @intCast(key_span.length),
                    .data_offset = start_off,
                    .byte_length = end_off - start_off,
                });
            }

            skip_ws(header, &pos);
            if (pos < header.len and header[pos] == ',') pos += 1;
        }
    }
};

const Span = struct {
    start: usize,
    length: usize,
};

inline fn skip_ws(data: []const u8, pos: *usize) void {
    while (pos.* < data.len) {
        const c = data[pos.*];
        if (c == ' ' or c == '\t' or c == '\n' or c == '\r') {
            pos.* += 1;
        } else {
            break;
        }
    }
}

fn parse_raw_json_string(data: []const u8, pos: *usize) Span {
    skip_ws(data, pos);
    tiger.assert(pos.* < data.len and data[pos.*] == '"');
    pos.* += 1;
    const start = pos.*;
    while (pos.* < data.len) {
        const c = data[pos.*];
        if (c == '\\') {
            pos.* += 2;
            continue;
        }
        if (c == '"') {
            const end = pos.*;
            pos.* += 1;
            return .{ .start = start, .length = end - start };
        }
        pos.* += 1;
    }
    @panic("Unterminated JSON string in safetensors header");
}

fn parse_u64(data: []const u8, pos: *usize) u64 {
    skip_ws(data, pos);
    tiger.assert(pos.* < data.len);
    var value: u64 = 0;
    var digits: u32 = 0;
    while (pos.* < data.len and digits < 20) : (digits += 1) {
        const c = data[pos.*];
        if (c < '0' or c > '9') break;
        value = value * 10 + @as(u64, c - '0');
        pos.* += 1;
    }
    tiger.assert(digits > 0);
    return value;
}

fn skip_json_value(data: []const u8, pos: *usize) void {
    skip_ws(data, pos);
    tiger.assert(pos.* < data.len);
    const first = data[pos.*];
    if (first == '"') {
        _ = parse_raw_json_string(data, pos);
        return;
    }
    if (first == '{' or first == '[') {
        var depth: i32 = 0;
        while (pos.* < data.len) {
            const c = data[pos.*];
            if (c == '"') {
                _ = parse_raw_json_string(data, pos);
                continue;
            }
            if (c == '{' or c == '[') {
                depth += 1;
            } else if (c == '}' or c == ']') {
                depth -= 1;
                pos.* += 1;
                if (depth == 0) return;
                continue;
            }
            pos.* += 1;
        }
        @panic("Unterminated JSON container in safetensors header");
    }
    while (pos.* < data.len) {
        const c = data[pos.*];
        if (c == ',' or c == '}' or c == ']' or c == ' ' or c == '\t' or c == '\n' or c == '\r') break;
        pos.* += 1;
    }
}
