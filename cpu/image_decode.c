// Image decoding for vision requests, from memory, with the system libpng (simplified API), libjpeg-turbo
// (the TurboJPEG 2 API, which 2.x and 3.x both export) and libwebp. Output is 8-bit RGB, row-major, no padding. Transparent images are composited onto
// white, as Hugging Face processors do (transformers' convert_to_rgb).

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <png.h>
#include <turbojpeg.h>
#include <webp/decode.h>

#define SQUID_IMAGE_SIDE_MAX 16384
#define SQUID_IMAGE_PIXELS_MAX (64u * 1024u * 1024u)

enum { FMT_UNKNOWN = 0, FMT_PNG = 1, FMT_JPEG = 2, FMT_WEBP = 3 };

static int sniff(const uint8_t* b, size_t n) {
    if (n >= 8 && memcmp(b, "\x89PNG\r\n\x1a\n", 8) == 0) return FMT_PNG;
    if (n >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF) return FMT_JPEG;
    if (n >= 12 && memcmp(b, "RIFF", 4) == 0 && memcmp(b + 8, "WEBP", 4) == 0) return FMT_WEBP;
    return FMT_UNKNOWN;
}

static int dims_ok(int w, int h) {
    return w > 0 && h > 0 && w <= SQUID_IMAGE_SIDE_MAX && h <= SQUID_IMAGE_SIDE_MAX &&
           (uint64_t)w * (uint64_t)h <= SQUID_IMAGE_PIXELS_MAX;
}

// RGBA (straight alpha) -> RGB over white, in place-safe forward order (dst may alias src).
static void composite_white(const uint8_t* rgba, uint8_t* rgb, size_t pixels) {
    for (size_t i = 0; i < pixels; ++i) {
        const unsigned a = rgba[4 * i + 3];
        for (int c = 0; c < 3; ++c) {
            const unsigned v = rgba[4 * i + c];
            rgb[3 * i + c] = (uint8_t)((v * a + 255u * (255u - a) + 127u) / 255u);
        }
    }
}

// Returns the format (1 PNG, 2 JPEG, 3 WebP) and the size, 0 for an unsupported format, -1 for a broken or
// oversized image.
int squid_image_probe(const uint8_t* buf, size_t len, int* w, int* h) {
    const int fmt = sniff(buf, len);
    *w = 0;
    *h = 0;
    if (fmt == FMT_PNG) {
        png_image img;
        memset(&img, 0, sizeof img);
        img.version = PNG_IMAGE_VERSION;
        if (!png_image_begin_read_from_memory(&img, buf, len)) return -1;
        *w = (int)img.width;
        *h = (int)img.height;
        png_image_free(&img);
    } else if (fmt == FMT_JPEG) {
        if (len > 0xFFFFFFFFu) return -1;
        tjhandle tj = tjInitDecompress();
        if (!tj) return -1;
        int subsamp = 0, colorspace = 0;
        const int ok = tjDecompressHeader3(tj, buf, (unsigned long)len, w, h, &subsamp, &colorspace) == 0;
        tjDestroy(tj);
        if (!ok) return -1;
    } else if (fmt == FMT_WEBP) {
        if (!WebPGetInfo(buf, len, w, h)) return -1;
    } else {
        return 0;
    }
    return dims_ok(*w, *h) ? fmt : -1;
}

// Decodes to malloc'd RGB8 [h][w][3] (free with squid_image_free), or NULL.
uint8_t* squid_image_decode_rgb(const uint8_t* buf, size_t len, int* w, int* h) {
    const int fmt = squid_image_probe(buf, len, w, h);
    if (fmt <= 0) return NULL;
    const size_t pixels = (size_t)(*w) * (size_t)(*h);
    if (fmt == FMT_PNG) {
        png_image img;
        memset(&img, 0, sizeof img);
        img.version = PNG_IMAGE_VERSION;
        if (!png_image_begin_read_from_memory(&img, buf, len)) return NULL;
        img.format = PNG_FORMAT_RGBA;
        uint8_t* px = (uint8_t*)malloc(pixels * 4);
        if (!px) {
            png_image_free(&img);
            return NULL;
        }
        if (!png_image_finish_read(&img, NULL, px, 0, NULL)) {
            png_image_free(&img);
            free(px);
            return NULL;
        }
        composite_white(px, px, pixels);
        return px;  // realloc-free: the RGB data is a prefix of the buffer
    }
    if (fmt == FMT_JPEG) {
        tjhandle tj = tjInitDecompress();
        if (!tj) return NULL;
        uint8_t* px = (uint8_t*)malloc(pixels * 3);
        if (px && tjDecompress2(tj, buf, (unsigned long)len, px, *w, 0, *h, TJPF_RGB, TJFLAG_ACCURATEDCT) != 0) {
            free(px);
            px = NULL;
        }
        tjDestroy(tj);
        return px;
    }
    // WebP
    uint8_t* rgba = WebPDecodeRGBA(buf, len, w, h);
    if (!rgba) return NULL;
    uint8_t* px = (uint8_t*)malloc(pixels * 3);
    if (px) composite_white(rgba, px, pixels);
    WebPFree(rgba);
    return px;
}

void squid_image_free(uint8_t* p) { free(p); }
