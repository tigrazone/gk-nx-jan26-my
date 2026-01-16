/**
 * wuffs_image.h - stb_image-like wrappers for Wuffs (PNG, JPEG)
 * 
 * This file provides analogs for stbi_info_from_memory, stbi_is_16_bit_from_memory,
 * stbi_load_from_memory, and stbi_load_16_from_memory using the Wuffs library.
 * 
 * To include the implementation, define WUFFS_IMAGE_IMPLEMENTATION in one C/C++ file.
 */

#ifndef WUFFS_IMAGE_H
#define WUFFS_IMAGE_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Identical parameters to stb_image functions */

int wuffs_info_from_memory(uint8_t const* buffer, int len, int* x, int* y, int* comp);
int wuffs_is_16_bit_from_memory(uint8_t const* buffer, int len);
uint16_t* wuffs_load_16_from_memory(uint8_t const* buffer, int len, int* x, int* y, int* channels_in_file, int desired_channels);
uint8_t* wuffs_load_from_memory(uint8_t const* buffer, int len, int* x, int* y, int* channels_in_file, int desired_channels);

#ifdef __cplusplus
}
#endif

#endif /* WUFFS_IMAGE_H */

#ifdef WUFFS_IMAGE_IMPLEMENTATION

#include <string.h>

/* We need some specific Wuffs defines to enable decoders */
#define WUFFS_CONFIG__MODULES
#define WUFFS_CONFIG__MODULE__BASE
#define WUFFS_CONFIG__MODULE__PNG
#define WUFFS_CONFIG__MODULE__JPEG
#define WUFFS_CONFIG__MODULE__ADLER32
#define WUFFS_CONFIG__MODULE__CRC32
#define WUFFS_CONFIG__MODULE__DEFLATE
#define WUFFS_CONFIG__MODULE__ZLIB
#define WUFFS_CONFIG__MODULE__AUX__BASE
#define WUFFS_CONFIG__MODULE__AUX__IMAGE

/* Include Wuffs implementation if not already project-wide */
#ifndef WUFFS_IMPLEMENTATION
#define WUFFS_IMPLEMENTATION
#endif

#include "wuffs-v0.4.c"

typedef enum {
    WUFFS_IMAGE_FORMAT_UNKNOWN = 0,
    WUFFS_IMAGE_FORMAT_PNG,
    WUFFS_IMAGE_FORMAT_JPEG
} wuffs_image_format;

static wuffs_image_format wuffs_image_detect_format(uint8_t const* buffer, int len) {
    if (len >= 8 && memcmp(buffer, "\x89PNG\x0D\x0A\x1A\x0A", 8) == 0) {
        return WUFFS_IMAGE_FORMAT_PNG;
    }
    if (len >= 2 && memcmp(buffer, "\xFF\xD8", 2) == 0) {
        return WUFFS_IMAGE_FORMAT_JPEG;
    }
    return WUFFS_IMAGE_FORMAT_UNKNOWN;
}

static wuffs_base__status wuffs_image_decode_config(wuffs_image_format format,
                                                 wuffs_base__image_config* config,
                                                 wuffs_base__io_buffer* io_buf,
                                                 void* decoder_ptr) {
    if (format == WUFFS_IMAGE_FORMAT_PNG) {
        return wuffs_png__decoder__decode_image_config((wuffs_png__decoder*)decoder_ptr, config, io_buf);
    } else if (format == WUFFS_IMAGE_FORMAT_JPEG) {
        return wuffs_jpeg__decoder__decode_image_config((wuffs_jpeg__decoder*)decoder_ptr, config, io_buf);
    }
    return wuffs_base__make_status(wuffs_base__error__unsupported_method);
}

int wuffs_info_from_memory(uint8_t const* buffer, int len, int* x, int* y, int* comp) {
    wuffs_image_format format = wuffs_image_detect_format(buffer, len);
    if (format == WUFFS_IMAGE_FORMAT_UNKNOWN) return 0;

    wuffs_base__io_buffer io_buf = wuffs_base__ptr_u8__reader((uint8_t*)buffer, (size_t)len, true);
    wuffs_base__image_config config;
    memset(&config, 0, sizeof(config));

    union decoder_union {
        wuffs_png__decoder png;
        wuffs_jpeg__decoder jpeg;
    };
    union decoder_union* dec = (union decoder_union*)malloc(sizeof(union decoder_union));
    if (!dec) return 0;

    wuffs_base__status status;
    if (format == WUFFS_IMAGE_FORMAT_PNG) {
        status = wuffs_png__decoder__initialize(&dec->png, sizeof(dec->png), WUFFS_VERSION, 0);
    } else if (format == WUFFS_IMAGE_FORMAT_JPEG) {
        status = wuffs_jpeg__decoder__initialize(&dec->jpeg, sizeof(dec->jpeg), WUFFS_VERSION, 0);
    } else {
        free(dec);
        return 0;
    }

    if (!wuffs_base__status__is_ok(&status)) { free(dec); return 0; }

    status = wuffs_image_decode_config(format, &config, &io_buf, dec);
    if (!wuffs_base__status__is_ok(&status)) { free(dec); return 0; }

    if (x) *x = (int)wuffs_base__pixel_config__width(&config.pixcfg);
    if (y) *y = (int)wuffs_base__pixel_config__height(&config.pixcfg);
    if (comp) {
        wuffs_base__pixel_format pfmt = wuffs_base__pixel_config__pixel_format(&config.pixcfg);
        /* Simple heuristic for channels */
        if (wuffs_base__pixel_format__coloration(&pfmt) == WUFFS_BASE__PIXEL_COLORATION__GRAY) *comp = 1;
        else *comp = 4; /* Wuffs often defaults to RGBA for PNG */
    }

    free(dec);
    return 1;
}

int wuffs_is_16_bit_from_memory(uint8_t const* buffer, int len) {
    wuffs_image_format format = wuffs_image_detect_format(buffer, len);
    if (format != WUFFS_IMAGE_FORMAT_PNG) return 0; /* Only PNG supports 16-bit in this context */

    wuffs_base__io_buffer io_buf = wuffs_base__ptr_u8__reader((uint8_t*)buffer, (size_t)len, true);
    wuffs_base__image_config config;
    memset(&config, 0, sizeof(config));
    wuffs_png__decoder* dec = (wuffs_png__decoder*)malloc(sizeof(wuffs_png__decoder));
    if (!dec) return 0;

    wuffs_base__status status = wuffs_png__decoder__initialize(dec, sizeof(*dec), WUFFS_VERSION, 0);
    if (!wuffs_base__status__is_ok(&status)) { free(dec); return 0; }
    status = wuffs_png__decoder__decode_image_config(dec, &config, &io_buf);
    if (!wuffs_base__status__is_ok(&status)) { free(dec); return 0; }
    
    free(dec);

    wuffs_base__pixel_format pfmt = wuffs_base__pixel_config__pixel_format(&config.pixcfg);
    return (pfmt.repr & 0x0F) >= 0x0B;
}

static void* wuffs_image_load_generic(uint8_t const* buffer, int len, int* x, int* y, int* channels_in_file, int desired_channels, int is_16bit) {
    wuffs_image_format format = wuffs_image_detect_format(buffer, len);
    if (format == WUFFS_IMAGE_FORMAT_UNKNOWN) return NULL;

    wuffs_base__io_buffer io_buf = wuffs_base__ptr_u8__reader((uint8_t*)buffer, (size_t)len, true);
    wuffs_base__image_config config;
    memset(&config, 0, sizeof(config));

    union decoder_union {
        wuffs_png__decoder png;
        wuffs_jpeg__decoder jpeg;
    };
    union decoder_union* dec = (union decoder_union*)malloc(sizeof(union decoder_union));
    if (!dec) return NULL;

    wuffs_base__status status;
    if (format == WUFFS_IMAGE_FORMAT_PNG) {
        status = wuffs_png__decoder__initialize(&dec->png, sizeof(dec->png), WUFFS_VERSION, 0);
    } else if (format == WUFFS_IMAGE_FORMAT_JPEG) {
        status = wuffs_jpeg__decoder__initialize(&dec->jpeg, sizeof(dec->jpeg), WUFFS_VERSION, 0);
    } else {
        free(dec);
        return NULL;
    }

    if (!wuffs_base__status__is_ok(&status)) { free(dec); return NULL; }

    status = wuffs_image_decode_config(format, &config, &io_buf, dec);
    if (!wuffs_base__status__is_ok(&status)) { free(dec); return NULL; }

    uint32_t width = wuffs_base__pixel_config__width(&config.pixcfg);
    uint32_t height = wuffs_base__pixel_config__height(&config.pixcfg);
    if (x) *x = (int)width;
    if (y) *y = (int)height;

    /* Determine input channels */
    wuffs_base__pixel_format src_pfmt = wuffs_base__pixel_config__pixel_format(&config.pixcfg);
    uint32_t src_coloration = wuffs_base__pixel_format__coloration(&src_pfmt);
    int src_channels = (src_coloration == WUFFS_BASE__PIXEL_COLORATION__GRAY) ? 1 : 
                       (src_coloration == WUFFS_BASE__PIXEL_COLORATION__RICH) ? 3 : 4; 
    
    /* Wuffs PNG often defaults to 4 even if 3. Check for OPAQUE. */
    if (src_channels == 3 && wuffs_base__pixel_format__transparency(&src_pfmt) != WUFFS_BASE__PIXEL_ALPHA_TRANSPARENCY__OPAQUE) {
         src_channels = 4;
    }
    if (channels_in_file) *channels_in_file = src_channels;

    int n = desired_channels ? desired_channels : src_channels;
    
    /* Set up pixel configuration for decoding */
    /* Set up pixel configuration for decoding */
    uint32_t dst_pfmt = WUFFS_BASE__PIXEL_FORMAT__INVALID;
    if (is_16bit) {
        switch (n) {
            case 1:  dst_pfmt = WUFFS_BASE__PIXEL_FORMAT__Y_16LE; break;
            case 2:  dst_pfmt = 0x210000BB; break; /* YA_NONPREMUL_2X16LE */
            case 3:  dst_pfmt = WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL_4X16LE; n = 4; break; /* Wuffs doesn't support RGB 16-bit target well, force RGBA */
            default: dst_pfmt = WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL_4X16LE; break;
        }
    } else {
        switch (n) {
            case 1:  dst_pfmt = WUFFS_BASE__PIXEL_FORMAT__Y; break;
            case 2:  dst_pfmt = WUFFS_BASE__PIXEL_FORMAT__YA_NONPREMUL; break;
            case 3:  dst_pfmt = WUFFS_BASE__PIXEL_FORMAT__RGB; break; // Or BGR? STB usually produces RGB.
            default: dst_pfmt = WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL; break;
        }
    }

    wuffs_base__pixel_config__set(&config.pixcfg, dst_pfmt, WUFFS_BASE__PIXEL_SUBSAMPLING__NONE, width, height);

    size_t bytes_per_pixel = (size_t)n * (is_16bit ? 2 : 1);
    size_t pixel_buf_size = (size_t)width * (size_t)height * bytes_per_pixel;
    void* pixels = malloc(pixel_buf_size);
    if (!pixels) return NULL;

    /* Allocate usage of work buffer */
    wuffs_base__slice_u8 workbuf = {0};
    wuffs_base__range_ii_u64 workbuf_len_range = {0};
    if (format == WUFFS_IMAGE_FORMAT_PNG) {
        workbuf_len_range = wuffs_png__decoder__workbuf_len(&dec->png);
    } else if (format == WUFFS_IMAGE_FORMAT_JPEG) {
        workbuf_len_range = wuffs_jpeg__decoder__workbuf_len(&dec->jpeg);
    }
    
    if (workbuf_len_range.max_incl > 0) {
        workbuf.len = (size_t)workbuf_len_range.max_incl;
        workbuf.ptr = (uint8_t*)malloc(workbuf.len);
        if (!workbuf.ptr) {
            free(pixels);
            return NULL;
        }
    }
    
    wuffs_base__pixel_buffer pb;
    memset(&pb, 0, sizeof(pb));
    status = wuffs_base__pixel_buffer__set_from_slice(&pb, &config.pixcfg,
                                                   wuffs_base__make_slice_u8((uint8_t*)pixels, pixel_buf_size));
    if (!wuffs_base__status__is_ok(&status)) {
        free(pixels);
        return NULL;
    }


    if (format == WUFFS_IMAGE_FORMAT_PNG) {
        status = wuffs_png__decoder__decode_frame(&dec->png, &pb, &io_buf, WUFFS_BASE__PIXEL_BLEND__SRC, workbuf, NULL);
    } else if (format == WUFFS_IMAGE_FORMAT_JPEG) {
        status = wuffs_jpeg__decoder__decode_frame(&dec->jpeg, &pb, &io_buf, WUFFS_BASE__PIXEL_BLEND__SRC, workbuf, NULL);
    }
    
    if (workbuf.ptr) free(workbuf.ptr);

    if (!wuffs_base__status__is_ok(&status)) {
        free(pixels);
        return NULL;
    }

    return pixels;
}

uint8_t* wuffs_load_from_memory(uint8_t const* buffer, int len, int* x, int* y, int* channels_in_file, int desired_channels) {
    return (uint8_t*)wuffs_image_load_generic(buffer, len, x, y, channels_in_file, desired_channels, 0);
}

uint16_t* wuffs_load_16_from_memory(uint8_t const* buffer, int len, int* x, int* y, int* channels_in_file, int desired_channels) {
    return (uint16_t*)wuffs_image_load_generic(buffer, len, x, y, channels_in_file, desired_channels, 1);
}

#endif /* WUFFS_IMAGE_IMPLEMENTATION */
