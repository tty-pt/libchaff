#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <ttypt/noise.h>
#include <ttypt/chaff.h>
#include <ttypt/iso.h>

ISO_EXPORT(int32_t, noise_wasm_point_1d, int32_t x, uint32_t seed, uint32_t shift) {
    return noise_point_1d((int16_t)x, seed, (uint8_t)shift);
}

ISO_EXPORT(int32_t, noise_wasm_point_2d, int32_t x, int32_t y, uint32_t seed, uint32_t shift) {
    return noise_point_2d((int16_t)x, (int16_t)y, seed, (uint8_t)shift);
}

ISO_EXPORT(int32_t, noise_wasm_point_3d, int32_t x, int32_t y, int32_t z, uint32_t seed, uint32_t shift) {
    return noise_point_3d((int16_t)x, (int16_t)y, (int16_t)z, seed, (uint8_t)shift);
}

ISO_EXPORT(int32_t, noise_wasm_point_4d, int32_t x, int32_t y, int32_t z, int32_t w, uint32_t seed, uint32_t shift) {
    int16_t p[4] = { (int16_t)x, (int16_t)y, (int16_t)z, (int16_t)w };
    return noise_point_n(p, seed, (uint8_t)shift, 4);
}

ISO_EXPORT(int32_t, noise_wasm_point_n, const int16_t *coords, uint32_t seed, uint32_t shift, uint32_t dim) {
    if (!coords || dim == 0 || dim > 4) return 128;
    return noise_point_n(coords, seed, (uint8_t)shift, (uint8_t)dim);
}

ISO_EXPORT(void, noise_wasm_octaves, uint32_t *m, const int16_t *s, size_t oct_n, const uint32_t *oct_tuples, uint32_t seed, uint32_t cy, uint32_t dim) {
    if (!m || !s || !oct_tuples || oct_n == 0 || dim == 0) return;
    octave_t oct[NOISE_MAX_OCTAVES];
    size_t count = oct_n > NOISE_MAX_OCTAVES ? NOISE_MAX_OCTAVES : oct_n;
    for (size_t i = 0; i < count; i++) {
        oct[i].x = oct_tuples[i * 2];
        oct[i].w = oct_tuples[i * 2 + 1];
    }
    noise_oct(m, (int16_t *)s, count, oct, seed, cy, (uint8_t)dim);
}

/* Canonical chaff_wasm_* exports */
ISO_EXPORT(int32_t, chaff_wasm_point_1d, int32_t x, uint32_t seed, uint32_t shift) {
    return chaff_point_1d((int16_t)x, seed, (uint8_t)shift);
}

ISO_EXPORT(int32_t, chaff_wasm_point_2d, int32_t x, int32_t y, uint32_t seed, uint32_t shift) {
    return chaff_point_2d((int16_t)x, (int16_t)y, seed, (uint8_t)shift);
}

ISO_EXPORT(int32_t, chaff_wasm_point_3d, int32_t x, int32_t y, int32_t z, uint32_t seed, uint32_t shift) {
    return chaff_point_3d((int16_t)x, (int16_t)y, (int16_t)z, seed, (uint8_t)shift);
}

ISO_EXPORT(int32_t, chaff_wasm_point_4d, int32_t x, int32_t y, int32_t z, int32_t w, uint32_t seed, uint32_t shift) {
    return chaff_point_4d((int16_t)x, (int16_t)y, (int16_t)z, (int16_t)w, seed, (uint8_t)shift);
}

ISO_EXPORT(uint32_t, chaff_wasm_wpoint_periodic_2d, int32_t x, int32_t y, uint32_t shift, uint32_t period_shift, uint32_t vshift, uint32_t seed) {
    return chaff_wpoint_periodic_2d(x, y, (uint8_t)shift, (uint8_t)period_shift, (uint8_t)vshift, seed);
}

ISO_EXPORT(int32_t, chaff_wasm_fbm_rotated_2d_q12_4,
           int32_t x, int32_t y, uint32_t min_shift, uint32_t max_shift,
           uint32_t ridged_min_shift, uint32_t amp_min_shift, int32_t base_amp,
           uint32_t seed, uint32_t seed_salt, int32_t cos_q15, int32_t sin_q15) {
    return chaff_fbm_rotated_2d_q12_4(x, y, (uint8_t)min_shift, (uint8_t)max_shift,
                                      (uint8_t)ridged_min_shift, (uint8_t)amp_min_shift, (int16_t)base_amp,
                                      seed, seed_salt, cos_q15, sin_q15);
}

ISO_EXPORT(void, chaff_wasm_sample_grad_2d,
           uint32_t seed, uint32_t shift, int32_t amp, uint32_t seed_mode, uint32_t contrast,
           int32_t wx, int32_t wy, int32_t *out_val_gx_gy) {
    if (!out_val_gx_gy) return;
    chaff_layer_t lay = {
        .shift = (uint8_t)shift,
        .vshift = CHAFF_DEFAULT_VSHIFT,
        .amp = (int16_t)amp,
        .seed_mode = (uint8_t)seed_mode,
        .skew = { 0, 0 },
        .contrast = (uint16_t)contrast
    };
    chaff_field_t field = {
        .seed = seed,
        .dim = 2,
        .flags = CHAFF_FLAG_FADE,
        .nlayers = 1,
        .layers = &lay
    };
    chaff_sample_2d_t s = chaff_sample_grad_2d(&field, (int16_t)wx, (int16_t)wy);
    out_val_gx_gy[0] = s.val;
    out_val_gx_gy[1] = s.grad_x;
    out_val_gx_gy[2] = s.grad_y;
}
