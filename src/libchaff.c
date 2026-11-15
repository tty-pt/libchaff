#include <stddef.h>
#include <stdint.h>
#include <ttypt/noise.h>
#include <ttypt/chaff.h>

int ttypt_noise_point_1d(int16_t x, uint32_t seed, uint8_t shift) {
    return noise_point_1d(x, seed, shift);
}

int ttypt_noise_point_2d(int16_t x, int16_t y, uint32_t seed, uint8_t shift) {
    return noise_point_2d(x, y, seed, shift);
}

int ttypt_noise_point_3d(int16_t x, int16_t y, int16_t z, uint32_t seed, uint8_t shift) {
    return noise_point_3d(x, y, z, seed, shift);
}

int ttypt_noise_point_4d(int16_t x, int16_t y, int16_t z, int16_t w, uint32_t seed, uint8_t shift) {
    return noise_point_4d(x, y, z, w, seed, shift);
}

int ttypt_noise_point_n(const int16_t *p, uint32_t seed, uint8_t shift, uint8_t dim) {
    return noise_point_n(p, seed, shift, dim);
}

void ttypt_noise_oct(uint32_t *m, int16_t *s, size_t oct_n, octave_t *oct, unsigned seed, unsigned cy, uint8_t dim) {
    noise_oct(m, s, oct_n, oct, seed, cy, dim);
}

/* Wrapping original-algorithm sampler (pure CPU, integer-only) */
uint32_t ttypt_noise_wpoint_n(const int16_t *p, unsigned x, unsigned w, unsigned seed, uint8_t dim, unsigned flags) {
    return noise_wpoint_n(p, x, w, seed, dim, flags);
}

uint32_t ttypt_noise_wpoint_c(noise_wcache_t *c, const int16_t *p, unsigned x, unsigned w, unsigned seed, uint8_t dim, unsigned flags) {
    return noise_wpoint_c(c, p, x, w, seed, dim, flags);
}

void ttypt_noise_wrow(uint32_t *out, unsigned n, const int16_t *p0, int16_t dstep, unsigned x, unsigned w, unsigned seed, uint8_t dim, unsigned flags, noise_wcache_t *c) {
    noise_wrow(out, n, p0, dstep, x, w, seed, dim, flags, c);
}

void ttypt_noise_wplane(uint32_t *out, unsigned n, unsigned rows, size_t stride, const int16_t *p0, unsigned x, unsigned w, unsigned seed, uint8_t dim, unsigned flags, noise_wcache_t *c) {
    noise_wplane(out, n, rows, stride, p0, x, w, seed, dim, flags, c);
}

/* Declarative Cascade & Universal Accessors */
void ttypt_noise_matrix_fill_2d(const noise_cascade_t *cfg, int16_t origin_x, int16_t origin_y, uint8_t size_shift, int32_t *out_matrix) {
    noise_matrix_fill_2d(cfg, origin_x, origin_y, size_shift, out_matrix);
}

void ttypt_noise_scanline_init(noise_scanline_t *scan, const noise_cascade_t *cfg, uint8_t dim) {
    noise_scanline_init(scan, cfg, dim);
}

void ttypt_noise_scanline_eval_row(noise_scanline_t *scan, int16_t start_x, int16_t y, int16_t z, size_t count, int32_t *out_row) {
    noise_scanline_eval_row(scan, start_x, y, z, count, out_row);
}

int32_t ttypt_noise_sample_2d(const noise_cascade_t *cfg, int16_t x, int16_t y) {
    return noise_sample_2d(cfg, x, y);
}

int32_t ttypt_noise_sample_3d(const noise_cascade_t *cfg, int16_t x, int16_t y, int16_t z) {
    return noise_sample_3d(cfg, x, y, z);
}

int32_t ttypt_noise_sample_4d(const noise_cascade_t *cfg, int16_t x, int16_t y, int16_t z, int16_t w) {
    return noise_sample_4d(cfg, x, y, z, w);
}

/* Canonical chaff_ symbols */
int chaff_point_1d_sym(int16_t x, uint32_t seed, uint8_t shift) {
    return noise_point_1d(x, seed, shift);
}
int chaff_point_2d_sym(int16_t x, int16_t y, uint32_t seed, uint8_t shift) {
    return noise_point_2d(x, y, seed, shift);
}
int chaff_point_3d_sym(int16_t x, int16_t y, int16_t z, uint32_t seed, uint8_t shift) {
    return noise_point_3d(x, y, z, seed, shift);
}
int chaff_point_4d_sym(int16_t x, int16_t y, int16_t z, int16_t w, uint32_t seed, uint8_t shift) {
    return noise_point_4d(x, y, z, w, seed, shift);
}
int chaff_point_n_sym(const int16_t *p, uint32_t seed, uint8_t shift, uint8_t dim) {
    return noise_point_n(p, seed, shift, dim);
}
void chaff_oct_sym(uint32_t *m, int16_t *s, size_t oct_n, octave_t *oct, unsigned seed, unsigned cy, uint8_t dim) {
    chaff_oct(m, s, oct_n, oct, seed, cy, dim);
}
void chaff_matrix_fill_2d_sym(const noise_cascade_t *cfg, int16_t origin_x, int16_t origin_y, uint8_t size_shift, int32_t *out_matrix) {
    noise_matrix_fill_2d(cfg, origin_x, origin_y, size_shift, out_matrix);
}
void chaff_scanline_init_sym(noise_scanline_t *scan, const noise_cascade_t *cfg, uint8_t dim) {
    noise_scanline_init(scan, cfg, dim);
}
void chaff_scanline_eval_row_sym(noise_scanline_t *scan, int16_t start_x, int16_t y, int16_t z, size_t count, int32_t *out_row) {
    noise_scanline_eval_row(scan, start_x, y, z, count, out_row);
}
int32_t chaff_sample_2d_sym(const noise_cascade_t *cfg, int16_t x, int16_t y) {
    return noise_sample_2d(cfg, x, y);
}
int32_t chaff_sample_3d_sym(const noise_cascade_t *cfg, int16_t x, int16_t y, int16_t z) {
    return noise_sample_3d(cfg, x, y, z);
}
int32_t chaff_sample_4d_sym(const noise_cascade_t *cfg, int16_t x, int16_t y, int16_t z, int16_t w) {
    return noise_sample_4d(cfg, x, y, z, w);
}

/* Universal chaff_fill & LOD entry points */
void ttypt_chaff_fill(const chaff_field_t *field,
                      uint8_t layer_lo, uint8_t layer_hi,
                      const chaff_coord_t origin[],
                      const int32_t step_q16[],
                      const uint16_t n[],
                      int32_t *out, size_t stride, unsigned mode) {
    chaff_fill(field, layer_lo, layer_hi, origin, step_q16, n, out, stride, mode);
}

void chaff_fill_sym(const chaff_field_t *field,
                    uint8_t layer_lo, uint8_t layer_hi,
                    const chaff_coord_t origin[],
                    const int32_t step_q16[],
                    const uint16_t n[],
                    int32_t *out, size_t stride, unsigned mode) {
    chaff_fill(field, layer_lo, layer_hi, origin, step_q16, n, out, stride, mode);
}

void chaff_band_range_for_step_q16_sym(const chaff_field_t *field,
                                       uint8_t min_shift,
                                       const int32_t step_q16[],
                                       uint8_t *out_layer_lo,
                                       uint8_t *out_layer_hi) {
    chaff_band_range_for_step_q16(field, min_shift, step_q16, out_layer_lo, out_layer_hi);
}

void chaff_fill_lod_sym(const chaff_field_t *field,
                        uint8_t min_shift,
                        const chaff_coord_t origin[],
                        const int32_t step_q16[],
                        const uint16_t n[],
                        int32_t *out, size_t stride, unsigned mode) {
    chaff_fill_lod(field, min_shift, origin, step_q16, n, out, stride, mode);
}

chaff_sample_2d_t chaff_sample_grad_2d_sym(const chaff_field_t *field, int16_t wx, int16_t wy) {
    return chaff_sample_grad_2d(field, wx, wy);
}

chaff_sample_3d_t chaff_sample_grad_3d_sym(const chaff_field_t *field, int16_t x, int16_t y, int16_t z) {
    return chaff_sample_grad_3d(field, x, y, z);
}

chaff_sample_4d_t chaff_sample_grad_4d_sym(const chaff_field_t *field, int16_t x, int16_t y, int16_t z, int16_t w) {
    return chaff_sample_grad_4d(field, x, y, z, w);
}

void chaff_fill_slice_3d_sym(const chaff_field_t *field,
                             int16_t z,
                             int16_t origin_x, int16_t origin_y,
                             uint16_t width, uint16_t height,
                             int32_t *out_val,
                             int16_t *out_grad_x,
                             int16_t *out_grad_y) {
    chaff_fill_slice_3d(field, z, origin_x, origin_y, width, height, out_val, out_grad_x, out_grad_y);
}

void chaff_fill_slice_4d_sym(const chaff_field_t *field,
                             int16_t z, int16_t w,
                             int16_t origin_x, int16_t origin_y,
                             uint16_t width, uint16_t height,
                             int32_t *out_val,
                             int16_t *out_grad_x,
                             int16_t *out_grad_y,
                             int16_t *out_grad_z,
                             int16_t *out_grad_w) {
    chaff_fill_slice_4d(field, z, w, origin_x, origin_y, width, height,
                        out_val, out_grad_x, out_grad_y, out_grad_z, out_grad_w);
}

chaff_sample_2d_t chaff_sample_rugged_2d_ex_sym(const chaff_field_t *field, int16_t wx, int16_t wy,
                                                int32_t cos_q15, int32_t sin_q15) {
    return chaff_sample_rugged_2d_ex(field, wx, wy, cos_q15, sin_q15);
}

chaff_sample_2d_t chaff_sample_rugged_2d_sym(const chaff_field_t *field, int16_t wx, int16_t wy) {
    return chaff_sample_rugged_2d(field, wx, wy);
}

uint32_t chaff_wpoint_periodic_2d_sym(int32_t x, int32_t y, uint8_t shift, uint8_t period_shift,
                                      uint8_t vshift, uint32_t seed) {
    return chaff_wpoint_periodic_2d(x, y, shift, period_shift, vshift, seed);
}

uint32_t chaff_sample_periodic_2d_sym(int32_t rx, int32_t ry, uint8_t shift,
                                      uint8_t period_shift, uint8_t vshift, uint32_t seed) {
    return chaff_sample_periodic_2d(rx, ry, shift, period_shift, vshift, seed);
}

int32_t chaff_fbm_rotated_2d_ex_sym(int32_t x, int32_t y,
                                    uint8_t min_shift, uint8_t max_shift,
                                    uint8_t ridged_min_shift,
                                    uint8_t amp_min_shift, int16_t base_amp,
                                    uint32_t seed, uint32_t seed_salt,
                                    int32_t cos_q15, int32_t sin_q15,
                                    uint8_t vshift, unsigned fade_flags, uint8_t out_shift) {
    return chaff_fbm_rotated_2d_ex(x, y, min_shift, max_shift, ridged_min_shift,
                                   amp_min_shift, base_amp, seed, seed_salt,
                                   cos_q15, sin_q15, vshift, fade_flags, out_shift);
}

int32_t chaff_fbm_rotated_2d_q12_4_sym(int32_t x, int32_t y,
                                       uint8_t min_shift, uint8_t max_shift,
                                       uint8_t ridged_min_shift,
                                       uint8_t amp_min_shift, int16_t base_amp,
                                       uint32_t seed, uint32_t seed_salt,
                                       int32_t cos_q15, int32_t sin_q15) {
    return chaff_fbm_rotated_2d_q12_4(x, y, min_shift, max_shift, ridged_min_shift,
                                      amp_min_shift, base_amp, seed, seed_salt,
                                      cos_q15, sin_q15);
}

chaff_flow_2d_t chaff_sample_flow_2d_sym(const chaff_field_t *psi_field,
                                         const chaff_flow_desc_t *desc,
                                         int16_t x, int16_t y, int16_t t) {
    return chaff_sample_flow_2d(psi_field, desc, x, y, t);
}
