#ifndef TTYPT_CHAFF_H
#define TTYPT_CHAFF_H

/*
 * ttypt/chaff.h — Canonical N-Dimensional Procedural Noise, Lattice Stepper,
 * and Multi-Scale Band-Limited LOD API.
 *
 * LICENSE: BSD-2-Clause
 * Copyright (c) 2026, tty-pt
 *
 * Direct include for libchaff. Provides:
 *   1. Band-limited Nyquist multi-layer region generation (chaff_fill_lod).
 *   2. Analytical 2D/3D spatial gradients and 2D slice stepping.
 *   3. Parameterized 2D fixed-point domain rotation (chaff_rot2d_q15).
 *   4. Generic periodic 2D domain sampling (chaff_wpoint_periodic_2d) and
 *      monotonic fractal amplitude scaling (chaff_octave_amplitude_ex).
 *   5. Header-only GPU shader contract via <ttypt/chaff_glsl.h>.
 *
 * Pure C99, integer-only core, zero floating point, zero game/domain constants.
 */

#include <ttypt/noise.h>
#include <ttypt/chaff_glsl.h>

/* Scale fixed-point coordinate: multiplies or divides by 2^scale_power_of_two */
static inline chaff_coord_t
chaff_coord_scale(chaff_coord_t c, int scale_power_of_two)
{
	int64_t total = (((int64_t)c.cell << 16) | (uint32_t)c.frac);
	if (scale_power_of_two >= 0) {
		total <<= scale_power_of_two;
	} else {
		total >>= -scale_power_of_two;
	}
	chaff_coord_t out;
	out.cell = (int16_t)(total >> 16);
	out.frac = (uint16_t)(total & 0xffff);
	return out;
}

/* Derives the Nyquist minimum wavelength shift directly from a Q16.16 sampling stride.
 * By the Nyquist-Shannon sampling theorem, an octave with wavelength 2^shift requires
 * a sampling period Delta <= 2^(shift - 1), so min_shift = floor(log2(2 * max_step)). */
static inline uint8_t
chaff_nyquist_shift_q16(const int32_t step_q16[], uint8_t dim)
{
	if (!step_q16 || dim == 0) return 0;
	uint32_t max_step = 0;
	for (uint8_t d = 0; d < dim; d++) {
		int64_t s = step_q16[d];
		uint32_t u = (uint32_t)(s < 0 ? -s : s);
		if (u > max_step) max_step = u;
	}
	uint8_t shift = 0;
	uint32_t cells = max_step >> 16;
	if (cells > 1u) {
		while (cells != 0u) {
			shift++;
			cells >>= 1;
		}
	}
	return shift;
}

/* Band-limited multi-layer region generator.
 * Automatically filters out octaves whose wavelength shift is below the maximum of:
 *   1. The explicit `min_shift` threshold supplied by the caller, and
 *   2. The continuous Nyquist cutoff shift derived directly from `step_q16`. */
static inline void
chaff_fill_lod(const chaff_field_t *field,
               uint8_t min_shift,
               const chaff_coord_t origin[],
               const int32_t step_q16[],
               const uint16_t n[],
               int32_t *out, size_t stride, unsigned mode)
{
	if (!field || !out || field->dim == 0 || field->dim > 4) return;

	uint8_t nyquist_shift = chaff_nyquist_shift_q16(step_q16, field->dim);
	uint8_t cutoff_shift = (min_shift > nyquist_shift) ? min_shift : nyquist_shift;

	uint8_t lo = 0;
	while (lo < field->nlayers && field->layers[lo].shift < cutoff_shift) {
		lo++;
	}

	chaff_fill(field, lo, field->nlayers, origin, step_q16, n, out, stride, mode);
}

/* Analytical sample output: value + continuous surface normal slopes */
typedef struct {
	int32_t elevation;  /* Surface value / elevation */
	int16_t grad_x;     /* Analytical dN/dx (scaled to [-32767..32767]) */
	int16_t grad_y;     /* Analytical dN/dy */
} chaff_sample_2d_t;

/* Generic 3D analytical sample: value + spatial gradients */
typedef struct {
	int32_t val;        /* 3D noise value / density */
	int16_t grad_x;     /* Analytical spatial gradient dN/dx */
	int16_t grad_y;     /* Analytical spatial gradient dN/dy */
	int16_t grad_z;     /* Analytical spatial gradient dN/dz */
} chaff_sample_3d_t;

/* Generic 4D analytical sample: value + 4D spatio-temporal gradients */
typedef struct {
	int32_t val;        /* 4D noise value / density */
	int16_t grad_x;     /* Analytical spatial gradient dN/dx */
	int16_t grad_y;     /* Analytical spatial gradient dN/dy */
	int16_t grad_z;     /* Analytical spatial gradient dN/dz */
	int16_t grad_w;     /* Analytical 4th-axis gradient dN/dw */
} chaff_sample_4d_t;

/* Exact fixed-point linear interpolation: v0 + (((v1 - v0) * f) >> x) */
static inline uint32_t
chaff_lerp(uint32_t v0, uint32_t v1, uint32_t f, unsigned x)
{
	int64_t diff = (int64_t) v1 - (int64_t) v0;
	return (uint32_t) ((int64_t) v0 + ((diff * (int64_t) f) >> x));
}

/* Computes value and analytical surface normal slopes in a single pass.
 * Evaluates the exact derivative of the cubic Hermite fade: f'(t) = 6t(1 - t).
 * Costs zero additional corner hashes compared to standard evaluation. */
static inline chaff_sample_2d_t
chaff_sample_grad_2d(const chaff_field_t *field, int16_t wx, int16_t wy)
{
	chaff_sample_2d_t res;
	res.elevation = 0;
	res.grad_x = 0;
	res.grad_y = 0;

	if (!field || field->dim < 2) return res;

	noise_wcache_t cache;

	for (uint8_t l = 0; l < field->nlayers; l++) {
		const chaff_layer_t *lay = &field->layers[l];
		uint32_t seed = ((lay->seed_mode & 0x01u) == CHAFF_SEED_SHARED) ?
		                field->seed :
		                (field->seed ^ (uint32_t)(l * 0x85EBCA77u + 1));
		unsigned x = lay->shift;
		unsigned vshift = lay->vshift ? lay->vshift : 8;
		uint16_t mask = COORDMASK_LSHIFT(x);
		noise_wcache_init(&cache);

		int16_t cell_x = (int16_t)(wx & mask);
		int16_t cell_y = (int16_t)(wy & mask);
		uint32_t fix = (uint16_t)wx & (uint16_t)~mask;
		uint32_t fiy = (uint16_t)wy & (uint16_t)~mask;

		/* Normalized intra-cell position in [0..65535] */
		uint32_t tx = (fix << 16) >> x;
		uint32_t ty = (fiy << 16) >> x;
		if (tx > 65535u) tx = 65535u;
		if (ty > 65535u) ty = 65535u;

		/* Cubic Hermite smoothstep: u(t) = t^2 * (3 - 2t) */
		uint64_t ux = (tx * tx * (3ull * 65536ull - 2ull * tx)) >> 32;
		uint64_t uy = (ty * ty * (3ull * 65536ull - 2ull * ty)) >> 32;

		/* Analytical derivative: u'(t) = 6t(1 - t) in Q16 */
		uint64_t du_x = (6ull * tx * (65536ull - tx)) >> 16;
		uint64_t du_y = (6ull * ty * (65536ull - ty)) >> 16;

		int16_t pt_s[2] = { cell_y, cell_x };
		const uint32_t *v = noise_wcorners(&cache, pt_s, x, vshift, seed, 2);

		int64_t v00 = (int64_t)v[0];
		int64_t v10 = (int64_t)v[1];
		int64_t v01 = (int64_t)v[2];
		int64_t v11 = (int64_t)v[3];

		int64_t b0 = v00 + (((v01 - v00) * (int64_t)uy) >> 16);
		int64_t b1 = v10 + (((v11 - v10) * (int64_t)uy) >> 16);
		int64_t raw = b0 + (((b1 - b0) * (int64_t)ux) >> 16);

		/* Partial derivatives */
		int64_t dH_dux = b1 - b0;
		int64_t c0 = v00 + (((v10 - v00) * (int64_t)ux) >> 16);
		int64_t c1 = v01 + (((v11 - v01) * (int64_t)ux) >> 16);
		int64_t dH_duy = c1 - c0;

		/* Scale by analytical Hermite derivative u'(t) and layer amplitude */
		int64_t gx = (dH_dux * (int64_t)du_x) >> 16;
		int64_t gy = (dH_duy * (int64_t)du_y) >> 16;

		int32_t sample;
		int32_t sample_gx, sample_gy;
		if (lay->amp == 0) {
			sample = (int32_t)raw;
			sample_gx = (int32_t)(gx >> x);
			sample_gy = (int32_t)(gy >> x);
		} else {
			int32_t val = (int32_t)(raw >> 16) - 128;
			if (lay->seed_mode & CHAFF_LAYER_RIDGED) {
				if (val >= 0) {
					gx = -gx;
					gy = -gy;
				}
				val = 64 - abs(val);
			}
			sample = (val * (int32_t)lay->amp) >> 7;
			sample_gx = (int32_t)(((gx >> 16) * (int64_t)lay->amp) >> (7 + x));
			sample_gy = (int32_t)(((gy >> 16) * (int64_t)lay->amp) >> (7 + x));
		}

		res.elevation += sample;
		res.grad_x = (int16_t)(res.grad_x + sample_gx);
		res.grad_y = (int16_t)(res.grad_y + sample_gy);
	}

	return res;
}

/* Generic 3D point query with analytical spatial gradient.
 * Evaluates value N(x, y, z) and continuous spatial gradient (dN/dx, dN/dy, dN/dz)
 * in a single 8-corner pass using the analytical Hermite derivative u'(t) = 6t(1 - t).
 * Costs 8 corner hashes instead of 24 hashes from finite differences (3x speedup). */
static inline chaff_sample_3d_t
chaff_sample_grad_3d(const chaff_field_t *field, int16_t x, int16_t y, int16_t z)
{
	chaff_sample_3d_t res;
	res.val = 0;
	res.grad_x = 0;
	res.grad_y = 0;
	res.grad_z = 0;

	if (!field || field->dim < 3) return res;

	noise_wcache_t cache;

	for (uint8_t l = 0; l < field->nlayers; l++) {
		const chaff_layer_t *lay = &field->layers[l];
		uint32_t seed = ((lay->seed_mode & 0x01u) == CHAFF_SEED_SHARED) ?
		                field->seed :
		                (field->seed ^ (uint32_t)(l * 0x85EBCA77u + 1));
		unsigned shift = lay->shift;
		unsigned vshift = lay->vshift ? lay->vshift : 8;
		uint16_t mask = COORDMASK_LSHIFT(shift);
		noise_wcache_init(&cache);

		int16_t cell_x = (int16_t)(x & mask);
		int16_t cell_y = (int16_t)(y & mask);
		int16_t cell_z = (int16_t)(z & mask);
		uint32_t fix = (uint16_t)x & (uint16_t)~mask;
		uint32_t fiy = (uint16_t)y & (uint16_t)~mask;
		uint32_t fiz = (uint16_t)z & (uint16_t)~mask;

		/* Normalized intra-cell position in [0..65535] */
		uint32_t tx = (fix << 16) >> shift;
		uint32_t ty = (fiy << 16) >> shift;
		uint32_t tz = (fiz << 16) >> shift;
		if (tx > 65535u) tx = 65535u;
		if (ty > 65535u) ty = 65535u;
		if (tz > 65535u) tz = 65535u;

		/* Cubic Hermite smoothstep: u(t) = t^2 * (3 - 2t) */
		uint64_t ux = (tx * tx * (3ull * 65536ull - 2ull * tx)) >> 32;
		uint64_t uy = (ty * ty * (3ull * 65536ull - 2ull * ty)) >> 32;
		uint64_t uz = (tz * tz * (3ull * 65536ull - 2ull * tz)) >> 32;

		/* Analytical derivative: u'(t) = 6t(1 - t) in Q16 */
		uint64_t du_x = (6ull * tx * (65536ull - tx)) >> 16;
		uint64_t du_y = (6ull * ty * (65536ull - ty)) >> 16;
		uint64_t du_z = (6ull * tz * (65536ull - tz)) >> 16;

		int16_t pt_s[3] = { cell_z, cell_y, cell_x };
		const uint32_t *v = noise_wcorners(&cache, pt_s, shift, vshift, seed, 3);

		/* Collapse Z axis into 2D quad */
		int64_t c0 = (int64_t)v[0] + ((((int64_t)v[4] - (int64_t)v[0]) * (int64_t)uz) >> 16);
		int64_t c1 = (int64_t)v[1] + ((((int64_t)v[5] - (int64_t)v[1]) * (int64_t)uz) >> 16);
		int64_t c2 = (int64_t)v[2] + ((((int64_t)v[6] - (int64_t)v[2]) * (int64_t)uz) >> 16);
		int64_t c3 = (int64_t)v[3] + ((((int64_t)v[7] - (int64_t)v[3]) * (int64_t)uz) >> 16);

		/* Interpolate Y axis */
		int64_t b0 = c0 + (((c2 - c0) * (int64_t)uy) >> 16);
		int64_t b1 = c1 + (((c3 - c1) * (int64_t)uy) >> 16);

		/* Interpolate X axis for scalar value */
		int64_t raw = b0 + (((b1 - b0) * (int64_t)ux) >> 16);

		/* Analytical spatial derivatives */
		int64_t dH_dux = b1 - b0;
		int64_t dH_duy = (c2 - c0) + ((((c3 - c1) - (c2 - c0)) * (int64_t)ux) >> 16);

		int64_t dz0 = ((int64_t)v[4] - (int64_t)v[0]) + (((((int64_t)v[6] - (int64_t)v[2]) - ((int64_t)v[4] - (int64_t)v[0])) * (int64_t)uy) >> 16);
		int64_t dz1 = ((int64_t)v[5] - (int64_t)v[1]) + (((((int64_t)v[7] - (int64_t)v[3]) - ((int64_t)v[5] - (int64_t)v[1])) * (int64_t)uy) >> 16);
		int64_t dH_duz = dz0 + (((dz1 - dz0) * (int64_t)ux) >> 16);

		int64_t gx = (dH_dux * (int64_t)du_x) >> 16;
		int64_t gy = (dH_duy * (int64_t)du_y) >> 16;
		int64_t gz = (dH_duz * (int64_t)du_z) >> 16;

		int32_t sample;
		int32_t sample_gx, sample_gy, sample_gz;
		if (lay->amp == 0) {
			sample = (int32_t)raw;
			sample_gx = (int32_t)(gx >> shift);
			sample_gy = (int32_t)(gy >> shift);
			sample_gz = (int32_t)(gz >> shift);
		} else {
			int32_t val = (int32_t)(raw >> 16) - 128;
			if (lay->seed_mode & CHAFF_LAYER_RIDGED) {
				if (val >= 0) {
					gx = -gx;
					gy = -gy;
					gz = -gz;
				}
				val = 64 - abs(val);
			}
			sample_gx = (int32_t)(((gx >> 16) * (int64_t)lay->amp) >> (7 + shift));
			sample_gy = (int32_t)(((gy >> 16) * (int64_t)lay->amp) >> (7 + shift));
			sample_gz = (int32_t)(((gz >> 16) * (int64_t)lay->amp) >> (7 + shift));
			if (lay->contrast != 0 && lay->contrast != 256) {
				val = (val * (int32_t)lay->contrast) >> 8;
				sample_gx = (sample_gx * (int32_t)lay->contrast) >> 8;
				sample_gy = (sample_gy * (int32_t)lay->contrast) >> 8;
				sample_gz = (sample_gz * (int32_t)lay->contrast) >> 8;
			}
			sample = (val * (int32_t)lay->amp) >> 7;
		}

		res.val += sample;
		res.grad_x = (int16_t)(res.grad_x + sample_gx);
		res.grad_y = (int16_t)(res.grad_y + sample_gy);
		res.grad_z = (int16_t)(res.grad_z + sample_gz);
	}

	return res;
}

/* Generic bulk 2D slice stepper across a 3D noise field (e.g. z = time).
 * For a fixed Z coordinate across an (X, Y) region:
 *   - Pre-collapses the invariant Z axis once per 3D cell into a 2D quad in registers.
 *   - Evaluates X and Y across the cell using pure integer multiply-adds.
 *   - Produces 100% bit-exact identical values to chaff_sample_grad_3d.
 *   - out_val, out_grad_x, and out_grad_y are optional (can be NULL). */
static inline void
chaff_fill_slice_3d(const chaff_field_t *field,
                    int16_t z,
                    int16_t origin_x, int16_t origin_y,
                    uint16_t width, uint16_t height,
                    int32_t *out_val,
                    int16_t *out_grad_x,
                    int16_t *out_grad_y)
{
	if (!field || width == 0 || height == 0 || field->dim < 3) return;

	if (out_val)
		memset(out_val, 0, (size_t)width * height * sizeof(int32_t));
	if (out_grad_x)
		memset(out_grad_x, 0, (size_t)width * height * sizeof(int16_t));
	if (out_grad_y)
		memset(out_grad_y, 0, (size_t)width * height * sizeof(int16_t));

	int32_t y_end = (int32_t)origin_y + height;
	int32_t x_end = (int32_t)origin_x + width;
	noise_wcache_t cache;

	for (uint8_t l = 0; l < field->nlayers; l++) {
		const chaff_layer_t *lay = &field->layers[l];
		uint32_t seed = ((lay->seed_mode & 0x01u) == CHAFF_SEED_SHARED) ?
		                field->seed :
		                (field->seed ^ (uint32_t)(l * 0x85EBCA77u + 1));
		unsigned shift = lay->shift;
		unsigned vshift = lay->vshift ? lay->vshift : 8;
		uint16_t mask = COORDMASK_LSHIFT(shift);
		int32_t step_px = 1 << shift;
		noise_wcache_init(&cache);

		int16_t cell_z = (int16_t)(z & mask);
		uint32_t fiz = (uint16_t)z & (uint16_t)~mask;
		uint32_t tz = (fiz << 16) >> shift;
		if (tz > 65535u) tz = 65535u;
		uint64_t uz = (tz * tz * (3ull * 65536ull - 2ull * tz)) >> 32;

		int32_t start_cy = (int32_t)(int16_t)(origin_y & mask);
		int32_t start_cx = (int32_t)(int16_t)(origin_x & mask);

		for (int32_t cy = start_cy; cy < y_end; cy += step_px) {
			int16_t py_min = (int16_t)((cy < (int32_t)origin_y) ? origin_y : cy);
			int32_t next_cy = cy + step_px;
			int16_t py_max = (int16_t)((next_cy > y_end) ? y_end : next_cy);

			for (int32_t cx = start_cx; cx < x_end; cx += step_px) {
				int16_t px_min = (int16_t)((cx < (int32_t)origin_x) ? origin_x : cx);
				int32_t next_cx = cx + step_px;
				int16_t px_max = (int16_t)((next_cx > x_end) ? x_end : next_cx);

				int16_t pt_s[3] = { cell_z, (int16_t)cy, (int16_t)cx };
				const uint32_t *v = noise_wcorners(&cache, pt_s, shift, vshift, seed, 3);

				/* Pre-collapse Z once per 3D lattice cell into a 2D quad in registers */
				int64_t c0 = (int64_t)v[0] + ((((int64_t)v[4] - (int64_t)v[0]) * (int64_t)uz) >> 16);
				int64_t c1 = (int64_t)v[1] + ((((int64_t)v[5] - (int64_t)v[1]) * (int64_t)uz) >> 16);
				int64_t c2 = (int64_t)v[2] + ((((int64_t)v[6] - (int64_t)v[2]) * (int64_t)uz) >> 16);
				int64_t c3 = (int64_t)v[3] + ((((int64_t)v[7] - (int64_t)v[3]) * (int64_t)uz) >> 16);

				int64_t dy_step0 = c2 - c0;
				int64_t dy_step1 = c3 - c1;

				int cell_w = px_max - px_min;
				uint64_t cell_ux[256];
				uint64_t cell_du_x[256];
				int need_grad = (out_grad_x != NULL || out_grad_y != NULL);
				for (int i = 0; i < cell_w; i++) {
					int16_t px = (int16_t)(px_min + i);
					uint32_t fix = (uint16_t)px & (uint16_t)~mask;
					uint32_t tx = (fix << 16) >> shift;
					if (tx > 65535u) tx = 65535u;
					cell_ux[i] = (tx * tx * (3ull * 65536ull - 2ull * tx)) >> 32;
					if (need_grad)
						cell_du_x[i] = (6ull * tx * (65536ull - tx)) >> 16;
				}

				for (int16_t py = py_min; py < py_max; py++) {
					uint32_t fiy = (uint16_t)py & (uint16_t)~mask;
					uint32_t ty = (fiy << 16) >> shift;
					if (ty > 65535u) ty = 65535u;
					uint64_t uy = (ty * ty * (3ull * 65536ull - 2ull * ty)) >> 32;
					uint64_t du_y = (6ull * ty * (65536ull - ty)) >> 16;

					int64_t b0 = c0 + (((dy_step0) * (int64_t)uy) >> 16);
					int64_t b1 = c1 + (((dy_step1) * (int64_t)uy) >> 16);
					int64_t diff_b = b1 - b0;
					int64_t ddy = dy_step1 - dy_step0;
					size_t row_offset = (size_t)(py - origin_y) * width;

					if (!need_grad) {
						if (lay->amp == 0) {
							for (int i = 0; i < cell_w; i++) {
								int64_t raw = b0 + ((diff_b * (int64_t)cell_ux[i]) >> 16);
								if (out_val) out_val[row_offset + (size_t)(px_min + i - origin_x)] += (int32_t)raw;
							}
						} else {
							int32_t amp = (int32_t)lay->amp;
							uint16_t contrast = lay->contrast;
							int has_contrast = (contrast != 0 && contrast != 256);
							int is_ridged = (lay->seed_mode & CHAFF_LAYER_RIDGED) != 0;
							for (int i = 0; i < cell_w; i++) {
								int64_t raw = b0 + ((diff_b * (int64_t)cell_ux[i]) >> 16);
								int32_t val = (int32_t)(raw >> 16) - 128;
								if (is_ridged)
									val = 64 - abs(val);
								if (has_contrast)
									val = (val * (int32_t)contrast) >> 8;
								if (out_val) out_val[row_offset + (size_t)(px_min + i - origin_x)] += (val * amp) >> 7;
							}
						}
					} else {
						for (int i = 0; i < cell_w; i++) {
							uint64_t ux = cell_ux[i];
							uint64_t du_x = cell_du_x[i];

							int64_t raw = b0 + ((diff_b * (int64_t)ux) >> 16);
							int64_t dH_dux = diff_b;
							int64_t dH_duy = dy_step0 + ((ddy * (int64_t)ux) >> 16);

							int64_t gx = (dH_dux * (int64_t)du_x) >> 16;
							int64_t gy = (dH_duy * (int64_t)du_y) >> 16;

							size_t idx = row_offset + (size_t)(px_min + i - origin_x);

							if (lay->amp == 0) {
								if (out_val) out_val[idx] += (int32_t)raw;
								if (out_grad_x) out_grad_x[idx] = (int16_t)(out_grad_x[idx] + (int32_t)(gx >> shift));
								if (out_grad_y) out_grad_y[idx] = (int16_t)(out_grad_y[idx] + (int32_t)(gy >> shift));
							} else {
								int32_t val = (int32_t)(raw >> 16) - 128;
								if (lay->seed_mode & CHAFF_LAYER_RIDGED) {
									if (val >= 0) {
										gx = -gx;
										gy = -gy;
									}
									val = 64 - abs(val);
								}
								int32_t sample_gx = (int32_t)(((gx >> 16) * (int64_t)lay->amp) >> (7 + shift));
								int32_t sample_gy = (int32_t)(((gy >> 16) * (int64_t)lay->amp) >> (7 + shift));
								if (lay->contrast != 0 && lay->contrast != 256) {
									val = (val * (int32_t)lay->contrast) >> 8;
									sample_gx = (sample_gx * (int32_t)lay->contrast) >> 8;
									sample_gy = (sample_gy * (int32_t)lay->contrast) >> 8;
								}
								int32_t sample = (val * (int32_t)lay->amp) >> 7;

								if (out_val) out_val[idx] += sample;
								if (out_grad_x) out_grad_x[idx] = (int16_t)(out_grad_x[idx] + sample_gx);
								if (out_grad_y) out_grad_y[idx] = (int16_t)(out_grad_y[idx] + sample_gy);
							}
						}
					}
				}
			}
		}
	}
}

/* Generic 4D point query with analytical spatio-temporal gradient.
 * Evaluates value N(x, y, z, w) and continuous 4D gradient (dN/dx, dN/dy, dN/dz, dN/dw)
 * in a single 16-corner pass using the analytical Hermite derivative u'(t) = 6t(1 - t).
 * Costs 16 corner hashes instead of 64 hashes from finite differences (4x speedup). */
static inline chaff_sample_4d_t
chaff_sample_grad_4d(const chaff_field_t *field, int16_t x, int16_t y, int16_t z, int16_t w)
{
	chaff_sample_4d_t res;
	res.val = 0;
	res.grad_x = 0;
	res.grad_y = 0;
	res.grad_z = 0;
	res.grad_w = 0;

	if (!field || field->dim < 4) return res;

	noise_wcache_t cache;

	for (uint8_t l = 0; l < field->nlayers; l++) {
		const chaff_layer_t *lay = &field->layers[l];
		uint32_t seed = ((lay->seed_mode & 0x01u) == CHAFF_SEED_SHARED) ?
		                field->seed :
		                (field->seed ^ (uint32_t)(l * 0x85EBCA77u + 1));
		unsigned shift = lay->shift;
		unsigned vshift = lay->vshift ? lay->vshift : 8;
		uint16_t mask = COORDMASK_LSHIFT(shift);
		noise_wcache_init(&cache);

		int16_t cell_x = (int16_t)(x & mask);
		int16_t cell_y = (int16_t)(y & mask);
		int16_t cell_z = (int16_t)(z & mask);
		int16_t cell_w = (int16_t)(w & mask);
		uint32_t fix = (uint16_t)x & (uint16_t)~mask;
		uint32_t fiy = (uint16_t)y & (uint16_t)~mask;
		uint32_t fiz = (uint16_t)z & (uint16_t)~mask;
		uint32_t fiw = (uint16_t)w & (uint16_t)~mask;

		/* Normalized intra-cell position in [0..65535] */
		uint32_t tx = (fix << 16) >> shift;
		uint32_t ty = (fiy << 16) >> shift;
		uint32_t tz = (fiz << 16) >> shift;
		uint32_t tw = (fiw << 16) >> shift;
		if (tx > 65535u) tx = 65535u;
		if (ty > 65535u) ty = 65535u;
		if (tz > 65535u) tz = 65535u;
		if (tw > 65535u) tw = 65535u;

		/* Cubic Hermite smoothstep: u(t) = t^2 * (3 - 2t) */
		uint64_t ux = (tx * tx * (3ull * 65536ull - 2ull * tx)) >> 32;
		uint64_t uy = (ty * ty * (3ull * 65536ull - 2ull * ty)) >> 32;
		uint64_t uz = (tz * tz * (3ull * 65536ull - 2ull * tz)) >> 32;
		uint64_t uw = (tw * tw * (3ull * 65536ull - 2ull * tw)) >> 32;

		/* Analytical derivative: u'(t) = 6t(1 - t) in Q16 */
		uint64_t du_x = (6ull * tx * (65536ull - tx)) >> 16;
		uint64_t du_y = (6ull * ty * (65536ull - ty)) >> 16;
		uint64_t du_z = (6ull * tz * (65536ull - tz)) >> 16;
		uint64_t du_w = (6ull * tw * (65536ull - tw)) >> 16;

		int16_t pt_s[4] = { cell_w, cell_z, cell_y, cell_x };
		const uint32_t *v = noise_wcorners(&cache, pt_s, shift, vshift, seed, 4);

		/* Stage 1: Collapse W axis (16 -> 8 corners) and track W-differences */
		int64_t d[8], dw_c[8];
		for (int i = 0; i < 8; i++) {
			dw_c[i] = (int64_t)v[i + 8] - (int64_t)v[i];
			d[i] = (int64_t)v[i] + ((dw_c[i] * (int64_t)uw) >> 16);
		}

		/* Stage 2: Collapse Z axis (8 -> 4 corners) and track Z/W-differences */
		int64_t c0 = d[0] + (((d[4] - d[0]) * (int64_t)uz) >> 16);
		int64_t c1 = d[1] + (((d[5] - d[1]) * (int64_t)uz) >> 16);
		int64_t c2 = d[2] + (((d[6] - d[2]) * (int64_t)uz) >> 16);
		int64_t c3 = d[3] + (((d[7] - d[3]) * (int64_t)uz) >> 16);

		int64_t dz_c0 = d[4] - d[0];
		int64_t dz_c1 = d[5] - d[1];
		int64_t dz_c2 = d[6] - d[2];
		int64_t dz_c3 = d[7] - d[3];

		int64_t dw0 = dw_c[0] + (((dw_c[4] - dw_c[0]) * (int64_t)uz) >> 16);
		int64_t dw1 = dw_c[1] + (((dw_c[5] - dw_c[1]) * (int64_t)uz) >> 16);
		int64_t dw2 = dw_c[2] + (((dw_c[6] - dw_c[2]) * (int64_t)uz) >> 16);
		int64_t dw3 = dw_c[3] + (((dw_c[7] - dw_c[3]) * (int64_t)uz) >> 16);

		/* Stage 3: Interpolate Y axis (4 -> 2 corners) */
		int64_t b0 = c0 + (((c2 - c0) * (int64_t)uy) >> 16);
		int64_t b1 = c1 + (((c3 - c1) * (int64_t)uy) >> 16);

		/* Stage 4: Interpolate X axis for scalar value (2 -> 1) */
		int64_t raw = b0 + (((b1 - b0) * (int64_t)ux) >> 16);

		/* Analytical 4D partial derivatives */
		int64_t dH_dux = b1 - b0;
		int64_t dH_duy = (c2 - c0) + ((((c3 - c1) - (c2 - c0)) * (int64_t)ux) >> 16);

		int64_t dz0 = dz_c0 + (((dz_c2 - dz_c0) * (int64_t)uy) >> 16);
		int64_t dz1 = dz_c1 + (((dz_c3 - dz_c1) * (int64_t)uy) >> 16);
		int64_t dH_duz = dz0 + (((dz1 - dz0) * (int64_t)ux) >> 16);

		int64_t dw_y0 = dw0 + (((dw2 - dw0) * (int64_t)uy) >> 16);
		int64_t dw_y1 = dw1 + (((dw3 - dw1) * (int64_t)uy) >> 16);
		int64_t dH_duw = dw_y0 + (((dw_y1 - dw_y0) * (int64_t)ux) >> 16);

		int64_t gx = (dH_dux * (int64_t)du_x) >> 16;
		int64_t gy = (dH_duy * (int64_t)du_y) >> 16;
		int64_t gz = (dH_duz * (int64_t)du_z) >> 16;
		int64_t gw = (dH_duw * (int64_t)du_w) >> 16;

		int32_t sample;
		int32_t sample_gx, sample_gy, sample_gz, sample_gw;
		if (lay->amp == 0) {
			sample = (int32_t)raw;
			sample_gx = (int32_t)(gx >> shift);
			sample_gy = (int32_t)(gy >> shift);
			sample_gz = (int32_t)(gz >> shift);
			sample_gw = (int32_t)(gw >> shift);
		} else {
			int32_t val = (int32_t)(raw >> 16) - 128;
			if (lay->seed_mode & CHAFF_LAYER_RIDGED) {
				if (val >= 0) {
					gx = -gx;
					gy = -gy;
					gz = -gz;
					gw = -gw;
				}
				val = 64 - abs(val);
			}
			sample_gx = (int32_t)(((gx >> 16) * (int64_t)lay->amp) >> (7 + shift));
			sample_gy = (int32_t)(((gy >> 16) * (int64_t)lay->amp) >> (7 + shift));
			sample_gz = (int32_t)(((gz >> 16) * (int64_t)lay->amp) >> (7 + shift));
			sample_gw = (int32_t)(((gw >> 16) * (int64_t)lay->amp) >> (7 + shift));
			if (lay->contrast != 0 && lay->contrast != 256) {
				val = (val * (int32_t)lay->contrast) >> 8;
				sample_gx = (sample_gx * (int32_t)lay->contrast) >> 8;
				sample_gy = (sample_gy * (int32_t)lay->contrast) >> 8;
				sample_gz = (sample_gz * (int32_t)lay->contrast) >> 8;
				sample_gw = (sample_gw * (int32_t)lay->contrast) >> 8;
			}
			sample = (val * (int32_t)lay->amp) >> 7;
		}

		res.val += sample;
		res.grad_x = (int16_t)(res.grad_x + sample_gx);
		res.grad_y = (int16_t)(res.grad_y + sample_gy);
		res.grad_z = (int16_t)(res.grad_z + sample_gz);
		res.grad_w = (int16_t)(res.grad_w + sample_gw);
	}

	return res;
}

/* Generic bulk 2D slice stepper across a 4D noise field (e.g. z = altitude, w = time).
 * For fixed (Z, W) coordinates across an (X, Y) region:
 *   - Pre-collapses the invariant W (16 -> 8) and Z (8 -> 4) axes once per 4D cell
 *     into a 2D quad in registers.
 *   - Evaluates X and Y across the cell using pure integer multiply-adds at 2D cost.
 *   - Produces 100% bit-exact identical values and 4D gradients to chaff_sample_grad_4d.
 *   - out_val and out_grad_* are optional (can be NULL). */
static inline void
chaff_fill_slice_4d(const chaff_field_t *field,
                    int16_t z, int16_t w,
                    int16_t origin_x, int16_t origin_y,
                    uint16_t width, uint16_t height,
                    int32_t *out_val,
                    int16_t *out_grad_x,
                    int16_t *out_grad_y,
                    int16_t *out_grad_z,
                    int16_t *out_grad_w)
{
	if (!field || width == 0 || height == 0 || field->dim < 4) return;

	if (out_val)
		memset(out_val, 0, (size_t)width * height * sizeof(int32_t));
	if (out_grad_x)
		memset(out_grad_x, 0, (size_t)width * height * sizeof(int16_t));
	if (out_grad_y)
		memset(out_grad_y, 0, (size_t)width * height * sizeof(int16_t));
	if (out_grad_z)
		memset(out_grad_z, 0, (size_t)width * height * sizeof(int16_t));
	if (out_grad_w)
		memset(out_grad_w, 0, (size_t)width * height * sizeof(int16_t));

	int32_t y_end = (int32_t)origin_y + height;
	int32_t x_end = (int32_t)origin_x + width;
	noise_wcache_t cache;

	for (uint8_t l = 0; l < field->nlayers; l++) {
		const chaff_layer_t *lay = &field->layers[l];
		uint32_t seed = ((lay->seed_mode & 0x01u) == CHAFF_SEED_SHARED) ?
		                field->seed :
		                (field->seed ^ (uint32_t)(l * 0x85EBCA77u + 1));
		unsigned shift = lay->shift;
		unsigned vshift = lay->vshift ? lay->vshift : 8;
		uint16_t mask = COORDMASK_LSHIFT(shift);
		int32_t step_px = 1 << shift;
		noise_wcache_init(&cache);

		int16_t cell_z = (int16_t)(z & mask);
		int16_t cell_w = (int16_t)(w & mask);
		uint32_t fiz = (uint16_t)z & (uint16_t)~mask;
		uint32_t fiw = (uint16_t)w & (uint16_t)~mask;
		uint32_t tz = (fiz << 16) >> shift;
		uint32_t tw = (fiw << 16) >> shift;
		if (tz > 65535u) tz = 65535u;
		if (tw > 65535u) tw = 65535u;
		uint64_t uz = (tz * tz * (3ull * 65536ull - 2ull * tz)) >> 32;
		uint64_t uw = (tw * tw * (3ull * 65536ull - 2ull * tw)) >> 32;
		uint64_t du_z = (6ull * tz * (65536ull - tz)) >> 16;
		uint64_t du_w = (6ull * tw * (65536ull - tw)) >> 16;

		int32_t start_cy = (int32_t)(int16_t)(origin_y & mask);
		int32_t start_cx = (int32_t)(int16_t)(origin_x & mask);

		for (int32_t cy = start_cy; cy < y_end; cy += step_px) {
			int16_t py_min = (int16_t)((cy < (int32_t)origin_y) ? origin_y : cy);
			int32_t next_cy = cy + step_px;
			int16_t py_max = (int16_t)((next_cy > y_end) ? y_end : next_cy);

			for (int32_t cx = start_cx; cx < x_end; cx += step_px) {
				int16_t px_min = (int16_t)((cx < (int32_t)origin_x) ? origin_x : cx);
				int32_t next_cx = cx + step_px;
				int16_t px_max = (int16_t)((next_cx > x_end) ? x_end : next_cx);

				int16_t pt_s[4] = { cell_w, cell_z, (int16_t)cy, (int16_t)cx };
				const uint32_t *v = noise_wcorners(&cache, pt_s, shift, vshift, seed, 4);

				/* Pre-collapse W (16 -> 8) and Z (8 -> 4) once per 4D cell into a 2D quad */
				int64_t d[8], dw_c[8];
				for (int i = 0; i < 8; i++) {
					dw_c[i] = (int64_t)v[i + 8] - (int64_t)v[i];
					d[i] = (int64_t)v[i] + ((dw_c[i] * (int64_t)uw) >> 16);
				}

				int64_t c0 = d[0] + (((d[4] - d[0]) * (int64_t)uz) >> 16);
				int64_t c1 = d[1] + (((d[5] - d[1]) * (int64_t)uz) >> 16);
				int64_t c2 = d[2] + (((d[6] - d[2]) * (int64_t)uz) >> 16);
				int64_t c3 = d[3] + (((d[7] - d[3]) * (int64_t)uz) >> 16);

				int64_t dy_step0 = c2 - c0;
				int64_t dy_step1 = c3 - c1;

				int64_t dz_c0 = d[4] - d[0];
				int64_t dz_c1 = d[5] - d[1];
				int64_t dz_c2 = d[6] - d[2];
				int64_t dz_c3 = d[7] - d[3];

				int64_t dw0 = dw_c[0] + (((dw_c[4] - dw_c[0]) * (int64_t)uz) >> 16);
				int64_t dw1 = dw_c[1] + (((dw_c[5] - dw_c[1]) * (int64_t)uz) >> 16);
				int64_t dw2 = dw_c[2] + (((dw_c[6] - dw_c[2]) * (int64_t)uz) >> 16);
				int64_t dw3 = dw_c[3] + (((dw_c[7] - dw_c[3]) * (int64_t)uz) >> 16);

				int cell_w_px = px_max - px_min;
				uint64_t cell_ux[256];
				uint64_t cell_du_x[256];
				int need_grad = (out_grad_x != NULL || out_grad_y != NULL ||
				                 out_grad_z != NULL || out_grad_w != NULL);
				for (int i = 0; i < cell_w_px; i++) {
					int16_t px = (int16_t)(px_min + i);
					uint32_t fix = (uint16_t)px & (uint16_t)~mask;
					uint32_t tx = (fix << 16) >> shift;
					if (tx > 65535u) tx = 65535u;
					cell_ux[i] = (tx * tx * (3ull * 65536ull - 2ull * tx)) >> 32;
					if (need_grad)
						cell_du_x[i] = (6ull * tx * (65536ull - tx)) >> 16;
				}

				for (int16_t py = py_min; py < py_max; py++) {
					uint32_t fiy = (uint16_t)py & (uint16_t)~mask;
					uint32_t ty = (fiy << 16) >> shift;
					if (ty > 65535u) ty = 65535u;
					uint64_t uy = (ty * ty * (3ull * 65536ull - 2ull * ty)) >> 32;
					uint64_t du_y = (6ull * ty * (65536ull - ty)) >> 16;

					int64_t b0 = c0 + ((dy_step0 * (int64_t)uy) >> 16);
					int64_t b1 = c1 + ((dy_step1 * (int64_t)uy) >> 16);
					int64_t diff_b = b1 - b0;
					int64_t ddy = dy_step1 - dy_step0;
					size_t row_offset = (size_t)(py - origin_y) * width;

					if (!need_grad) {
						if (lay->amp == 0) {
							for (int i = 0; i < cell_w_px; i++) {
								int64_t raw = b0 + ((diff_b * (int64_t)cell_ux[i]) >> 16);
								if (out_val) out_val[row_offset + (size_t)(px_min + i - origin_x)] += (int32_t)raw;
							}
						} else {
							int32_t amp = (int32_t)lay->amp;
							uint16_t contrast = lay->contrast;
							int has_contrast = (contrast != 0 && contrast != 256);
							int is_ridged = (lay->seed_mode & CHAFF_LAYER_RIDGED) != 0;
							for (int i = 0; i < cell_w_px; i++) {
								int64_t raw = b0 + ((diff_b * (int64_t)cell_ux[i]) >> 16);
								int32_t val = (int32_t)(raw >> 16) - 128;
								if (is_ridged)
									val = 64 - abs(val);
								if (has_contrast)
									val = (val * (int32_t)contrast) >> 8;
								if (out_val) out_val[row_offset + (size_t)(px_min + i - origin_x)] += (val * amp) >> 7;
							}
						}
					} else {
						int64_t dz0 = dz_c0 + (((dz_c2 - dz_c0) * (int64_t)uy) >> 16);
						int64_t dz1 = dz_c1 + (((dz_c3 - dz_c1) * (int64_t)uy) >> 16);
						int64_t diff_dz = dz1 - dz0;

						int64_t dw_y0 = dw0 + (((dw2 - dw0) * (int64_t)uy) >> 16);
						int64_t dw_y1 = dw1 + (((dw3 - dw1) * (int64_t)uy) >> 16);
						int64_t diff_dw = dw_y1 - dw_y0;

						for (int i = 0; i < cell_w_px; i++) {
							uint64_t ux = cell_ux[i];
							uint64_t du_x = cell_du_x[i];

							int64_t raw = b0 + ((diff_b * (int64_t)ux) >> 16);
							int64_t dH_dux = diff_b;
							int64_t dH_duy = dy_step0 + ((ddy * (int64_t)ux) >> 16);
							int64_t dH_duz = dz0 + ((diff_dz * (int64_t)ux) >> 16);
							int64_t dH_duw = dw_y0 + ((diff_dw * (int64_t)ux) >> 16);

							int64_t gx = (dH_dux * (int64_t)du_x) >> 16;
							int64_t gy = (dH_duy * (int64_t)du_y) >> 16;
							int64_t gz = (dH_duz * (int64_t)du_z) >> 16;
							int64_t gw = (dH_duw * (int64_t)du_w) >> 16;

							size_t idx = row_offset + (size_t)(px_min + i - origin_x);

							if (lay->amp == 0) {
								if (out_val) out_val[idx] += (int32_t)raw;
								if (out_grad_x) out_grad_x[idx] = (int16_t)(out_grad_x[idx] + (int32_t)(gx >> shift));
								if (out_grad_y) out_grad_y[idx] = (int16_t)(out_grad_y[idx] + (int32_t)(gy >> shift));
								if (out_grad_z) out_grad_z[idx] = (int16_t)(out_grad_z[idx] + (int32_t)(gz >> shift));
								if (out_grad_w) out_grad_w[idx] = (int16_t)(out_grad_w[idx] + (int32_t)(gw >> shift));
							} else {
								int32_t val = (int32_t)(raw >> 16) - 128;
								if (lay->seed_mode & CHAFF_LAYER_RIDGED) {
									if (val >= 0) {
										gx = -gx;
										gy = -gy;
										gz = -gz;
										gw = -gw;
									}
									val = 64 - abs(val);
								}
								int32_t sample_gx = (int32_t)(((gx >> 16) * (int64_t)lay->amp) >> (7 + shift));
								int32_t sample_gy = (int32_t)(((gy >> 16) * (int64_t)lay->amp) >> (7 + shift));
								int32_t sample_gz = (int32_t)(((gz >> 16) * (int64_t)lay->amp) >> (7 + shift));
								int32_t sample_gw = (int32_t)(((gw >> 16) * (int64_t)lay->amp) >> (7 + shift));
								if (lay->contrast != 0 && lay->contrast != 256) {
									val = (val * (int32_t)lay->contrast) >> 8;
									sample_gx = (sample_gx * (int32_t)lay->contrast) >> 8;
									sample_gy = (sample_gy * (int32_t)lay->contrast) >> 8;
									sample_gz = (sample_gz * (int32_t)lay->contrast) >> 8;
									sample_gw = (sample_gw * (int32_t)lay->contrast) >> 8;
								}
								int32_t sample = (val * (int32_t)lay->amp) >> 7;

								if (out_val) out_val[idx] += sample;
								if (out_grad_x) out_grad_x[idx] = (int16_t)(out_grad_x[idx] + sample_gx);
								if (out_grad_y) out_grad_y[idx] = (int16_t)(out_grad_y[idx] + sample_gy);
								if (out_grad_z) out_grad_z[idx] = (int16_t)(out_grad_z[idx] + sample_gz);
								if (out_grad_w) out_grad_w[idx] = (int16_t)(out_grad_w[idx] + sample_gw);
							}
						}
					}
				}
			}
		}
	}
}

/* --- Parameterized 2D Fixed-Point Rotation & Anti-Resonance Skew ---------- */
#define CHAFF_ROT_GOLD_COS_Q15 27874 /* cos(31.717 deg) * 32768 */
#define CHAFF_ROT_GOLD_SIN_Q15 17227 /* sin(31.717 deg) * 32768 */
#define CHAFF_ROT_ATAN_COS_Q15 29304 /* cos(arctan(0.5)) * 32768 */
#define CHAFF_ROT_ATAN_SIN_Q15 14652 /* sin(arctan(0.5)) * 32768 */

#define CHAFF_ROT_COS CHAFF_ROT_GOLD_COS_Q15
#define CHAFF_ROT_SIN CHAFF_ROT_GOLD_SIN_Q15

/* Parameterized 2D fixed-point Q15 rotation for int16_t coordinates */
static inline void
chaff_rot2d_q15(int16_t *x, int16_t *y, int32_t cos_q15, int32_t sin_q15)
{
	int32_t x0 = *x;
	int32_t y0 = *y;
	*x = (int16_t)((x0 * cos_q15 - y0 * sin_q15) >> 15);
	*y = (int16_t)((x0 * sin_q15 + y0 * cos_q15) >> 15);
}

/* Parameterized 2D fixed-point Q15 rotation for int32_t coordinates */
static inline void
chaff_rot2d_i32_q15(int32_t *x, int32_t *y, int32_t cos_q15, int32_t sin_q15)
{
	int64_t x0 = *x;
	int64_t y0 = *y;
	*x = (int32_t)((x0 * cos_q15 - y0 * sin_q15) >> 15);
	*y = (int32_t)((x0 * sin_q15 + y0 * cos_q15) >> 15);
}

static inline void
chaff_rotate_q15(int16_t *x, int16_t *y)
{
	chaff_rot2d_q15(x, y, CHAFF_ROT_GOLD_COS_Q15, CHAFF_ROT_GOLD_SIN_Q15);
}

static inline void
chaff_rotate_q15_inplace(int16_t *x, int16_t *y)
{
	chaff_rot2d_q15(x, y, CHAFF_ROT_ATAN_COS_Q15, CHAFF_ROT_ATAN_SIN_Q15);
}

/* Rugged point evaluation: rotates octave coordinate systems
 * to break power-of-two Cartesian lattice resonance and terrace alignment. */
static inline chaff_sample_2d_t
chaff_sample_rugged_2d(const chaff_field_t *field, int16_t wx, int16_t wy)
{
	chaff_sample_2d_t res;
	res.elevation = 0;
	res.grad_x = 0;
	res.grad_y = 0;
	if (!field || field->nlayers == 0) return res;

	int16_t cur_x = wx;
	int16_t cur_y = wy;

	for (uint8_t l = 0; l < field->nlayers; l++) {
		const chaff_layer_t *lay = &field->layers[l];
		uint32_t oct_seed = ((lay->seed_mode & 0x01u) == CHAFF_SEED_SHARED) ?
		                     field->seed :
		                     (field->seed ^ (uint32_t)(l * 0x85EBCA77u + 1));

		chaff_field_t single_layer = {
			.seed = oct_seed,
			.dim = 2,
			.flags = field->flags,
			.nlayers = 1,
			.layers = lay
		};

		chaff_sample_2d_t sample = chaff_sample_grad_2d(&single_layer, cur_x, cur_y);
		res.elevation += sample.elevation;
		res.grad_x = (int16_t)(res.grad_x + sample.grad_x);
		res.grad_y = (int16_t)(res.grad_y + sample.grad_y);

		chaff_rotate_q15(&cur_x, &cur_y);
	}

	return res;
}

/* --- Generic Fractal Amplitude Scaling & Periodic Domain Sampling --------- */

/* Parameterized 1/f continuous fractal amplitude scaling (zero switch-ladders):
 * Higher shifts (longer wavelengths) strictly produce higher amplitudes:
 *   A(s) = base_amp * 2^((s - min_shift) / 2)
 * using exact integer half-octave steps (1, 1.5, 2, 3, 4, 6, 8, ...). */
static inline int16_t
chaff_octave_amplitude_ex(uint8_t shift, uint8_t min_shift, int16_t base_amp)
{
	if (shift < min_shift || base_amp <= 0) return 0;
	uint8_t k = (uint8_t)(shift - min_shift);
	int32_t base = (int32_t)base_amp << (k >> 1);
	if (k & 1u) {
		int32_t half = base >> 1;
		base += half;
	}
	return (int16_t)base;
}

static inline int16_t
chaff_octave_amplitude(uint8_t shift)
{
	return chaff_octave_amplitude_ex(shift, 2, 1);
}

/* Evaluates a continuous 2D C2-quintic periodic octave with domain period 2^period_shift
 * and feature cell wavelength 2^shift (where shift <= period_shift). */
static inline uint32_t
chaff_wpoint_periodic_2d(int32_t x, int32_t y, uint8_t shift, uint8_t period_shift,
                         uint8_t vshift, uint32_t seed)
{
	if (period_shift == 0 || shift > period_shift) return 0;
	uint32_t period_mask = (period_shift >= 32) ? 0xFFFFFFFFu : ((1u << period_shift) - 1u);
	uint32_t cell_count_mask = (1u << (period_shift - shift)) - 1u;
	uint32_t mask_frac = (1u << shift) - 1u;

	uint32_t ux = (uint32_t)x & period_mask;
	uint32_t uy = (uint32_t)y & period_mask;

	uint16_t kx0 = (uint16_t)(ux >> shift);
	uint16_t ky0 = (uint16_t)(uy >> shift);
	uint16_t kx1 = (uint16_t)((kx0 + 1u) & cell_count_mask);
	uint16_t ky1 = (uint16_t)((ky0 + 1u) & cell_count_mask);

	uint32_t frac_x = ux & mask_frac;
	uint32_t frac_y = uy & mask_frac;

	uint32_t fx = noise_wfade5(frac_x, shift);
	uint32_t fy = noise_wfade5(frac_y, shift);

	int16_t p00[2] = { (int16_t)ky0, (int16_t)kx0 };
	int16_t p10[2] = { (int16_t)ky0, (int16_t)kx1 };
	int16_t p01[2] = { (int16_t)ky1, (int16_t)kx0 };
	int16_t p11[2] = { (int16_t)ky1, (int16_t)kx1 };

	uint32_t v00 = noise_r(p00, seed, vshift, 2);
	uint32_t v10 = noise_r(p10, seed, vshift, 2);
	uint32_t v01 = noise_r(p01, seed, vshift, 2);
	uint32_t v11 = noise_r(p11, seed, vshift, 2);

	uint32_t b0 = noise_wstep(v00, v10, fx, shift);
	uint32_t b1 = noise_wstep(v01, v11, fx, shift);
	return noise_wstep(b0, b1, fy, shift);
}

/* Convenience wrapper for 16-bit periodic 2D domain (period = 2^16, vshift = 8) */
static inline uint32_t
chaff_sample_torus_macro(int32_t rx, int32_t ry, uint8_t shift, uint32_t seed)
{
	return chaff_wpoint_periodic_2d(rx, ry, shift, 16, 8, seed);
}

static inline uint32_t
chaff_sample_mega_2d(int32_t rx, int32_t ry, unsigned shift, unsigned seed)
{
	uint8_t s = (shift > 15) ? 15 : (uint8_t)shift;
	return chaff_wpoint_periodic_2d(rx, ry, s, 16, 8, seed);
}

/* Generic 2D rotated fractal octave accumulator in Q12.4 fixed-point.
 * Evaluates octaves from `max_shift` down to `min_shift` with C2 quintic fades (NOISE_FADE5),
 * applying ridged multifractal folding on octaves >= `ridged_min_shift` (pass 255 to disable)
 * and rotating coordinates by `(cos_q15, sin_q15)` between octaves. */
static inline int32_t
chaff_fbm_rotated_2d_q12_4(int32_t x, int32_t y,
                           uint8_t min_shift, uint8_t max_shift,
                           uint8_t ridged_min_shift,
                           uint8_t amp_min_shift, int16_t base_amp,
                           uint32_t seed, uint32_t seed_salt,
                           int32_t cos_q15, int32_t sin_q15)
{
	if (min_shift > max_shift) return 0;
	int32_t cur_x = x;
	int32_t cur_y = y;
	int32_t acc_q12_4 = 0;

	for (int8_t s = (int8_t)max_shift; s >= (int8_t)min_shift; s--) {
		int16_t amp = chaff_octave_amplitude_ex((uint8_t)s, amp_min_shift, base_amp);
		uint32_t oct_seed = seed ^ (uint32_t)((uint8_t)s * 0x85EBCA77u + seed_salt);
		int16_t p[2] = { (int16_t)cur_y, (int16_t)cur_x };
		uint32_t raw = noise_wpoint_n(p, (uint8_t)s, 8, oct_seed, 2, NOISE_FADE5);
		int32_t val = (int32_t)(raw >> 16) - 128;
		if ((uint8_t)s >= ridged_min_shift) {
			val = 64 - abs(val);
		}
		acc_q12_4 += (int32_t)((((int64_t)val * amp) << 4) / 128);
		chaff_rot2d_i32_q15(&cur_x, &cur_y, cos_q15, sin_q15);
	}
	return acc_q12_4;
}

#endif /* TTYPT_CHAFF_H */
