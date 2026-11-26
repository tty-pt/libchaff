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
 *   2. Analytical 2D/3D/4D spatial gradients and 2D slice stepping.
 *   3. Parameterized 2D fixed-point domain rotation (chaff_rot2d_q15).
 *   4. Generic periodic 2D domain sampling (chaff_wpoint_periodic_2d) and
 *      monotonic fractal amplitude scaling (chaff_octave_amplitude_ex).
 *   5. Optional header-only GPU shader contract via <ttypt/chaff_glsl.h>.
 *
 * Pure C99, integer-only core, zero floating point, zero game/domain constants.
 */

#include <ttypt/noise.h>

#ifndef _CHAFF_TLS
#if defined(_MSC_VER)
#define _CHAFF_TLS __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#define _CHAFF_TLS __thread
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_THREADS__)
#define _CHAFF_TLS _Thread_local
#else
#define _CHAFF_TLS
#endif
#endif

/** Default octave value shift (vshift = 8 yields [0..255] after >> 16). */
#define CHAFF_DEFAULT_VSHIFT 8u

/** Default periodic domain shift (2^16 = 65536 cells). */
#define CHAFF_DEFAULT_PERIOD_SHIFT 16u

/** Default minimum octave shift for fractal amplitude scaling. */
#define CHAFF_DEFAULT_MIN_SHIFT 2u

/** Default base octave amplitude for fractal amplitude scaling. */
#define CHAFF_DEFAULT_BASE_AMP 1

/** Default fractional output shift for Q12.4 rotated FBM accumulators. */
#define CHAFF_DEFAULT_OUT_SHIFT 4u

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

/* Continuous C2 Nyquist low-pass transfer function L(lambda, Delta) in Q16 [0..65536]:
 *   - For step_q16 <= 0 (point sample): returns 65536 (1.0).
 *   - For wavelength <= 2 * step_q16 (sub-Nyquist): returns 0 (skip evaluation).
 *   - For wavelength >= 4 * step_q16 (fully resolved): returns 65536 (1.0).
 *   - For 2 * step_q16 < wavelength < 4 * step_q16: smooth C2 quintic transition. */
static inline uint32_t
chaff_wavelength_weight_q16(int32_t wavelength_q16, int32_t step_q16)
{
	if (step_q16 <= 0) return 65536u;
	if (wavelength_q16 <= 0) return 0u;
	uint64_t wl = (uint64_t)(uint32_t)wavelength_q16;
	uint64_t st = (uint64_t)(uint32_t)step_q16;
	uint64_t nyq_lo = st << 1;
	if (wl <= nyq_lo) return 0u;
	uint64_t nyq_hi = st << 2;
	if (wl >= nyq_hi) return 65536u;
	uint32_t t_q16 = (uint32_t)(((wl - nyq_lo) << 16) / (nyq_hi - nyq_lo));
	return noise_wfade5(t_q16, 16);
}

/* Continuous C2 Nyquist low-pass weight L(2^shift, Delta) in Q16 [0..65536] for dyadic octave `shift`. */
static inline uint32_t
chaff_octave_weight_q16(uint8_t shift, int32_t step_q16)
{
	if (step_q16 <= 0) return 65536u;
	if (shift >= 31u) return 65536u;
	uint64_t wl = 1ULL << (shift + 16u);
	uint64_t st = (uint64_t)(uint32_t)step_q16;
	uint64_t nyq_lo = st << 1;
	if (wl <= nyq_lo) return 0u;
	uint64_t nyq_hi = st << 2;
	if (wl >= nyq_hi) return 65536u;
	uint32_t t_q16 = (uint32_t)(((wl - nyq_lo) << 16) / (nyq_hi - nyq_lo));
	return noise_wfade5(t_q16, 16);
}

/* Complementary scale-space band-pass weight B(2^shift; fine_step, coarse_step) in Q16 [0..65536]:
 * B(lambda; Delta_fine, Delta_coarse) = L(lambda, Delta_fine) * (1 - L(lambda, Delta_coarse)).
 * Satisfies exact partition of unity: L(lambda, Delta_coarse) + B(lambda; 0, Delta_coarse) == 1. */
static inline uint32_t
chaff_band_weight_q16(uint8_t shift, int32_t fine_step_q16, int32_t coarse_step_q16)
{
	uint32_t w_fine = chaff_octave_weight_q16(shift, fine_step_q16);
	if (w_fine == 0u) return 0u;
	uint32_t w_coarse = chaff_octave_weight_q16(shift, coarse_step_q16);
	if (w_coarse >= 65536u) return 0u;
	return (uint32_t)(((uint64_t)w_fine * (65536u - w_coarse)) >> 16);
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

/* Computes active layer index range [out_lo, out_hi) for a given Q16.16 sampling stride
 * and explicit minimum shift cutoff. */
static inline void
chaff_band_range_for_step_q16(const chaff_field_t *field,
                              uint8_t min_shift,
                              const int32_t step_q16[],
                              uint8_t *out_lo, uint8_t *out_hi)
{
	if (!field) {
		if (out_lo) *out_lo = 0;
		if (out_hi) *out_hi = 0;
		return;
	}
	uint8_t nyquist_shift = chaff_nyquist_shift_q16(step_q16, field->dim);
	uint8_t cutoff_shift = (min_shift > nyquist_shift) ? min_shift : nyquist_shift;
	uint8_t lo = 0;
	while (lo < field->nlayers && field->layers[lo].shift < cutoff_shift) {
		lo++;
	}
	if (out_lo) *out_lo = lo;
	if (out_hi) *out_hi = field->nlayers;
}

/* Generic 2D analytical sample: scalar value + continuous spatial gradients */
typedef struct {
	union {
		int32_t val;        /* 2D scalar noise value */
		int32_t elevation;  /* Alias for backward source compatibility */
	};
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

/* Evaluates intra-cell fade u(t) and analytical derivative u'(t) in Q16 based on field->flags:
 *   - CHAFF_FLAG_LINEAR: u(t) = t,                 u'(t) = 1.0
 *   - CHAFF_FLAG_FADE5:  u(t) = t^3(6t^2-15t+10),  u'(t) = 30 t^2 (1 - t)^2
 *   - Default / FADE:    u(t) = t^2(3 - 2t),       u'(t) = 6 t (1 - t) */
static inline void
chaff_eval_fade_and_deriv(uint32_t t, uint8_t flags, uint64_t *out_u, uint64_t *out_du)
{
	if (t > 65535u) t = 65535u;
	if (flags & CHAFF_FLAG_LINEAR) {
		*out_u = (uint64_t)t;
		*out_du = 65536ull;
	} else if (flags & CHAFF_FLAG_FADE5) {
		uint64_t t64 = (uint64_t)t;
		uint64_t p1 = (t64 * 6ull) >> 16;
		int64_t  p2 = (int64_t)p1 - (15ll << 16);
		int64_t  p3 = (((int64_t)t64 * p2) >> 16) + (10ll << 16);
		uint64_t t2 = (t64 * t64) >> 16;
		uint64_t t3 = (t2 * t64) >> 16;
		uint64_t u  = (t3 * (uint64_t)p3) >> 16;
		if (u > 65536ull) u = 65536ull;
		uint64_t omt = 65536ull - t64;
		uint64_t omt2 = (omt * omt) >> 16;
		*out_u = u;
		*out_du = (30ull * t2 * omt2) >> 16;
	} else {
		uint64_t t64 = (uint64_t)t;
		*out_u  = (t64 * t64 * (3ull * 65536ull - 2ull * t64)) >> 32;
		*out_du = (6ull * t64 * (65536ull - t64)) >> 16;
	}
}

/* Computes value and analytical surface normal slopes in a single pass.
 * Evaluates the exact derivative of the configured fade curve.
 * Costs zero additional corner hashes compared to standard evaluation. */
static inline chaff_sample_2d_t
chaff_sample_grad_2d(const chaff_field_t *field, int16_t wx, int16_t wy)
{
	chaff_sample_2d_t res;
	res.val = 0;
	res.grad_x = 0;
	res.grad_y = 0;

	if (!field || field->dim < 2) return res;

	noise_wcache_t cache;

	for (uint8_t l = 0; l < field->nlayers; l++) {
		const chaff_layer_t *lay = &field->layers[l];
		uint32_t seed = chaff_layer_seed(field, lay, l);
		unsigned x = lay->shift;
		unsigned vshift = (lay->vshift && lay->vshift < 16u) ? lay->vshift : CHAFF_DEFAULT_VSHIFT;
		int32_t mid = (int32_t)(1u << (15u - vshift));
		int32_t ridge_fold = mid >> 1;
		unsigned amp_shift = 15u - vshift;
		uint16_t mask = COORDMASK_LSHIFT(x);
		noise_wcache_init(&cache);

		int16_t sx = wx, sy = wy;
		if (lay->skew[0]) sx = (int16_t)(sx + sy / lay->skew[0]);
		if (lay->skew[1]) sy = (int16_t)(sy - sx / lay->skew[1]);

		int16_t cell_x = (int16_t)(sx & mask);
		int16_t cell_y = (int16_t)(sy & mask);
		uint32_t fix = (uint16_t)sx & (uint16_t)~mask;
		uint32_t fiy = (uint16_t)sy & (uint16_t)~mask;

		/* Normalized intra-cell position in [0..65535] */
		uint32_t tx = (fix << 16) >> x;
		uint32_t ty = (fiy << 16) >> x;

		uint64_t ux, uy, du_x, du_y;
		chaff_eval_fade_and_deriv(tx, field->flags, &ux, &du_x);
		chaff_eval_fade_and_deriv(ty, field->flags, &uy, &du_y);

		int16_t pt_s[2] = { cell_y, cell_x };
		const uint32_t *v = noise_wcorners_periodic(&cache, pt_s, x, vshift, seed, 2, lay->period_shift);
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

		/* Scale by analytical fade derivative u'(t) and layer amplitude */
		int64_t gx = (dH_dux * (int64_t)du_x) >> 16;
		int64_t gy = (dH_duy * (int64_t)du_y) >> 16;

		int32_t sample;
		int32_t sample_gx, sample_gy;
		if (lay->amp == 0) {
			sample = (int32_t)raw;
			sample_gx = (int32_t)(gx >> x);
			sample_gy = (int32_t)(gy >> x);
		} else {
			int32_t val = (int32_t)(raw >> 16) - mid;
			if (lay->seed_mode & CHAFF_LAYER_RIDGED) {
				if (val >= 0) {
					gx = -gx;
					gy = -gy;
				}
				val = ridge_fold - abs(val);
			}
			sample_gx = (int32_t)(((gx >> 16) * (int64_t)lay->amp) >> (amp_shift + x));
			sample_gy = (int32_t)(((gy >> 16) * (int64_t)lay->amp) >> (amp_shift + x));
			if (lay->contrast != 0 && lay->contrast != NOISE_CONTRAST_IDENTITY) {
				val = (val * (int32_t)lay->contrast) >> 8;
				sample_gx = (sample_gx * (int32_t)lay->contrast) >> 8;
				sample_gy = (sample_gy * (int32_t)lay->contrast) >> 8;
			}
			sample = (val * (int32_t)lay->amp) >> amp_shift;
		}

		res.val += sample;
		res.grad_x = (int16_t)(res.grad_x + sample_gx);
		res.grad_y = (int16_t)(res.grad_y + sample_gy);
	}

	return res;
}

/* Generic 3D point query with analytical spatial gradient.
 * Evaluates value N(x, y, z) and continuous spatial gradient (dN/dx, dN/dy, dN/dz)
 * in a single 8-corner pass using the analytical fade derivative.
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
		uint32_t seed = chaff_layer_seed(field, lay, l);
		unsigned shift = lay->shift;
		unsigned vshift = (lay->vshift && lay->vshift < 16u) ? lay->vshift : CHAFF_DEFAULT_VSHIFT;
		int32_t mid = (int32_t)(1u << (15u - vshift));
		int32_t ridge_fold = mid >> 1;
		unsigned amp_shift = 15u - vshift;
		uint16_t mask = COORDMASK_LSHIFT(shift);
		noise_wcache_init(&cache);

		int16_t sx = x, sy = y;
		if (lay->skew[0]) sx = (int16_t)(sx + sy / lay->skew[0]);
		if (lay->skew[1]) sy = (int16_t)(sy - sx / lay->skew[1]);

		int16_t cell_x = (int16_t)(sx & mask);
		int16_t cell_y = (int16_t)(sy & mask);
		int16_t cell_z = (int16_t)(z & mask);
		uint32_t fix = (uint16_t)sx & (uint16_t)~mask;
		uint32_t fiy = (uint16_t)sy & (uint16_t)~mask;
		uint32_t fiz = (uint16_t)z & (uint16_t)~mask;

		/* Normalized intra-cell position in [0..65535] */
		uint32_t tx = (fix << 16) >> shift;
		uint32_t ty = (fiy << 16) >> shift;
		uint32_t tz = (fiz << 16) >> shift;

		uint64_t ux, uy, uz, du_x, du_y, du_z;
		chaff_eval_fade_and_deriv(tx, field->flags, &ux, &du_x);
		chaff_eval_fade_and_deriv(ty, field->flags, &uy, &du_y);
		chaff_eval_fade_and_deriv(tz, field->flags, &uz, &du_z);

		int16_t pt_s[3] = { cell_z, cell_y, cell_x };
		const uint32_t *v = noise_wcorners_periodic(&cache, pt_s, shift, vshift, seed, 3, lay->period_shift);

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
			int32_t val = (int32_t)(raw >> 16) - mid;
			if (lay->seed_mode & CHAFF_LAYER_RIDGED) {
				if (val >= 0) {
					gx = -gx;
					gy = -gy;
					gz = -gz;
				}
				val = ridge_fold - abs(val);
			}
			sample_gx = (int32_t)(((gx >> 16) * (int64_t)lay->amp) >> (amp_shift + shift));
			sample_gy = (int32_t)(((gy >> 16) * (int64_t)lay->amp) >> (amp_shift + shift));
			sample_gz = (int32_t)(((gz >> 16) * (int64_t)lay->amp) >> (amp_shift + shift));
			if (lay->contrast != 0 && lay->contrast != NOISE_CONTRAST_IDENTITY) {
				val = (val * (int32_t)lay->contrast) >> 8;
				sample_gx = (sample_gx * (int32_t)lay->contrast) >> 8;
				sample_gy = (sample_gy * (int32_t)lay->contrast) >> 8;
				sample_gz = (sample_gz * (int32_t)lay->contrast) >> 8;
			}
			sample = (val * (int32_t)lay->amp) >> amp_shift;
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
		if (lay->skew[0] != 0 || lay->skew[1] != 0) {
			chaff_field_t single_layer = *field;
			single_layer.nlayers = 1;
			uint32_t seed = chaff_layer_seed(field, lay, l);
			chaff_layer_t lay_copy = *lay;
			lay_copy.seed_mode = (uint8_t)(lay->seed_mode & CHAFF_LAYER_RIDGED);
			lay_copy.seed_salt = 0;
			single_layer.seed = seed;
			single_layer.layers = &lay_copy;
			for (uint16_t ry = 0; ry < height; ry++) {
				size_t row_off = (size_t)ry * width;
				int16_t py = (int16_t)(origin_y + ry);
				for (uint16_t rx = 0; rx < width; rx++) {
					chaff_sample_3d_t s = chaff_sample_grad_3d(&single_layer, (int16_t)(origin_x + rx), py, z);
					size_t idx = row_off + rx;
					if (out_val) out_val[idx] += s.val;
					if (out_grad_x) out_grad_x[idx] = (int16_t)(out_grad_x[idx] + s.grad_x);
					if (out_grad_y) out_grad_y[idx] = (int16_t)(out_grad_y[idx] + s.grad_y);
				}
			}
			continue;
		}
		uint32_t seed = chaff_layer_seed(field, lay, l);
		unsigned shift = lay->shift;
		unsigned vshift = (lay->vshift && lay->vshift < 16u) ? lay->vshift : CHAFF_DEFAULT_VSHIFT;
		int32_t mid = (int32_t)(1u << (15u - vshift));
		int32_t ridge_fold = mid >> 1;
		unsigned amp_shift = 15u - vshift;
		uint16_t mask = COORDMASK_LSHIFT(shift);
		int32_t step_px = 1 << shift;
		noise_wcache_init(&cache);

		int16_t cell_z = (int16_t)(z & mask);
		uint32_t fiz = (uint16_t)z & (uint16_t)~mask;
		uint32_t tz = (fiz << 16) >> shift;
		uint64_t uz, du_z_unused;
		chaff_eval_fade_and_deriv(tz, field->flags, &uz, &du_z_unused);

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
				const uint32_t *v = noise_wcorners_periodic(&cache, pt_s, shift, vshift, seed, 3, lay->period_shift);

				/* Pre-collapse Z once per 3D lattice cell into a 2D quad in registers */
				int64_t c0 = (int64_t)v[0] + ((((int64_t)v[4] - (int64_t)v[0]) * (int64_t)uz) >> 16);
				int64_t c1 = (int64_t)v[1] + ((((int64_t)v[5] - (int64_t)v[1]) * (int64_t)uz) >> 16);
				int64_t c2 = (int64_t)v[2] + ((((int64_t)v[6] - (int64_t)v[2]) * (int64_t)uz) >> 16);
				int64_t c3 = (int64_t)v[3] + ((((int64_t)v[7] - (int64_t)v[3]) * (int64_t)uz) >> 16);

				int64_t dy_step0 = c2 - c0;
				int64_t dy_step1 = c3 - c1;

/** Maximum strip width in pixels buffered on stack during slice evaluation. */
#define CHAFF_SLICE_STRIP_MAX 256

				int need_grad = (out_grad_x != NULL || out_grad_y != NULL);
				for (int16_t strip_x = px_min; strip_x < px_max; ) {
					int strip_w = px_max - strip_x;
					if (strip_w > CHAFF_SLICE_STRIP_MAX)
						strip_w = CHAFF_SLICE_STRIP_MAX;

					uint64_t cell_ux[CHAFF_SLICE_STRIP_MAX];
					uint64_t cell_du_x[CHAFF_SLICE_STRIP_MAX];
					for (int i = 0; i < strip_w; i++) {
						int16_t px = (int16_t)(strip_x + i);
						uint32_t fix = (uint16_t)px & (uint16_t)~mask;
						uint32_t tx = (fix << 16) >> shift;
						uint64_t du_tmp = 0;
						chaff_eval_fade_and_deriv(tx, field->flags, &cell_ux[i], &du_tmp);
						if (need_grad)
							cell_du_x[i] = du_tmp;
					}

					for (int16_t py = py_min; py < py_max; py++) {
						uint32_t fiy = (uint16_t)py & (uint16_t)~mask;
						uint32_t ty = (fiy << 16) >> shift;
						uint64_t uy, du_y;
						chaff_eval_fade_and_deriv(ty, field->flags, &uy, &du_y);

						int64_t b0 = c0 + (((dy_step0) * (int64_t)uy) >> 16);
						int64_t b1 = c1 + (((dy_step1) * (int64_t)uy) >> 16);
						int64_t diff_b = b1 - b0;
						int64_t ddy = dy_step1 - dy_step0;
						size_t row_offset = (size_t)(py - origin_y) * width;

						if (!need_grad) {
							if (lay->amp == 0) {
								for (int i = 0; i < strip_w; i++) {
									int64_t raw = b0 + ((diff_b * (int64_t)cell_ux[i]) >> 16);
									if (out_val) out_val[row_offset + (size_t)(strip_x + i - origin_x)] += (int32_t)raw;
								}
							} else {
								int32_t amp = (int32_t)lay->amp;
								uint16_t contrast = lay->contrast;
								int has_contrast = (contrast != 0 && contrast != NOISE_CONTRAST_IDENTITY);
								int is_ridged = (lay->seed_mode & CHAFF_LAYER_RIDGED) != 0;
								for (int i = 0; i < strip_w; i++) {
									int64_t raw = b0 + ((diff_b * (int64_t)cell_ux[i]) >> 16);
									int32_t val = (int32_t)(raw >> 16) - mid;
									if (is_ridged)
										val = ridge_fold - abs(val);
									if (has_contrast)
										val = (val * (int32_t)contrast) >> 8;
									if (out_val) out_val[row_offset + (size_t)(strip_x + i - origin_x)] += (val * amp) >> amp_shift;
								}
							}
						} else {
							for (int i = 0; i < strip_w; i++) {
								uint64_t ux = cell_ux[i];
								uint64_t du_x = cell_du_x[i];

								int64_t raw = b0 + ((diff_b * (int64_t)ux) >> 16);
								int64_t dH_dux = diff_b;
								int64_t dH_duy = dy_step0 + ((ddy * (int64_t)ux) >> 16);

								int64_t gx = (dH_dux * (int64_t)du_x) >> 16;
								int64_t gy = (dH_duy * (int64_t)du_y) >> 16;

								size_t idx = row_offset + (size_t)(strip_x + i - origin_x);

								if (lay->amp == 0) {
									if (out_val) out_val[idx] += (int32_t)raw;
									if (out_grad_x) out_grad_x[idx] = (int16_t)(out_grad_x[idx] + (int32_t)(gx >> shift));
									if (out_grad_y) out_grad_y[idx] = (int16_t)(out_grad_y[idx] + (int32_t)(gy >> shift));
								} else {
									int32_t val = (int32_t)(raw >> 16) - mid;
									if (lay->seed_mode & CHAFF_LAYER_RIDGED) {
										if (val >= 0) {
											gx = -gx;
											gy = -gy;
										}
										val = ridge_fold - abs(val);
									}
									int32_t sample_gx = (int32_t)(((gx >> 16) * (int64_t)lay->amp) >> (amp_shift + shift));
									int32_t sample_gy = (int32_t)(((gy >> 16) * (int64_t)lay->amp) >> (amp_shift + shift));
									if (lay->contrast != 0 && lay->contrast != NOISE_CONTRAST_IDENTITY) {
										val = (val * (int32_t)lay->contrast) >> 8;
										sample_gx = (sample_gx * (int32_t)lay->contrast) >> 8;
										sample_gy = (sample_gy * (int32_t)lay->contrast) >> 8;
									}
									int32_t sample = (val * (int32_t)lay->amp) >> amp_shift;

									if (out_val) out_val[idx] += sample;
									if (out_grad_x) out_grad_x[idx] = (int16_t)(out_grad_x[idx] + sample_gx);
									if (out_grad_y) out_grad_y[idx] = (int16_t)(out_grad_y[idx] + sample_gy);
								}
							}
						}
					}

					strip_x = (int16_t)(strip_x + strip_w);
				}
			}
		}
	}
}

/* Generic 4D point query with analytical spatio-temporal gradient.
 * Evaluates value N(x, y, z, w) and continuous 4D gradient (dN/dx, dN/dy, dN/dz, dN/dw)
 * in a single 16-corner pass using the analytical fade derivative.
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
		uint32_t seed = chaff_layer_seed(field, lay, l);
		unsigned shift = lay->shift;
		unsigned vshift = (lay->vshift && lay->vshift < 16u) ? lay->vshift : CHAFF_DEFAULT_VSHIFT;
		int32_t mid = (int32_t)(1u << (15u - vshift));
		int32_t ridge_fold = mid >> 1;
		unsigned amp_shift = 15u - vshift;
		uint16_t mask = COORDMASK_LSHIFT(shift);
		noise_wcache_init(&cache);

		int16_t sx = x, sy = y;
		if (lay->skew[0]) sx = (int16_t)(sx + sy / lay->skew[0]);
		if (lay->skew[1]) sy = (int16_t)(sy - sx / lay->skew[1]);

		int16_t cell_x = (int16_t)(sx & mask);
		int16_t cell_y = (int16_t)(sy & mask);
		int16_t cell_z = (int16_t)(z & mask);
		int16_t cell_w = (int16_t)(w & mask);
		uint32_t fix = (uint16_t)sx & (uint16_t)~mask;
		uint32_t fiy = (uint16_t)sy & (uint16_t)~mask;
		uint32_t fiz = (uint16_t)z & (uint16_t)~mask;
		uint32_t fiw = (uint16_t)w & (uint16_t)~mask;

		/* Normalized intra-cell position in [0..65535] */
		uint32_t tx = (fix << 16) >> shift;
		uint32_t ty = (fiy << 16) >> shift;
		uint32_t tz = (fiz << 16) >> shift;
		uint32_t tw = (fiw << 16) >> shift;

		uint64_t ux, uy, uz, uw, du_x, du_y, du_z, du_w;
		chaff_eval_fade_and_deriv(tx, field->flags, &ux, &du_x);
		chaff_eval_fade_and_deriv(ty, field->flags, &uy, &du_y);
		chaff_eval_fade_and_deriv(tz, field->flags, &uz, &du_z);
		chaff_eval_fade_and_deriv(tw, field->flags, &uw, &du_w);

		int16_t pt_s[4] = { cell_w, cell_z, cell_y, cell_x };
		const uint32_t *v = noise_wcorners_periodic(&cache, pt_s, shift, vshift, seed, 4, lay->period_shift);

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
			int32_t val = (int32_t)(raw >> 16) - mid;
			if (lay->seed_mode & CHAFF_LAYER_RIDGED) {
				if (val >= 0) {
					gx = -gx;
					gy = -gy;
					gz = -gz;
					gw = -gw;
				}
				val = ridge_fold - abs(val);
			}
			sample_gx = (int32_t)(((gx >> 16) * (int64_t)lay->amp) >> (amp_shift + shift));
			sample_gy = (int32_t)(((gy >> 16) * (int64_t)lay->amp) >> (amp_shift + shift));
			sample_gz = (int32_t)(((gz >> 16) * (int64_t)lay->amp) >> (amp_shift + shift));
			sample_gw = (int32_t)(((gw >> 16) * (int64_t)lay->amp) >> (amp_shift + shift));
			if (lay->contrast != 0 && lay->contrast != NOISE_CONTRAST_IDENTITY) {
				val = (val * (int32_t)lay->contrast) >> 8;
				sample_gx = (sample_gx * (int32_t)lay->contrast) >> 8;
				sample_gy = (sample_gy * (int32_t)lay->contrast) >> 8;
				sample_gz = (sample_gz * (int32_t)lay->contrast) >> 8;
				sample_gw = (sample_gw * (int32_t)lay->contrast) >> 8;
			}
			sample = (val * (int32_t)lay->amp) >> amp_shift;
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
		if (lay->skew[0] != 0 || lay->skew[1] != 0) {
			chaff_field_t single_layer = *field;
			single_layer.nlayers = 1;
			uint32_t seed = chaff_layer_seed(field, lay, l);
			chaff_layer_t lay_copy = *lay;
			lay_copy.seed_mode = (uint8_t)(lay->seed_mode & CHAFF_LAYER_RIDGED);
			lay_copy.seed_salt = 0;
			single_layer.seed = seed;
			single_layer.layers = &lay_copy;
			for (uint16_t ry = 0; ry < height; ry++) {
				size_t row_off = (size_t)ry * width;
				int16_t py = (int16_t)(origin_y + ry);
				for (uint16_t rx = 0; rx < width; rx++) {
					chaff_sample_4d_t s = chaff_sample_grad_4d(&single_layer, (int16_t)(origin_x + rx), py, z, w);
					size_t idx = row_off + rx;
					if (out_val) out_val[idx] += s.val;
					if (out_grad_x) out_grad_x[idx] = (int16_t)(out_grad_x[idx] + s.grad_x);
					if (out_grad_y) out_grad_y[idx] = (int16_t)(out_grad_y[idx] + s.grad_y);
					if (out_grad_z) out_grad_z[idx] = (int16_t)(out_grad_z[idx] + s.grad_z);
					if (out_grad_w) out_grad_w[idx] = (int16_t)(out_grad_w[idx] + s.grad_w);
				}
			}
			continue;
		}
		uint32_t seed = chaff_layer_seed(field, lay, l);
		unsigned shift = lay->shift;
		unsigned vshift = (lay->vshift && lay->vshift < 16u) ? lay->vshift : CHAFF_DEFAULT_VSHIFT;
		int32_t mid = (int32_t)(1u << (15u - vshift));
		int32_t ridge_fold = mid >> 1;
		unsigned amp_shift = 15u - vshift;
		uint16_t mask = COORDMASK_LSHIFT(shift);
		int32_t step_px = 1 << shift;
		noise_wcache_init(&cache);

		int16_t cell_z = (int16_t)(z & mask);
		int16_t cell_w = (int16_t)(w & mask);
		uint32_t fiz = (uint16_t)z & (uint16_t)~mask;
		uint32_t fiw = (uint16_t)w & (uint16_t)~mask;
		uint32_t tz = (fiz << 16) >> shift;
		uint32_t tw = (fiw << 16) >> shift;
		uint64_t uz, uw, du_z, du_w;
		chaff_eval_fade_and_deriv(tz, field->flags, &uz, &du_z);
		chaff_eval_fade_and_deriv(tw, field->flags, &uw, &du_w);

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
				const uint32_t *v = noise_wcorners_periodic(&cache, pt_s, shift, vshift, seed, 4, lay->period_shift);

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

				int need_grad = (out_grad_x != NULL || out_grad_y != NULL ||
				                 out_grad_z != NULL || out_grad_w != NULL);
				for (int16_t strip_x = px_min; strip_x < px_max; ) {
					int strip_w = px_max - strip_x;
					if (strip_w > CHAFF_SLICE_STRIP_MAX)
						strip_w = CHAFF_SLICE_STRIP_MAX;

					uint64_t cell_ux[CHAFF_SLICE_STRIP_MAX];
					uint64_t cell_du_x[CHAFF_SLICE_STRIP_MAX];
					for (int i = 0; i < strip_w; i++) {
						int16_t px = (int16_t)(strip_x + i);
						uint32_t fix = (uint16_t)px & (uint16_t)~mask;
						uint32_t tx = (fix << 16) >> shift;
						uint64_t du_tmp = 0;
						chaff_eval_fade_and_deriv(tx, field->flags, &cell_ux[i], &du_tmp);
						if (need_grad)
							cell_du_x[i] = du_tmp;
					}

					for (int16_t py = py_min; py < py_max; py++) {
						uint32_t fiy = (uint16_t)py & (uint16_t)~mask;
						uint32_t ty = (fiy << 16) >> shift;
						uint64_t uy, du_y;
						chaff_eval_fade_and_deriv(ty, field->flags, &uy, &du_y);

						int64_t b0 = c0 + ((dy_step0 * (int64_t)uy) >> 16);
						int64_t b1 = c1 + ((dy_step1 * (int64_t)uy) >> 16);
						int64_t diff_b = b1 - b0;
						int64_t ddy = dy_step1 - dy_step0;
						size_t row_offset = (size_t)(py - origin_y) * width;

						if (!need_grad) {
							if (lay->amp == 0) {
								for (int i = 0; i < strip_w; i++) {
									int64_t raw = b0 + ((diff_b * (int64_t)cell_ux[i]) >> 16);
									if (out_val) out_val[row_offset + (size_t)(strip_x + i - origin_x)] += (int32_t)raw;
								}
							} else {
								int32_t amp = (int32_t)lay->amp;
								uint16_t contrast = lay->contrast;
								int has_contrast = (contrast != 0 && contrast != NOISE_CONTRAST_IDENTITY);
								int is_ridged = (lay->seed_mode & CHAFF_LAYER_RIDGED) != 0;
								for (int i = 0; i < strip_w; i++) {
									int64_t raw = b0 + ((diff_b * (int64_t)cell_ux[i]) >> 16);
									int32_t val = (int32_t)(raw >> 16) - mid;
									if (is_ridged)
										val = ridge_fold - abs(val);
									if (has_contrast)
										val = (val * (int32_t)contrast) >> 8;
									if (out_val) out_val[row_offset + (size_t)(strip_x + i - origin_x)] += (val * amp) >> amp_shift;
								}
							}
						} else {
							int64_t dz0 = dz_c0 + (((dz_c2 - dz_c0) * (int64_t)uy) >> 16);
							int64_t dz1 = dz_c1 + (((dz_c3 - dz_c1) * (int64_t)uy) >> 16);
							int64_t diff_dz = dz1 - dz0;

							int64_t dw_y0 = dw0 + (((dw2 - dw0) * (int64_t)uy) >> 16);
							int64_t dw_y1 = dw1 + (((dw3 - dw1) * (int64_t)uy) >> 16);
							int64_t diff_dw = dw_y1 - dw_y0;

							for (int i = 0; i < strip_w; i++) {
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

								size_t idx = row_offset + (size_t)(strip_x + i - origin_x);

								if (lay->amp == 0) {
									if (out_val) out_val[idx] += (int32_t)raw;
									if (out_grad_x) out_grad_x[idx] = (int16_t)(out_grad_x[idx] + (int32_t)(gx >> shift));
									if (out_grad_y) out_grad_y[idx] = (int16_t)(out_grad_y[idx] + (int32_t)(gy >> shift));
									if (out_grad_z) out_grad_z[idx] = (int16_t)(out_grad_z[idx] + (int32_t)(gz >> shift));
									if (out_grad_w) out_grad_w[idx] = (int16_t)(out_grad_w[idx] + (int32_t)(gw >> shift));
								} else {
									int32_t val = (int32_t)(raw >> 16) - mid;
									if (lay->seed_mode & CHAFF_LAYER_RIDGED) {
										if (val >= 0) {
											gx = -gx;
											gy = -gy;
											gz = -gz;
											gw = -gw;
										}
										val = ridge_fold - abs(val);
									}
									int32_t sample_gx = (int32_t)(((gx >> 16) * (int64_t)lay->amp) >> (amp_shift + shift));
									int32_t sample_gy = (int32_t)(((gy >> 16) * (int64_t)lay->amp) >> (amp_shift + shift));
									int32_t sample_gz = (int32_t)(((gz >> 16) * (int64_t)lay->amp) >> (amp_shift + shift));
									int32_t sample_gw = (int32_t)(((gw >> 16) * (int64_t)lay->amp) >> (amp_shift + shift));
									if (lay->contrast != 0 && lay->contrast != NOISE_CONTRAST_IDENTITY) {
										val = (val * (int32_t)lay->contrast) >> 8;
										sample_gx = (sample_gx * (int32_t)lay->contrast) >> 8;
										sample_gy = (sample_gy * (int32_t)lay->contrast) >> 8;
										sample_gz = (sample_gz * (int32_t)lay->contrast) >> 8;
										sample_gw = (sample_gw * (int32_t)lay->contrast) >> 8;
									}
									int32_t sample = (val * (int32_t)lay->amp) >> amp_shift;

									if (out_val) out_val[idx] += sample;
									if (out_grad_x) out_grad_x[idx] = (int16_t)(out_grad_x[idx] + sample_gx);
									if (out_grad_y) out_grad_y[idx] = (int16_t)(out_grad_y[idx] + sample_gy);
									if (out_grad_z) out_grad_z[idx] = (int16_t)(out_grad_z[idx] + sample_gz);
									if (out_grad_w) out_grad_w[idx] = (int16_t)(out_grad_w[idx] + sample_gw);
								}
							}
						}
					}

					strip_x = (int16_t)(strip_x + strip_w);
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

/* 4-wide SIMD parameterized 2D fixed-point Q15 rotation for int32_t coordinates.
 * Bit-exact with 4 calls to chaff_rot2d_i32_q15. */
static inline void
chaff_rot2d_i32_q15_4x(int32_t x[4], int32_t y[4], int32_t cos_q15, int32_t sin_q15)
{
#if defined(__wasm_simd128__)
	v128_t vx = wasm_v128_load(x);
	v128_t vy = wasm_v128_load(y);
	v128_t vc = wasm_i32x4_splat(cos_q15);
	v128_t vs = wasm_i32x4_splat(sin_q15);
	v128_t nx_lo = wasm_i64x2_shr(
		wasm_i64x2_sub(wasm_i64x2_extmul_low_i32x4(vx, vc),
		               wasm_i64x2_extmul_low_i32x4(vy, vs)), 15);
	v128_t nx_hi = wasm_i64x2_shr(
		wasm_i64x2_sub(wasm_i64x2_extmul_high_i32x4(vx, vc),
		               wasm_i64x2_extmul_high_i32x4(vy, vs)), 15);
	v128_t ny_lo = wasm_i64x2_shr(
		wasm_i64x2_add(wasm_i64x2_extmul_low_i32x4(vx, vs),
		               wasm_i64x2_extmul_low_i32x4(vy, vc)), 15);
	v128_t ny_hi = wasm_i64x2_shr(
		wasm_i64x2_add(wasm_i64x2_extmul_high_i32x4(vx, vs),
		               wasm_i64x2_extmul_high_i32x4(vy, vc)), 15);
	wasm_v128_store(x, wasm_i32x4_shuffle(nx_lo, nx_hi, 0, 2, 4, 6));
	wasm_v128_store(y, wasm_i32x4_shuffle(ny_lo, ny_hi, 0, 2, 4, 6));
#else
	for (int i = 0; i < 4; i++) {
		int64_t x0 = x[i];
		int64_t y0 = y[i];
		x[i] = (int32_t)((x0 * cos_q15 - y0 * sin_q15) >> 15);
		y[i] = (int32_t)((x0 * sin_q15 + y0 * cos_q15) >> 15);
	}
#endif
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

/* Parameterized rugged point evaluation: rotates octave coordinate systems by (cos_q15, sin_q15)
 * per octave to break power-of-two Cartesian lattice resonance, and inverse-rotates the analytical
 * gradients back into world space via the chain rule (R^l)^T. */
static inline chaff_sample_2d_t
chaff_sample_rugged_2d_ex(const chaff_field_t *field, int16_t wx, int16_t wy,
                          int32_t cos_q15, int32_t sin_q15)
{
	chaff_sample_2d_t res;
	res.val = 0;
	res.grad_x = 0;
	res.grad_y = 0;
	if (!field || field->nlayers == 0) return res;

	int16_t cur_x = wx;
	int16_t cur_y = wy;
	int32_t cum_cos_q15 = 32768;
	int32_t cum_sin_q15 = 0;

	for (uint8_t l = 0; l < field->nlayers; l++) {
		const chaff_layer_t *lay = &field->layers[l];
		uint32_t oct_seed = chaff_layer_seed(field, lay, l);
		chaff_layer_t lay_copy = *lay;
		lay_copy.seed_mode = (uint8_t)(lay->seed_mode & CHAFF_LAYER_RIDGED);
		lay_copy.seed_salt = 0;

		chaff_field_t single_layer = {
			.seed = oct_seed,
			.dim = 2,
			.flags = field->flags,
			.nlayers = 1,
			.layers = &lay_copy,
			.seed_prime = field->seed_prime,
			.seed_salt = 0,
			.rot_cos_q15 = 0,
			.rot_sin_q15 = 0
		};

		chaff_sample_2d_t sample = chaff_sample_grad_2d(&single_layer, cur_x, cur_y);
		if (l > 0) {
			/* Inverse rotate local gradient (df/dx', df/dy') back to world space by (R^l)^T */
			chaff_rot2d_q15(&sample.grad_x, &sample.grad_y, cum_cos_q15, -cum_sin_q15);
		}
		res.val += sample.val;
		res.grad_x = (int16_t)(res.grad_x + sample.grad_x);
		res.grad_y = (int16_t)(res.grad_y + sample.grad_y);

		chaff_rot2d_q15(&cur_x, &cur_y, cos_q15, sin_q15);
		int32_t next_cos = (int32_t)(((int64_t)cum_cos_q15 * cos_q15 - (int64_t)cum_sin_q15 * sin_q15) >> 15);
		int32_t next_sin = (int32_t)(((int64_t)cum_sin_q15 * cos_q15 + (int64_t)cum_cos_q15 * sin_q15) >> 15);
		cum_cos_q15 = next_cos;
		cum_sin_q15 = next_sin;
	}

	return res;
}

static inline chaff_sample_2d_t
chaff_sample_rugged_2d(const chaff_field_t *field, int16_t wx, int16_t wy)
{
	int32_t cos_q15 = (field && (field->rot_cos_q15 != 0 || field->rot_sin_q15 != 0))
	                  ? (int32_t)field->rot_cos_q15 : CHAFF_ROT_GOLD_COS_Q15;
	int32_t sin_q15 = (field && (field->rot_cos_q15 != 0 || field->rot_sin_q15 != 0))
	                  ? (int32_t)field->rot_sin_q15 : CHAFF_ROT_GOLD_SIN_Q15;
	return chaff_sample_rugged_2d_ex(field, wx, wy, cos_q15, sin_q15);
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
	return chaff_octave_amplitude_ex(shift, CHAFF_DEFAULT_MIN_SHIFT, CHAFF_DEFAULT_BASE_AMP);
}

/* Settings descriptor for periodic 2D domain sampling */
typedef struct {
	uint8_t  shift;        /* Octave cell wavelength shift (2^shift) */
	uint8_t  period_shift; /* Domain period shift (0 = CHAFF_DEFAULT_PERIOD_SHIFT) */
	uint8_t  vshift;       /* Hash value right-shift (0 = CHAFF_DEFAULT_VSHIFT) */
	uint32_t seed;         /* Hash seed */
	unsigned fade_flags;   /* Fade flags: 0 or NOISE_FADE5 = quintic, NOISE_FADE = cubic, CHAFF_FLAG_LINEAR = linear */
} chaff_periodic_opts_t;

/* Evaluates a continuous 2D periodic octave with domain period 2^period_shift
 * and feature cell wavelength 2^shift (where shift <= period_shift). */
static inline uint32_t
chaff_wpoint_periodic_2d_ex(int32_t x, int32_t y, uint8_t shift, uint8_t period_shift,
                            uint8_t vshift, uint32_t seed, unsigned fade_flags)
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

	uint32_t fx, fy;
	if (fade_flags & CHAFF_FLAG_LINEAR) {
		fx = frac_x;
		fy = frac_y;
	} else if ((fade_flags & NOISE_FADE) && !(fade_flags & NOISE_FADE5)) {
		fx = noise_wfade(frac_x, shift);
		fy = noise_wfade(frac_y, shift);
	} else {
		fx = noise_wfade5(frac_x, shift);
		fy = noise_wfade5(frac_y, shift);
	}

	typedef struct {
		uint16_t kx0, ky0, kx1, ky1;
		uint32_t seed;
		uint8_t vshift, valid;
		uint32_t v00, v10, v01, v11;
	} chaff_per_cache_t;
	static _CHAFF_TLS chaff_per_cache_t s_per_cache[32];
	chaff_per_cache_t *pc = &s_per_cache[(shift ^ (seed >> 8)) & 31u];
	uint32_t v00, v10, v01, v11;
	if (pc->valid && pc->kx0 == kx0 && pc->ky0 == ky0 &&
	    pc->kx1 == kx1 && pc->ky1 == ky1 &&
	    pc->seed == seed && pc->vshift == vshift) {
		v00 = pc->v00;
		v10 = pc->v10;
		v01 = pc->v01;
		v11 = pc->v11;
	} else {
#if NOISE_FAST_XXH32_4B && defined(__wasm_simd128__)
		v128_t vk = wasm_i32x4_make(
			(int32_t)noise_pack2((int16_t)ky0, (int16_t)kx0),
			(int32_t)noise_pack2((int16_t)ky0, (int16_t)kx1),
			(int32_t)noise_pack2((int16_t)ky1, (int16_t)kx0),
			(int32_t)noise_pack2((int16_t)ky1, (int16_t)kx1));
		uint32_t vc[4];
		wasm_v128_store(vc, noise_hash2_v128(vk, seed, vshift));
		v00 = vc[0]; v10 = vc[1]; v01 = vc[2]; v11 = vc[3];
#else
		v00 = (vshift >= 32u) ? 0u : (noise_hash2_1((int16_t)ky0, (int16_t)kx0, seed) >> vshift);
		v10 = (vshift >= 32u) ? 0u : (noise_hash2_1((int16_t)ky0, (int16_t)kx1, seed) >> vshift);
		v01 = (vshift >= 32u) ? 0u : (noise_hash2_1((int16_t)ky1, (int16_t)kx0, seed) >> vshift);
		v11 = (vshift >= 32u) ? 0u : (noise_hash2_1((int16_t)ky1, (int16_t)kx1, seed) >> vshift);
#endif
		pc->kx0 = kx0; pc->ky0 = ky0;
		pc->kx1 = kx1; pc->ky1 = ky1;
		pc->seed = seed; pc->vshift = vshift;
		pc->v00 = v00; pc->v10 = v10;
		pc->v01 = v01; pc->v11 = v11;
		pc->valid = 1;
	}

	uint32_t b0 = noise_wstep(v00, v10, fx, shift);
	uint32_t b1 = noise_wstep(v01, v11, fx, shift);
	return noise_wstep(b0, b1, fy, shift);
}

static inline uint32_t
chaff_wpoint_periodic_2d(int32_t x, int32_t y, uint8_t shift, uint8_t period_shift,
                         uint8_t vshift, uint32_t seed)
{
	return chaff_wpoint_periodic_2d_ex(x, y, shift, period_shift, vshift, seed, NOISE_FADE5);
}

/* 4-wide SIMD continuous 2D periodic octave sampler (NOISE_FADE5).
 * Bit-exact with 4 calls to chaff_wpoint_periodic_2d(x[i], y[i], ...). */
static inline void
chaff_wpoint_periodic_2d_4x(const int32_t x[4], const int32_t y[4],
                            uint8_t shift, uint8_t period_shift,
                            uint8_t vshift, uint32_t seed, uint32_t out[4])
{
	if (period_shift == 0 || shift > period_shift) {
		out[0] = out[1] = out[2] = out[3] = 0u;
		return;
	}
	uint32_t period_mask = (period_shift >= 32) ? 0xFFFFFFFFu : ((1u << period_shift) - 1u);
	uint32_t cell_count_mask = (1u << (period_shift - shift)) - 1u;
	uint32_t mask_frac = (1u << shift) - 1u;

	uint16_t kx0[4], ky0[4], kx1[4], ky1[4];
	uint32_t frac_x[4], frac_y[4];
	for (int i = 0; i < 4; i++) {
		uint32_t ux = (uint32_t)x[i] & period_mask;
		uint32_t uy = (uint32_t)y[i] & period_mask;
		kx0[i] = (uint16_t)(ux >> shift);
		ky0[i] = (uint16_t)(uy >> shift);
		kx1[i] = (uint16_t)((kx0[i] + 1u) & cell_count_mask);
		ky1[i] = (uint16_t)((ky0[i] + 1u) & cell_count_mask);
		frac_x[i] = ux & mask_frac;
		frac_y[i] = uy & mask_frac;
	}

	if (shift == 0u) {
#if NOISE_FAST_XXH32_4B && defined(__wasm_simd128__)
		v128_t vk = wasm_i32x4_make(
			(int32_t)noise_pack2((int16_t)ky0[0], (int16_t)kx0[0]),
			(int32_t)noise_pack2((int16_t)ky0[1], (int16_t)kx0[1]),
			(int32_t)noise_pack2((int16_t)ky0[2], (int16_t)kx0[2]),
			(int32_t)noise_pack2((int16_t)ky0[3], (int16_t)kx0[3]));
		wasm_v128_store(out, noise_hash2_v128(vk, seed, vshift));
#else
		for (int i = 0; i < 4; i++)
			out[i] = (vshift >= 32u) ? 0u : (noise_hash2_1((int16_t)ky0[i], (int16_t)kx0[i], seed) >> vshift);
#endif
		return;
	}

	typedef struct {
		uint16_t kx0, ky0, kx1, ky1;
		uint32_t seed;
		uint8_t vshift, valid;
		uint32_t v00, v10, v01, v11;
	} chaff_per4_cache_t;
	static _CHAFF_TLS chaff_per4_cache_t s_per4_cache[32];
	chaff_per4_cache_t *pc = &s_per4_cache[(shift ^ (seed >> 8)) & 31u];

#if defined(__wasm_simd128__)
	if (shift <= 8u && vshift >= 8u) {
		v128_t vv00, vv10, vv01, vv11;
		if (kx0[0] == kx0[1] && kx0[0] == kx0[2] && kx0[0] == kx0[3] &&
		    ky0[0] == ky0[1] && ky0[0] == ky0[2] && ky0[0] == ky0[3]) {
			uint32_t c00, c10, c01, c11;
			if (pc->valid && pc->kx0 == kx0[0] && pc->ky0 == ky0[0] &&
			    pc->kx1 == kx1[0] && pc->ky1 == ky1[0] &&
			    pc->seed == seed && pc->vshift == vshift) {
				c00 = pc->v00; c10 = pc->v10; c01 = pc->v01; c11 = pc->v11;
			} else {
				v128_t vk = wasm_i32x4_make(
					(int32_t)noise_pack2((int16_t)ky0[0], (int16_t)kx0[0]),
					(int32_t)noise_pack2((int16_t)ky0[0], (int16_t)kx1[0]),
					(int32_t)noise_pack2((int16_t)ky1[0], (int16_t)kx0[0]),
					(int32_t)noise_pack2((int16_t)ky1[0], (int16_t)kx1[0]));
				uint32_t vc[4];
				wasm_v128_store(vc, noise_hash2_v128(vk, seed, vshift));
				c00 = vc[0]; c10 = vc[1]; c01 = vc[2]; c11 = vc[3];
				pc->kx0 = kx0[0]; pc->ky0 = ky0[0];
				pc->kx1 = kx1[0]; pc->ky1 = ky1[0];
				pc->seed = seed; pc->vshift = vshift;
				pc->v00 = c00; pc->v10 = c10; pc->v01 = c01; pc->v11 = c11;
				pc->valid = 1;
			}
			vv00 = wasm_i32x4_splat((int32_t)c00);
			vv10 = wasm_i32x4_splat((int32_t)c10);
			vv01 = wasm_i32x4_splat((int32_t)c01);
			vv11 = wasm_i32x4_splat((int32_t)c11);
		} else {
			uint32_t v00[4], v10[4], v01[4], v11[4];
			for (int i = 0; i < 4; i++) {
				if (i > 0 && kx0[i] == kx0[i - 1] && ky0[i] == ky0[i - 1]) {
					v00[i] = v00[i - 1]; v10[i] = v10[i - 1];
					v01[i] = v01[i - 1]; v11[i] = v11[i - 1];
				} else if (pc->valid && pc->kx0 == kx0[i] && pc->ky0 == ky0[i] &&
				           pc->kx1 == kx1[i] && pc->ky1 == ky1[i] &&
				           pc->seed == seed && pc->vshift == vshift) {
					v00[i] = pc->v00; v10[i] = pc->v10;
					v01[i] = pc->v01; v11[i] = pc->v11;
				} else {
					v128_t vk = wasm_i32x4_make(
						(int32_t)noise_pack2((int16_t)ky0[i], (int16_t)kx0[i]),
						(int32_t)noise_pack2((int16_t)ky0[i], (int16_t)kx1[i]),
						(int32_t)noise_pack2((int16_t)ky1[i], (int16_t)kx0[i]),
						(int32_t)noise_pack2((int16_t)ky1[i], (int16_t)kx1[i]));
					uint32_t vc[4];
					wasm_v128_store(vc, noise_hash2_v128(vk, seed, vshift));
					v00[i] = vc[0]; v10[i] = vc[1]; v01[i] = vc[2]; v11[i] = vc[3];
					pc->kx0 = kx0[i]; pc->ky0 = ky0[i];
					pc->kx1 = kx1[i]; pc->ky1 = ky1[i];
					pc->seed = seed; pc->vshift = vshift;
					pc->v00 = vc[0]; pc->v10 = vc[1]; pc->v01 = vc[2]; pc->v11 = vc[3];
					pc->valid = 1;
				}
			}
			vv00 = wasm_v128_load(v00);
			vv10 = wasm_v128_load(v10);
			vv01 = wasm_v128_load(v01);
			vv11 = wasm_v128_load(v11);
		}
		const unsigned lsh = 9u - shift;
		const unsigned rsh = 16u - shift;
		v128_t fx = wasm_u32x4_shr(wasm_i32x4_make(
			(int32_t)s_noise_fade5_q16_lut[frac_x[0] << lsh],
			(int32_t)s_noise_fade5_q16_lut[frac_x[1] << lsh],
			(int32_t)s_noise_fade5_q16_lut[frac_x[2] << lsh],
			(int32_t)s_noise_fade5_q16_lut[frac_x[3] << lsh]), rsh);
		v128_t fy = wasm_u32x4_shr(wasm_i32x4_make(
			(int32_t)s_noise_fade5_q16_lut[frac_y[0] << lsh],
			(int32_t)s_noise_fade5_q16_lut[frac_y[1] << lsh],
			(int32_t)s_noise_fade5_q16_lut[frac_y[2] << lsh],
			(int32_t)s_noise_fade5_q16_lut[frac_y[3] << lsh]), rsh);
		v128_t b0 = noise_wstep_v128_u24(vv00, vv10, fx, shift);
		v128_t b1 = noise_wstep_v128_u24(vv01, vv11, fx, shift);
		wasm_v128_store(out, noise_wstep_v128_u24(b0, b1, fy, shift));
		return;
	}
#endif
	if (kx0[0] == kx0[1] && kx0[0] == kx0[2] && kx0[0] == kx0[3] &&
	    ky0[0] == ky0[1] && ky0[0] == ky0[2] && ky0[0] == ky0[3]) {
		uint32_t c00, c10, c01, c11;
		if (pc->valid && pc->kx0 == kx0[0] && pc->ky0 == ky0[0] &&
		    pc->kx1 == kx1[0] && pc->ky1 == ky1[0] &&
		    pc->seed == seed && pc->vshift == vshift) {
			c00 = pc->v00; c10 = pc->v10; c01 = pc->v01; c11 = pc->v11;
		} else {
#if NOISE_FAST_XXH32_4B && defined(__wasm_simd128__)
			v128_t vk = wasm_i32x4_make(
				(int32_t)noise_pack2((int16_t)ky0[0], (int16_t)kx0[0]),
				(int32_t)noise_pack2((int16_t)ky0[0], (int16_t)kx1[0]),
				(int32_t)noise_pack2((int16_t)ky1[0], (int16_t)kx0[0]),
				(int32_t)noise_pack2((int16_t)ky1[0], (int16_t)kx1[0]));
			uint32_t vc[4];
			wasm_v128_store(vc, noise_hash2_v128(vk, seed, vshift));
			c00 = vc[0]; c10 = vc[1]; c01 = vc[2]; c11 = vc[3];
#else
			c00 = (vshift >= 32u) ? 0u : (noise_hash2_1((int16_t)ky0[0], (int16_t)kx0[0], seed) >> vshift);
			c10 = (vshift >= 32u) ? 0u : (noise_hash2_1((int16_t)ky0[0], (int16_t)kx1[0], seed) >> vshift);
			c01 = (vshift >= 32u) ? 0u : (noise_hash2_1((int16_t)ky1[0], (int16_t)kx0[0], seed) >> vshift);
			c11 = (vshift >= 32u) ? 0u : (noise_hash2_1((int16_t)ky1[0], (int16_t)kx1[0], seed) >> vshift);
#endif
			pc->kx0 = kx0[0]; pc->ky0 = ky0[0];
			pc->kx1 = kx1[0]; pc->ky1 = ky1[0];
			pc->seed = seed; pc->vshift = vshift;
			pc->v00 = c00; pc->v10 = c10; pc->v01 = c01; pc->v11 = c11;
			pc->valid = 1;
		}
		for (int i = 0; i < 4; i++) {
			uint32_t fx = noise_wfade5(frac_x[i], shift);
			uint32_t fy = noise_wfade5(frac_y[i], shift);
			uint32_t b0 = noise_wstep(c00, c10, fx, shift);
			uint32_t b1 = noise_wstep(c01, c11, fx, shift);
			out[i] = noise_wstep(b0, b1, fy, shift);
		}
	} else {
		uint32_t c00 = 0, c10 = 0, c01 = 0, c11 = 0;
		for (int i = 0; i < 4; i++) {
			if (i == 0 || kx0[i] != kx0[i - 1] || ky0[i] != ky0[i - 1]) {
				if (pc->valid && pc->kx0 == kx0[i] && pc->ky0 == ky0[i] &&
				    pc->kx1 == kx1[i] && pc->ky1 == ky1[i] &&
				    pc->seed == seed && pc->vshift == vshift) {
					c00 = pc->v00; c10 = pc->v10; c01 = pc->v01; c11 = pc->v11;
				} else {
#if NOISE_FAST_XXH32_4B && defined(__wasm_simd128__)
					v128_t vk = wasm_i32x4_make(
						(int32_t)noise_pack2((int16_t)ky0[i], (int16_t)kx0[i]),
						(int32_t)noise_pack2((int16_t)ky0[i], (int16_t)kx1[i]),
						(int32_t)noise_pack2((int16_t)ky1[i], (int16_t)kx0[i]),
						(int32_t)noise_pack2((int16_t)ky1[i], (int16_t)kx1[i]));
					uint32_t vc[4];
					wasm_v128_store(vc, noise_hash2_v128(vk, seed, vshift));
					c00 = vc[0]; c10 = vc[1]; c01 = vc[2]; c11 = vc[3];
#else
					c00 = (vshift >= 32u) ? 0u : (noise_hash2_1((int16_t)ky0[i], (int16_t)kx0[i], seed) >> vshift);
					c10 = (vshift >= 32u) ? 0u : (noise_hash2_1((int16_t)ky0[i], (int16_t)kx1[i], seed) >> vshift);
					c01 = (vshift >= 32u) ? 0u : (noise_hash2_1((int16_t)ky1[i], (int16_t)kx0[i], seed) >> vshift);
					c11 = (vshift >= 32u) ? 0u : (noise_hash2_1((int16_t)ky1[i], (int16_t)kx1[i], seed) >> vshift);
#endif
					pc->kx0 = kx0[i]; pc->ky0 = ky0[i];
					pc->kx1 = kx1[i]; pc->ky1 = ky1[i];
					pc->seed = seed; pc->vshift = vshift;
					pc->v00 = c00; pc->v10 = c10; pc->v01 = c01; pc->v11 = c11;
					pc->valid = 1;
				}
			}
			uint32_t fx = noise_wfade5(frac_x[i], shift);
			uint32_t fy = noise_wfade5(frac_y[i], shift);
			uint32_t b0 = noise_wstep(c00, c10, fx, shift);
			uint32_t b1 = noise_wstep(c01, c11, fx, shift);
			out[i] = noise_wstep(b0, b1, fy, shift);
		}
	}
}

/**
 * Parameterized periodic 2D domain sampler with explicit period_shift and vshift.
 * Clamps shift to period_shift - 1 so at least 2 lattice cells span the period.
 */
static inline uint32_t
chaff_sample_periodic_2d(int32_t rx, int32_t ry, uint8_t shift,
                         uint8_t period_shift, uint8_t vshift, uint32_t seed)
{
	if (period_shift == 0) return 0;
	uint8_t max_s = (period_shift > 1) ? (uint8_t)(period_shift - 1u) : 0;
	uint8_t s = (shift > max_s) ? max_s : shift;
	return chaff_wpoint_periodic_2d(rx, ry, s, period_shift, vshift ? vshift : CHAFF_DEFAULT_VSHIFT, seed);
}

static inline uint32_t
chaff_sample_periodic_2d_opts(const chaff_periodic_opts_t *opts, int32_t rx, int32_t ry)
{
	if (!opts) return 0;
	uint8_t ps = opts->period_shift ? opts->period_shift : CHAFF_DEFAULT_PERIOD_SHIFT;
	uint8_t vs = opts->vshift ? opts->vshift : CHAFF_DEFAULT_VSHIFT;
	uint8_t max_s = (ps > 1) ? (uint8_t)(ps - 1u) : 0;
	uint8_t s = (opts->shift > max_s) ? max_s : opts->shift;
	return chaff_wpoint_periodic_2d_ex(rx, ry, s, ps, vs, opts->seed, opts->fade_flags);
}

/* Convenience wrapper for 16-bit periodic 2D domain (period = 2^16, vshift = 8) */
static inline uint32_t
chaff_sample_torus_macro(int32_t rx, int32_t ry, uint8_t shift, uint32_t seed)
{
	return chaff_wpoint_periodic_2d(rx, ry, shift, CHAFF_DEFAULT_PERIOD_SHIFT, CHAFF_DEFAULT_VSHIFT, seed);
}

static inline uint32_t
chaff_sample_mega_2d(int32_t rx, int32_t ry, unsigned shift, unsigned seed)
{
	return chaff_sample_periodic_2d(rx, ry, (uint8_t)shift, CHAFF_DEFAULT_PERIOD_SHIFT, CHAFF_DEFAULT_VSHIFT, seed);
}

/* Settings descriptor for 2D rotated fractal octave accumulation */
typedef struct {
	uint8_t  min_shift;        /* Minimum octave shift (inclusive) */
	uint8_t  max_shift;        /* Maximum octave shift (inclusive) */
	uint8_t  ridged_min_shift; /* Ridged folding threshold (pass 255 to disable) */
	uint8_t  amp_min_shift;    /* Reference shift for amplitude scaling */
	int16_t  base_amp;         /* Base amplitude at amp_min_shift (0 = CHAFF_DEFAULT_BASE_AMP) */
	uint32_t seed;             /* Primary seed */
	uint32_t seed_salt;        /* Per-octave salt offset */
	uint32_t seed_prime;       /* Per-octave prime (0 = CHAFF_SEED_PRIME) */
	int32_t  cos_q15;          /* Octave rotation cos in Q15 (0 when sin_q15==0 -> CHAFF_ROT_GOLD_COS_Q15) */
	int32_t  sin_q15;          /* Octave rotation sin in Q15 (0 when cos_q15==0 -> CHAFF_ROT_GOLD_SIN_Q15) */
	uint8_t  vshift;           /* Value shift (0 = CHAFF_DEFAULT_VSHIFT) */
	unsigned fade_flags;       /* Fade flags (0 = NOISE_FADE5) */
	uint8_t  out_shift;        /* Output fractional shift (e.g. CHAFF_DEFAULT_OUT_SHIFT = 4 for Q12.4) */
	uint8_t  period_shift;     /* Optional periodic domain shift (0 = aperiodic) */
	int32_t  step_q16;         /* Optional sampling stride in Q16.16 for continuous Nyquist low-pass (0 = point sample) */
} chaff_fbm_opts_t;

/* Struct-driven 2D rotated fractal octave accumulator */
static inline int32_t
chaff_fbm_rotated_2d(const chaff_fbm_opts_t *opts, int32_t x, int32_t y)
{
	if (!opts || opts->min_shift > opts->max_shift) return 0;
	static _CHAFF_TLS noise_wcache_t s_fbm_wcache[64];
	unsigned vs = (opts->vshift && opts->vshift < 16u) ? (unsigned)opts->vshift : CHAFF_DEFAULT_VSHIFT;
	unsigned fade_fl = opts->fade_flags ? opts->fade_flags : NOISE_FADE5;
	int16_t base_amp = (opts->base_amp != 0) ? opts->base_amp : CHAFF_DEFAULT_BASE_AMP;
	uint32_t prime = opts->seed_prime ? opts->seed_prime : CHAFF_SEED_PRIME;
	int32_t cos_q15 = (opts->cos_q15 != 0 || opts->sin_q15 != 0) ? opts->cos_q15 : CHAFF_ROT_GOLD_COS_Q15;
	int32_t sin_q15 = (opts->cos_q15 != 0 || opts->sin_q15 != 0) ? opts->sin_q15 : CHAFF_ROT_GOLD_SIN_Q15;
	int32_t mid = (int32_t)(1u << (15u - vs));
	int32_t ridge_fold = mid >> 1;
	int32_t cur_x = x;
	int32_t cur_y = y;
	int32_t acc = 0;
	uint32_t bank = ((opts->seed_salt ^ (opts->seed >> 12)) & 3u) << 4;

	for (int8_t s = (int8_t)opts->max_shift; s >= (int8_t)opts->min_shift; s--) {
		uint32_t w_q16 = chaff_octave_weight_q16((uint8_t)s, opts->step_q16);
		if (w_q16 == 0u)
			break;
		int16_t amp = chaff_octave_amplitude_ex((uint8_t)s, opts->amp_min_shift, base_amp);
		uint32_t oct_seed = opts->seed ^ (uint32_t)((uint8_t)s * prime + opts->seed_salt);
		int16_t p[2] = { (int16_t)cur_y, (int16_t)cur_x };
		noise_wcache_t *wc = &s_fbm_wcache[(bank + ((uint32_t)s & 15u)) & 63u];
		uint32_t raw = noise_wpoint_c_periodic(wc, p, (uint8_t)s, vs, oct_seed, 2, fade_fl, opts->period_shift);
		int32_t val = (int32_t)(raw >> 16) - mid;
		if ((uint8_t)s >= opts->ridged_min_shift) {
			val = ridge_fold - abs(val);
		}
		int64_t term = (((int64_t)val * amp) << opts->out_shift) / mid;
		if (w_q16 < 65536u) {
			term = (term * (int64_t)w_q16) >> 16;
		}
		acc += (int32_t)term;
		chaff_rot2d_i32_q15(&cur_x, &cur_y, cos_q15, sin_q15);
	}
	return acc;
}

/* 4-wide SIMD struct-driven 2D rotated fractal octave accumulator.
 * Bit-exact with 4 calls to chaff_fbm_rotated_2d(opts, x[i], y[i]). */
static inline void
chaff_fbm_rotated_2d_4x(const chaff_fbm_opts_t *opts,
                        const int32_t x[4], const int32_t y[4],
                        int32_t out[4])
{
	if (!opts || opts->min_shift > opts->max_shift) {
		out[0] = out[1] = out[2] = out[3] = 0;
		return;
	}
	unsigned fade_fl = opts->fade_flags ? opts->fade_flags : NOISE_FADE5;
	if (opts->period_shift != 0u || (fade_fl & (NOISE_FADE | NOISE_FADE5)) != NOISE_FADE5) {
		for (int i = 0; i < 4; i++)
			out[i] = chaff_fbm_rotated_2d(opts, x[i], y[i]);
		return;
	}
	static _CHAFF_TLS noise_wcache_t s_fbm4_wcache[64];
	unsigned vs = (opts->vshift && opts->vshift < 16u) ? (unsigned)opts->vshift : CHAFF_DEFAULT_VSHIFT;
	unsigned mid_shift = 15u - vs;
	int32_t mid = (int32_t)(1u << mid_shift);
	int64_t mid_mask = (int64_t)mid - 1;
	int32_t ridge_fold = mid >> 1;
	int16_t base_amp = (opts->base_amp != 0) ? opts->base_amp : CHAFF_DEFAULT_BASE_AMP;
	uint32_t prime = opts->seed_prime ? opts->seed_prime : CHAFF_SEED_PRIME;
	int32_t cos_q15 = (opts->cos_q15 != 0 || opts->sin_q15 != 0) ? opts->cos_q15 : CHAFF_ROT_GOLD_COS_Q15;
	int32_t sin_q15 = (opts->cos_q15 != 0 || opts->sin_q15 != 0) ? opts->sin_q15 : CHAFF_ROT_GOLD_SIN_Q15;
	uint32_t bank = ((opts->seed_salt ^ (opts->seed >> 12)) & 3u) << 4;

	int32_t cur_x[4] = { x[0], x[1], x[2], x[3] };
	int32_t cur_y[4] = { y[0], y[1], y[2], y[3] };
	int32_t acc[4] = { 0, 0, 0, 0 };

	for (int8_t s = (int8_t)opts->max_shift; s >= (int8_t)opts->min_shift; s--) {
		uint32_t w_q16 = chaff_octave_weight_q16((uint8_t)s, opts->step_q16);
		if (w_q16 == 0u)
			break;
		int16_t amp = chaff_octave_amplitude_ex((uint8_t)s, opts->amp_min_shift, base_amp);
		uint32_t oct_seed = opts->seed ^ (uint32_t)((uint8_t)s * prime + opts->seed_salt);
		int16_t py[4] = { (int16_t)cur_y[0], (int16_t)cur_y[1], (int16_t)cur_y[2], (int16_t)cur_y[3] };
		int16_t px[4] = { (int16_t)cur_x[0], (int16_t)cur_x[1], (int16_t)cur_x[2], (int16_t)cur_x[3] };
		noise_wcache_t *wc = &s_fbm4_wcache[(bank + ((uint32_t)s & 15u)) & 63u];
		uint32_t raw[4];
		noise_wpoint_2d_4x(wc, py, px, (uint8_t)s, vs, oct_seed, NOISE_FADE5, raw);
		int is_ridged = ((uint8_t)s >= opts->ridged_min_shift);
		for (int i = 0; i < 4; i++) {
			int32_t val = (int32_t)(raw[i] >> 16) - mid;
			if (is_ridged)
				val = ridge_fold - (val < 0 ? -val : val);
			int64_t num = ((int64_t)val * amp) << opts->out_shift;
			/* Exact signed division by power-of-2 mid (truncating toward zero) */
			int64_t term = (num + ((num >> 63) & mid_mask)) >> mid_shift;
			if (w_q16 < 65536u)
				term = (term * (int64_t)w_q16) >> 16;
			acc[i] += (int32_t)term;
		}
		if (s > (int8_t)opts->min_shift)
			chaff_rot2d_i32_q15_4x(cur_x, cur_y, cos_q15, sin_q15);
	}
	out[0] = acc[0]; out[1] = acc[1]; out[2] = acc[2]; out[3] = acc[3];
}

/* Parameterized 2D rotated fractal octave accumulator with configurable vshift, fade_flags,
 * and fixed-point fractional output shift (out_shift = 4 produces Q12.4). */
static inline int32_t
chaff_fbm_rotated_2d_ex(int32_t x, int32_t y,
                        uint8_t min_shift, uint8_t max_shift,
                        uint8_t ridged_min_shift,
                        uint8_t amp_min_shift, int16_t base_amp,
                        uint32_t seed, uint32_t seed_salt,
                        int32_t cos_q15, int32_t sin_q15,
                        uint8_t vshift, unsigned fade_flags, uint8_t out_shift)
{
	if (min_shift > max_shift || base_amp <= 0) return 0;
	chaff_fbm_opts_t opts = {
		.min_shift = min_shift,
		.max_shift = max_shift,
		.ridged_min_shift = ridged_min_shift,
		.amp_min_shift = amp_min_shift,
		.base_amp = base_amp,
		.seed = seed,
		.seed_salt = seed_salt,
		.seed_prime = CHAFF_SEED_PRIME,
		.cos_q15 = cos_q15,
		.sin_q15 = sin_q15,
		.vshift = vshift,
		.fade_flags = fade_flags,
		.out_shift = out_shift,
		.step_q16 = 0
	};
	return chaff_fbm_rotated_2d(&opts, x, y);
}

/* Scale-filtered 2D rotated fractal octave accumulator in Q12.4 fixed-point with continuous Nyquist step_q16. */
static inline int32_t
chaff_fbm_rotated_2d_step_q12_4(int32_t x, int32_t y,
                                int32_t step_q16,
                                uint8_t min_shift, uint8_t max_shift,
                                uint8_t ridged_min_shift,
                                uint8_t amp_min_shift, int16_t base_amp,
                                uint32_t seed, uint32_t seed_salt,
                                int32_t cos_q15, int32_t sin_q15)
{
	if (min_shift > max_shift || base_amp <= 0) return 0;
	chaff_fbm_opts_t opts = {
		.min_shift = min_shift,
		.max_shift = max_shift,
		.ridged_min_shift = ridged_min_shift,
		.amp_min_shift = amp_min_shift,
		.base_amp = base_amp,
		.seed = seed,
		.seed_salt = seed_salt,
		.seed_prime = CHAFF_SEED_PRIME,
		.cos_q15 = cos_q15,
		.sin_q15 = sin_q15,
		.vshift = CHAFF_DEFAULT_VSHIFT,
		.fade_flags = NOISE_FADE5,
		.out_shift = CHAFF_DEFAULT_OUT_SHIFT,
		.step_q16 = step_q16
	};
	return chaff_fbm_rotated_2d(&opts, x, y);
}

/* 4-wide SIMD scale-filtered 2D rotated fractal octave accumulator in Q12.4 fixed-point.
 * Bit-exact with 4 calls to chaff_fbm_rotated_2d_step_q12_4. */
static inline void
chaff_fbm_rotated_2d_step_q12_4_4x(const int32_t x[4], const int32_t y[4],
                                   int32_t step_q16,
                                   uint8_t min_shift, uint8_t max_shift,
                                   uint8_t ridged_min_shift,
                                   uint8_t amp_min_shift, int16_t base_amp,
                                   uint32_t seed, uint32_t seed_salt,
                                   int32_t cos_q15, int32_t sin_q15,
                                   int32_t out[4])
{
	if (min_shift > max_shift || base_amp <= 0) {
		out[0] = out[1] = out[2] = out[3] = 0;
		return;
	}
	chaff_fbm_opts_t opts = {
		.min_shift = min_shift,
		.max_shift = max_shift,
		.ridged_min_shift = ridged_min_shift,
		.amp_min_shift = amp_min_shift,
		.base_amp = base_amp,
		.seed = seed,
		.seed_salt = seed_salt,
		.seed_prime = CHAFF_SEED_PRIME,
		.cos_q15 = cos_q15,
		.sin_q15 = sin_q15,
		.vshift = CHAFF_DEFAULT_VSHIFT,
		.fade_flags = NOISE_FADE5,
		.out_shift = CHAFF_DEFAULT_OUT_SHIFT,
		.step_q16 = step_q16
	};
	chaff_fbm_rotated_2d_4x(&opts, x, y, out);
}

/* 4-wide batch 2D analytical gradient noise evaluator.
 * Reuses cached 2D lattice corners across lanes sharing a cell (with SIMD 4-corner hash on miss).
 * Bit-exact with 4 calls to chaff_sample_grad_2d(field, wx[i], wy[i]). */
static inline void
chaff_grad_noise2d_4x(const chaff_field_t *field,
                      const int16_t wx[4], const int16_t wy[4],
                      chaff_sample_2d_t out[4])
{
	for (int i = 0; i < 4; i++) {
		out[i].val = 0;
		out[i].grad_x = 0;
		out[i].grad_y = 0;
	}
	if (!field || field->dim < 2) return;

	noise_wcache_t cache;
	for (uint8_t l = 0; l < field->nlayers; l++) {
		const chaff_layer_t *lay = &field->layers[l];
		uint32_t seed = chaff_layer_seed(field, lay, l);
		unsigned x = lay->shift;
		unsigned vshift = (lay->vshift && lay->vshift < 16u) ? lay->vshift : CHAFF_DEFAULT_VSHIFT;
		int32_t mid = (int32_t)(1u << (15u - vshift));
		int32_t ridge_fold = mid >> 1;
		unsigned amp_shift = 15u - vshift;
		uint16_t mask = COORDMASK_LSHIFT(x);
		noise_wcache_init(&cache);

		for (int i = 0; i < 4; i++) {
			int16_t sx = wx[i], sy = wy[i];
			if (lay->skew[0]) sx = (int16_t)(sx + sy / lay->skew[0]);
			if (lay->skew[1]) sy = (int16_t)(sy - sx / lay->skew[1]);

			int16_t cell_x = (int16_t)(sx & mask);
			int16_t cell_y = (int16_t)(sy & mask);
			uint32_t fix = (uint16_t)sx & (uint16_t)~mask;
			uint32_t fiy = (uint16_t)sy & (uint16_t)~mask;

			uint32_t tx = (fix << 16) >> x;
			uint32_t ty = (fiy << 16) >> x;

			uint64_t ux, uy, du_x, du_y;
			chaff_eval_fade_and_deriv(tx, field->flags, &ux, &du_x);
			chaff_eval_fade_and_deriv(ty, field->flags, &uy, &du_y);

			int16_t pt_s[2] = { cell_y, cell_x };
			const uint32_t *v = noise_wcorners_periodic(&cache, pt_s, x, vshift, seed, 2, lay->period_shift);
			int64_t v00 = (int64_t)v[0];
			int64_t v10 = (int64_t)v[1];
			int64_t v01 = (int64_t)v[2];
			int64_t v11 = (int64_t)v[3];

			int64_t b0 = v00 + (((v01 - v00) * (int64_t)uy) >> 16);
			int64_t b1 = v10 + (((v11 - v10) * (int64_t)uy) >> 16);
			int64_t raw = b0 + (((b1 - b0) * (int64_t)ux) >> 16);

			int64_t dH_dux = b1 - b0;
			int64_t c0 = v00 + (((v10 - v00) * (int64_t)ux) >> 16);
			int64_t c1 = v01 + (((v11 - v01) * (int64_t)ux) >> 16);
			int64_t dH_duy = c1 - c0;

			int64_t gx = (dH_dux * (int64_t)du_x) >> 16;
			int64_t gy = (dH_duy * (int64_t)du_y) >> 16;

			int32_t sample;
			int32_t sample_gx, sample_gy;
			if (lay->amp == 0) {
				sample = (int32_t)raw;
				sample_gx = (int32_t)(gx >> x);
				sample_gy = (int32_t)(gy >> x);
			} else {
				int32_t val = (int32_t)(raw >> 16) - mid;
				if (lay->seed_mode & CHAFF_LAYER_RIDGED) {
					if (val >= 0) {
						gx = -gx;
						gy = -gy;
					}
					val = ridge_fold - abs(val);
				}
				sample_gx = (int32_t)(((gx >> 16) * (int64_t)lay->amp) >> (amp_shift + x));
				sample_gy = (int32_t)(((gy >> 16) * (int64_t)lay->amp) >> (amp_shift + x));
				if (lay->contrast != 0 && lay->contrast != NOISE_CONTRAST_IDENTITY) {
					val = (val * (int32_t)lay->contrast) >> 8;
					sample_gx = (sample_gx * (int32_t)lay->contrast) >> 8;
					sample_gy = (sample_gy * (int32_t)lay->contrast) >> 8;
				}
				sample = (val * (int32_t)lay->amp) >> amp_shift;
			}

			out[i].val += sample;
			out[i].grad_x = (int16_t)(out[i].grad_x + sample_gx);
			out[i].grad_y = (int16_t)(out[i].grad_y + sample_gy);
		}
	}
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
	return chaff_fbm_rotated_2d_ex(x, y, min_shift, max_shift, ridged_min_shift,
	                               amp_min_shift, base_amp, seed, seed_salt,
	                               cos_q15, sin_q15, CHAFF_DEFAULT_VSHIFT, NOISE_FADE5, CHAFF_DEFAULT_OUT_SHIFT);
}

/* --- Divergence-Free 2D Advection & Curl-Flow Primitives ----------------- */

typedef struct {
	int16_t base_vx_q8;  /* Base advection velocity X in Q8.8 */
	int16_t base_vy_q8;  /* Base advection velocity Y in Q8.8 */
	int16_t curl_amp_q8; /* Curl streamfunction amplitude weight in Q8.8 */
} chaff_flow_desc_t;

typedef struct {
	int16_t vx_q8;       /* Total 2D flow velocity X = base_vx + dPsi/dy */
	int16_t vy_q8;       /* Total 2D flow velocity Y = base_vy - dPsi/dx */
	int32_t psi;         /* Streamfunction scalar potential Psi */
} chaff_flow_2d_t;

/* Evaluates a divergence-free 2D curl-flow vector field from a 3D streamfunction
 * field Psi(x, y, t): v = v_base + curl(Psi) = (base_vx + dPsi/dy, base_vy - dPsi/dx).
 * Because div(curl(Psi)) = d^2Psi/dxdy - d^2Psi/dydx = 0 identically, the resulting
 * velocity field is guaranteed divergence-free for any streamfunction Psi. */
static inline chaff_flow_2d_t
chaff_sample_flow_2d(const chaff_field_t *psi_field,
                     const chaff_flow_desc_t *desc,
                     int16_t x, int16_t y, int16_t t)
{
	chaff_flow_2d_t out;
	out.vx_q8 = desc ? desc->base_vx_q8 : 0;
	out.vy_q8 = desc ? desc->base_vy_q8 : 0;
	out.psi = 0;

	if (!psi_field || !desc) return out;

	chaff_sample_3d_t s = chaff_sample_grad_3d(psi_field, x, y, t);
	out.psi = s.val;

	int32_t curl_vx = ((int32_t)s.grad_y * (int32_t)desc->curl_amp_q8) >> 8;
	int32_t curl_vy = (-(int32_t)s.grad_x * (int32_t)desc->curl_amp_q8) >> 8;

	out.vx_q8 = (int16_t)(out.vx_q8 + curl_vx);
	out.vy_q8 = (int16_t)(out.vy_q8 + curl_vy);
	return out;
}

#endif /* TTYPT_CHAFF_H */
