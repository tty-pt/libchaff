#ifndef TTYPT_NOISE_H
#define TTYPT_NOISE_H

/*
 * ttypt/noise.h — Standalone Generic N-Dimensional Procedural Noise Library.
 *
 * A noise algorithm that wraps well around data type limits.
 *
 * LICENSE: BSD-2-Clause
 * Copyright (c) 2026, tty-pt
 *
 * DEFINITIONS:
 *
 * A point has DIM coordinates (int16_t).
 * A matrix has 2^y (cy) edge length and (2^cy)^dim total elements.
 * A noise feature quad has 2^x edge length and 2^dim vertices with random values.
 * Other values are the result of a fade (multilinear smoothstep) between those.
 *
 * Provides:
 *   1. Dense Matrix Stepper (noise_oct, noise_m, noise_quad): bulk hypercube generation.
 *   2. Generic Point Sampler (noise_point_n, noise_point_2d, noise_point_3d): O(1) zero-allocation point queries.
 *   3. Wrapping Original-Algorithm Sampler (noise_wpoint_n, noise_wpoint_c,
 *      noise_wrow): point / cached / row-stepped queries that reproduce the
 *      matrix stepper exactly, so coordinates (e.g. an ever-increasing time
 *      counter cast to int16_t) may keep incrementing across the data-type
 *      boundary without a seam.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

#ifdef __has_include
  #if __has_include(<xxhash.h>)
    /* Inline the implementation: a bare XXH32 reference becomes an
     * unresolved env import in WASM builds, which hosts stub to 0 and
     * flatten every noise sample to a constant. */
    #ifndef XXH_INLINE_ALL
      #define XXH_INLINE_ALL
    #endif
    /* Byte-wise reads: the default packed-union reader type-puns the
     * int16_t lattice key as uint32 and clang -O3 -flto (WASM) drops the
     * key stores under strict aliasing, so every sample read 0. Method 3
     * is alias-safe, libc-free, and yields the identical hash. */
    #ifndef XXH_FORCE_MEMORY_ACCESS
      #define XXH_FORCE_MEMORY_ACCESS 3
    #endif
    #include <xxhash.h>
    #define TTYPT_NOISE_HASH XXH32
  #endif
#endif

#ifndef TTYPT_NOISE_HASH
static inline uint32_t
ttypt_noise_mix32(const void *key, size_t len, uint32_t seed)
{
	const uint8_t *data = (const uint8_t *)key;
	uint32_t h = seed ^ 0x9E3779B1u;
	for (size_t i = 0; i < len; i++) {
		h ^= (uint32_t)data[i] * 0x85EBCA6Bu;
		h ^= h >> 13;
		h *= 0xC2B2AE35u;
		h ^= h >> 16;
	}
	return h;
}
#define TTYPT_NOISE_HASH ttypt_noise_mix32
#endif

#if defined(XXH_VERSION_NUMBER) && (!defined(__BYTE_ORDER__) || __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define NOISE_FAST_XXH32_4B 1
#else
#define NOISE_FAST_XXH32_4B 0
#endif

#define NOISE_MAX ((uint32_t) -1)
#define NOISE_MAX_DIM 4u
#define NOISE_DEFAULT_VSHIFT 8u
#define COORDMASK_LSHIFT(x) (uint16_t) ((((uint16_t) -1) << (x)))

typedef struct {
	unsigned x; /* 2^x = side length of noise quad */
	unsigned w; /* weight / bitshift divider */
} octave_t;

/* Euclidean integer floor division */
static inline int
noise_fdiv(int a, int b)
{
	int d = a / b;
	return (a ^ b) < 0 && (d * b != a) ? d - 1 : d;
}

/* Pack 2D lattice coordinate (p0, p1) into little-endian 4-byte key */
static inline uint32_t
noise_pack2(int16_t p0, int16_t p1)
{
	return (uint32_t)(uint16_t)p0 | ((uint32_t)(uint16_t)p1 << 16u);
}

/* Branchless 4-byte XXH32 matching XXH32(int16_t[2]{p0,p1}, 4, seed) bit-for-bit */
static inline uint32_t
noise_hash2_1(int16_t p0, int16_t p1, uint32_t seed)
{
#if NOISE_FAST_XXH32_4B
	uint32_t val = noise_pack2(p0, p1);
	uint32_t h = seed + 0x165667B1u + 4u;
	h += val * 0xC2B2AE3Du;
	h = ((h << 17u) | (h >> 15u)) * 0x27D4EB2Fu;
	h ^= h >> 15u;
	h *= 0x85EBCA77u;
	h ^= h >> 13u;
	h *= 0xC2B2AE3Du;
	h ^= h >> 16u;
	return h;
#else
	int16_t k[2] = { p0, p1 };
	return TTYPT_NOISE_HASH(k, sizeof(k), seed);
#endif
}

#if NOISE_FAST_XXH32_4B && defined(__wasm_simd128__)
/* Register-resident 4-wide WASM SIMD128 4-byte XXH32 hash + right-shift by w. */
static inline v128_t
noise_hash2_v128(v128_t vk, uint32_t seed, unsigned w)
{
	v128_t vh = wasm_i32x4_splat((int32_t)(seed + 0x165667B5u));
	const v128_t p3 = wasm_i32x4_splat((int32_t)0xC2B2AE3Du);
	const v128_t p4 = wasm_i32x4_splat((int32_t)0x27D4EB2Fu);
	const v128_t p2 = wasm_i32x4_splat((int32_t)0x85EBCA77u);
	vh = wasm_i32x4_add(vh, wasm_i32x4_mul(vk, p3));
	vh = wasm_v128_or(wasm_i32x4_shl(vh, 17), wasm_u32x4_shr(vh, 15));
	vh = wasm_i32x4_mul(vh, p4);
	vh = wasm_v128_xor(vh, wasm_u32x4_shr(vh, 15));
	vh = wasm_i32x4_mul(vh, p2);
	vh = wasm_v128_xor(vh, wasm_u32x4_shr(vh, 13));
	vh = wasm_i32x4_mul(vh, p3);
	vh = wasm_v128_xor(vh, wasm_u32x4_shr(vh, 16));
	if (w >= 32u)
		return wasm_i32x4_splat(0);
	if (w > 0u)
		vh = wasm_u32x4_shr(vh, w);
	return vh;
}
#endif

/* 4-wide 4-byte XXH32 hash + right-shift by w. Bit-exact with noise_r(p, seed, w, 2). */
static inline void
noise_hash2_4x(const uint32_t packed[4], uint32_t seed, unsigned w, uint32_t out[4])
{
#if NOISE_FAST_XXH32_4B && defined(__wasm_simd128__)
	wasm_v128_store(out, noise_hash2_v128(wasm_v128_load(packed), seed, w));
#elif NOISE_FAST_XXH32_4B
	const uint32_t base = seed + 0x165667B5u;
	if (w >= 32u) {
		out[0] = out[1] = out[2] = out[3] = 0u;
		return;
	}
	for (int i = 0; i < 4; i++) {
		uint32_t h = base + packed[i] * 0xC2B2AE3Du;
		h = ((h << 17u) | (h >> 15u)) * 0x27D4EB2Fu;
		h ^= h >> 15u;
		h *= 0x85EBCA77u;
		h ^= h >> 13u;
		h *= 0xC2B2AE3Du;
		h ^= h >> 16u;
		out[i] = h >> w;
	}
#else
	for (int i = 0; i < 4; i++) {
		int16_t k0 = (int16_t)(packed[i] & 0xFFFFu);
		int16_t k1 = (int16_t)(packed[i] >> 16u);
		uint32_t h = noise_hash2_1(k0, k1, seed);
		out[i] = (w >= 32u) ? 0u : (h >> w);
	}
#endif
}

/* Deterministic random value for an N-dimensional point p */
static inline uint32_t
noise_r(int16_t *p, unsigned seed, unsigned w, uint8_t dim)
{
#if NOISE_FAST_XXH32_4B
	if (dim == 2u) {
		uint32_t v = noise_hash2_1(p[0], p[1], seed);
		return (uint32_t)(((long long unsigned) v) >> w);
	}
#endif
	register uint32_t v = TTYPT_NOISE_HASH(p, sizeof(int16_t) * dim, seed);
	return ((long long unsigned) v) >> w;
}

/* Generate values at N-dimensional quad vertices */
static inline void
noise_get_v(uint32_t *v, int16_t *s, uint16_t x, unsigned w, unsigned seed, uint8_t dim)
{
	const int16_t d = (int16_t)(1u << x);
	if (dim == 2u) {
		int16_t s0 = s[0], s1 = s[1];
		int16_t s0d = (int16_t)(s0 + d), s1d = (int16_t)(s1 + d);
#if NOISE_FAST_XXH32_4B && defined(__wasm_simd128__)
		v128_t vk = wasm_i32x4_make(
			(int32_t)noise_pack2(s0,  s1),
			(int32_t)noise_pack2(s0,  s1d),
			(int32_t)noise_pack2(s0d, s1),
			(int32_t)noise_pack2(s0d, s1d));
		wasm_v128_store(v, noise_hash2_v128(vk, seed, w));
#else
		if (w >= 32u) {
			v[0] = v[1] = v[2] = v[3] = 0u;
		} else {
			v[0] = noise_hash2_1(s0,  s1,  seed) >> w;
			v[1] = noise_hash2_1(s0,  s1d, seed) >> w;
			v[2] = noise_hash2_1(s0d, s1,  seed) >> w;
			v[3] = noise_hash2_1(s0d, s1d, seed) >> w;
		}
#endif
		return;
	}
	int16_t va[NOISE_MAX_DIM * (1u << NOISE_MAX_DIM)];
	int i, j;

	for (i = 0; i < (1 << dim); i++) {
		for (j = 0; j < dim; j++)
			va[i * dim + j] = s[j] + ((i & (1 << (dim - j - 1))) ? d : 0);

		v[i] = noise_r(va + dim * i, seed, w, dim);
	}
}

/* Generate values at N-dimensional quad vertices with optional periodic domain wrapping */
static inline void
noise_get_v_periodic(uint32_t *v, const int16_t *s, uint16_t x, unsigned w,
                     unsigned seed, uint8_t dim, uint8_t period_shift)
{
	if (period_shift == 0 || x > period_shift) {
		noise_get_v(v, (int16_t *)s, x, w, seed, dim);
		return;
	}
	const uint16_t period_mask = (period_shift >= 16u) ? 0xFFFFu : (uint16_t)((1u << period_shift) - 1u);
	const uint16_t step_p = (uint16_t)(1u << x);
	int16_t va[dim * (1 << dim)];
	int i, j;

	for (i = 0; i < (1 << dim); i++) {
		for (j = 0; j < dim; j++) {
			uint16_t base = (uint16_t)s[j] & period_mask;
			uint16_t off = (i & (1 << (dim - j - 1))) ? step_p : 0u;
			va[i * dim + j] = (int16_t)((base + off) & period_mask);
		}
		v[i] = noise_r(va + dim * i, seed, w, dim);
	}
}

static inline void
noise_calc_steps(int32_t *st, uint32_t *v, uint16_t z, uint16_t vl, unsigned y)
{
	int n;
	for (n = 0; n < vl; n++)
		st[n] = ((long) v[n + vl] - v[n]) >> z << y;
}

static inline void
noise_step(uint32_t *v, int32_t *st, uint16_t vl, uint16_t mul)
{
	int n;
	for (n = 0; n < vl; n++)
		v[n] = (long) v[n] + ((long) st[n] * mul);
}

/* Fade between vertices and step target values across matrix */
static inline void
noise_quad(uint32_t *c, uint32_t *vc, unsigned z, unsigned cy, uint8_t dim)
{
	uint16_t ndim = dim - 1;
	uint16_t tvl = 1 << ndim;
	int32_t st[(tvl << 1) - 1], *stc = st;
	uint32_t *ce_p[dim], *vt;
	size_t cd = 1 << (cy * ndim);
	goto start;

	do {
		do {
			cd >>= cy; ndim--; vc = vt; stc += tvl; tvl >>= 1;
start:			ce_p[ndim] = c + (cd << z);
			vt = vc + (tvl << 1);

			noise_calc_steps(stc, vc, z, tvl, 0);
			memcpy(vt, vc, tvl * sizeof(uint32_t));
		} while (ndim);

		for (; c < ce_p[0]; c += cd, vt[0] += *stc)
			*c += vt[0];

		do {
			++ndim;
			if (ndim >= dim)
				return;
			c -= cd << z; cd <<= cy; vt = vc;
			tvl <<= 1; stc -= tvl; vc -= (tvl << 1);

			noise_step(vt, stc, tvl, 1);
			c += cd;
		} while (c >= ce_p[ndim]);
	} while (1);
}

static inline void
_noise_mr(uint32_t *c, uint32_t *v, unsigned x, int16_t *qs, uint16_t ndim,
          unsigned w, unsigned seed, unsigned cy, uint8_t dim)
{
	int i = dim - 1 - ndim;
	uint16_t ced = (1 << (cy * (ndim + 1))), cd;
	uint32_t *ce = c + ced;

	ced >>= cy;
	cd = ced << x;

	for (; c < ce; qs[i] += (1 << x), c += cd) {
		if (ndim == 0) {
			noise_get_v(v, qs, x, w, seed, dim);
			noise_quad(c, v, x, cy, dim);
		} else {
			_noise_mr(c, v, x, qs, ndim - 1, w, seed, cy, dim);
		}
	}
	qs[i] -= 1 << cy;
}

static inline void
_noise_m(uint32_t *c, uint32_t *v, unsigned x, int16_t *qs, unsigned w,
         unsigned seed, unsigned cy, uint8_t dim)
{
	_noise_mr(c, v, x, qs, dim - 1, w, seed, cy, dim);
}

static inline void
_noise_fix_v_step(uint32_t *v, int32_t *st, int16_t *ms, int16_t *qs, unsigned x, uint16_t vl, unsigned cy)
{
	register uint32_t *vn = v + vl;
	register uint16_t dd = (*ms - *qs) >> cy;

	noise_calc_steps(st, v, x, vl, cy);
	if (dd)
		noise_step(v, st, vl, dd);

	memcpy(vn, v, sizeof(uint32_t) * vl);
	noise_step(vn, st, vl, 1);
}

static inline void
noise_fix_v(uint32_t *v, int16_t *qs, int16_t *ms, unsigned x, unsigned cy, uint8_t dim)
{
	uint16_t vl = 1 << dim;
	int32_t st[vl - 1], *stc = st;
	int ndim = dim - 1;
	int first = 1;
	vl >>= 1;

	for (;;) {
		_noise_fix_v_step(v, stc, ms, qs, x, vl, cy);
		if (ndim) {
			ndim--; qs++; ms++; vl >>= 1; stc += vl;
		} else if (first) {
			v += vl << 1;
			first = 0;
		} else break;
	}
}

static inline void
noise_get_s(int16_t *s, int16_t *p, uint16_t mask, uint8_t dim)
{
	for (int i = 0; i < dim; i++)
		s[i] = ((int16_t) p[i]) & mask;
}

/* Fills matrix "mat" with deterministic N-dimensional noise */
static inline void
noise_m(uint32_t *mat, int16_t *ms, unsigned x, unsigned w, unsigned seed, unsigned cy, uint8_t dim)
{
	uint32_t v[(1 << (dim + 1)) - 1];

	if (cy > x)
		_noise_m(mat, v, x, ms, w, seed, cy, dim);
	else {
		int16_t qs[dim];
		noise_get_s(qs, ms, COORDMASK_LSHIFT(x), dim);
		noise_get_v(v, qs, x, w, seed, dim);
		if (x > cy)
			noise_fix_v(v, qs, ms, x, cy, dim);
		noise_quad(mat, v, cy, cy, dim);
	}
}

/* Multi-octave N-dimensional matrix noise evaluation */
static inline void
noise_oct(uint32_t *m, int16_t *s, size_t oct_n, octave_t *oct, unsigned seed, unsigned cy, uint8_t dim)
{
	unsigned cm = 1 << (dim * cy);
	memset(m, 0, sizeof(uint32_t) * cm);
	for (size_t i = 0; i < oct_n; i++)
		noise_m(m, s, oct[i].x, oct[i].w, seed, cy, dim);
}

/* --- Generic N-Dimensional Point Sampler --------------------------------- */

/* Evaluates continuous N-dimensional smoothstep value noise at point p[dim]
 * with configurable bit-shift divider `vshift` (1..15; 0 defaults to NOISE_DEFAULT_VSHIFT = 8).
 * Returns fixed-point value in range 0 .. (1 << (16 - vshift)) - 1. Zero heap allocations. */
static inline int
noise_point_n_ex(const int16_t *p, uint32_t seed, uint8_t shift, uint8_t dim, uint8_t vshift)
{
	unsigned vs = (vshift > 0u && vshift < 16u) ? (unsigned)vshift : NOISE_DEFAULT_VSHIFT;
	unsigned bits = 16u - vs;
	uint32_t range = 1u << bits;
	uint32_t mask = range - 1u;
	unsigned hash_rshift = 16u + vs;

	if (!p || dim == 0 || dim > NOISE_MAX_DIM)
		return (int)(range >> 1);

	int cell = 1 << shift;
	int16_t g[NOISE_MAX_DIM];
	int s[NOISE_MAX_DIM];

	for (int i = 0; i < dim; i++) {
		int gi = noise_fdiv(p[i], cell);
		g[i] = (int16_t)gi;
		int fi = ((p[i] - gi * cell) * 256) >> shift;
		s[i] = (fi * fi * (768 - 2 * fi)) >> 16; /* Hermite smoothstep 0..255 */
	}

	/* Fetch random values at the 2^dim hypercube vertices */
	int num_v = 1 << dim;
	int v[1 << NOISE_MAX_DIM];
	int16_t corner[NOISE_MAX_DIM] = { 0, 0, 0, 0 };

	for (int i = 0; i < num_v; i++) {
		for (int j = 0; j < dim; j++)
			corner[j] = g[j] + ((i >> (dim - 1 - j)) & 1);
		v[i] = (int)(noise_r(corner, seed, hash_rshift, dim) & mask);
	}

	/* Multilinear interpolation across dimensions */
	int cur_v = num_v;
	for (int d = 0; d < dim; d++) {
		int next_v = cur_v >> 1;
		int step_weight = s[dim - 1 - d];
		for (int i = 0; i < next_v; i++) {
			int v0 = v[i << 1];
			int v1 = v[(i << 1) + 1];
			v[i] = v0 + (((v1 - v0) * step_weight) >> 8);
		}
		cur_v = next_v;
	}

	return v[0];
}

/* Evaluates continuous N-dimensional smoothstep value noise at point p[dim].
 * Returns fixed-point value in range 0..255. Zero heap allocations. */
static inline int
noise_point_n(const int16_t *p, uint32_t seed, uint8_t shift, uint8_t dim)
{
	return noise_point_n_ex(p, seed, shift, dim, NOISE_DEFAULT_VSHIFT);
}

/* --- Wrapping Original-Algorithm Sampler --------------------------------- */

/* Point queries that follow the ORIGINAL lattice algorithm of noise_m:
 *   - lattice keys are the masked int16 coordinates themselves
 *     (s = p & COORDMASK_LSHIFT(x)), hashed with noise_r;
 *   - the far vertex of a quad is s + 2^x computed in int16_t, so it wraps
 *     exactly like the coordinate does (32767 + 1 == -32768): no seam at
 *     the data-type boundary on any axis;
 *   - interpolation is the matrix stepper's: axis 0 first, step = (v1 - v0)
 *     >> x, value = v0 + f * step (see calc_steps / step / noise_quad).
 * Results are therefore value-equal to noise_m over lattice-aligned
 * matrices. Values are on noise_m's scale: (hash >> w), interpolated.
 * With NOISE_FADE, f is remapped by a Hermite smoothstep onto [0, 2^x)
 * (lattice values are unchanged, derivatives vanish at cell borders).
 * Requires 1 <= dim <= 4 and 1 <= x <= 15. */

#define NOISE_FADE  1u
#define NOISE_FADE5 2u

/* Hermite smoothstep of f in [0, 2^x), mapped onto [0, 2^x) */
static inline uint32_t
noise_wfade(uint32_t f, unsigned x)
{
	return (uint32_t) (((uint64_t) f * f * ((3ull << x) - 2ull * f)) >> (2 * x));
}

/* Exact 512-entry L1 LUT of Perlin quintic fade5 Q16 output for all x <= 9 (f < 2^x).
 * For x <= 9 and f < (1u << x), t = f << (16 - x) == (f << (9 - x)) << 7, so
 * noise_wfade5(f, x) == s_noise_fade5_q16_lut[f << (9u - x)] >> (16u - x) bit-for-bit. */
static const uint16_t s_noise_fade5_q16_lut[512] = {
	    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     9,     9,     9,
	   19,    19,    18,    28,    28,    37,    46,    46,    55,    64,    74,    83,    91,   100,   118,   127,
	  145,   153,   171,   180,   197,   214,   231,   248,   274,   291,   317,   333,   358,   384,   408,   433,
	  466,   491,   524,   548,   580,   612,   644,   685,   716,   756,   795,   835,   874,   912,   959,  1005,
	 1052,  1097,  1143,  1188,  1241,  1293,  1346,  1398,  1457,  1508,  1567,  1625,  1691,  1749,  1814,  1878,
	 1950,  2014,  2085,  2155,  2225,  2294,  2371,  2447,  2522,  2605,  2679,  2761,  2849,  2929,  3016,  3102,
	 3196,  3281,  3373,  3464,  3562,  3659,  3755,  3851,  3953,  4054,  4155,  4262,  4368,  4473,  4578,  4689,
	 4806,  4915,  5030,  5144,  5265,  5377,  5503,  5620,  5744,  5866,  5995,  6122,  6248,  6380,  6511,  6648,
	 6784,  6918,  7051,  7190,  7335,  7471,  7613,  7760,  7907,  8051,  8202,  8351,  8498,  8651,  8809,  8959,
	 9121,  9274,  9433,  9591,  9753,  9920, 10080, 10250, 10413, 10581, 10753, 10924, 11093, 11267, 11439, 11615,
	11796, 11970, 12148, 12330, 12510, 12695, 12878, 13065, 13251, 13435, 13623, 13814, 14004, 14198, 14391, 14581,
	14781, 14973, 15169, 15369, 15567, 15769, 15969, 16172, 16373, 16578, 16786, 16987, 17196, 17404, 17615, 17824,
	18036, 18245, 18458, 18669, 18883, 19095, 19315, 19528, 19749, 19967, 20184, 20403, 20625, 20844, 21067, 21287,
	21515, 21735, 21958, 22184, 22412, 22637, 22865, 23091, 23319, 23549, 23777, 24007, 24240, 24470, 24702, 24931,
	25167, 25396, 25632, 25865, 26100, 26332, 26566, 26802, 27040, 27275, 27512, 27746, 27987, 28220, 28459, 28695,
	28937, 29173, 29414, 29652, 29891, 30128, 30370, 30605, 30846, 31087, 31326, 31566, 31804, 32046, 32286, 32526,
	32768, 33006, 33245, 33486, 33727, 33965, 34205, 34444, 34685, 34923, 35161, 35400, 35640, 35880, 36118, 36355,
	36598, 36833, 37072, 37308, 37545, 37782, 38019, 38257, 38491, 38726, 38964, 39196, 39432, 39667, 39899, 40132,
	40368, 40597, 40830, 41062, 41292, 41521, 41753, 41982, 42212, 42437, 42666, 42894, 43119, 43344, 43572, 43793,
	44020, 44241, 44464, 44684, 44906, 45125, 45347, 45565, 45783, 46000, 46216, 46432, 46647, 46862, 47073, 47287,
	47500, 47708, 47917, 48128, 48335, 48541, 48746, 48950, 49157, 49360, 49562, 49763, 49963, 50162, 50361, 50555,
	50754, 50946, 51141, 51334, 51526, 51717, 51907, 52093, 52281, 52467, 52653, 52836, 53020, 53198, 53382, 53558,
	53739, 53913, 54091, 54262, 54437, 54608, 54778, 54947, 55117, 55282, 55449, 55612, 55776, 55936, 56097, 56254,
	56414, 56568, 56724, 56877, 57031, 57181, 57330, 57479, 57625, 57771, 57916, 58057, 58198, 58338, 58478, 58614,
	58752, 58884, 59018, 59148, 59280, 59406, 59535, 59660, 59786, 59907, 60029, 60149, 60268, 60384, 60501, 60616,
	60729, 60839, 60951, 61058, 61165, 61271, 61376, 61478, 61579, 61678, 61777, 61874, 61970, 62064, 62157, 62247,
	62339, 62426, 62514, 62598, 62683, 62766, 62848, 62927, 63006, 63083, 63159, 63232, 63306, 63377, 63447, 63517,
	63585, 63649, 63715, 63777, 63840, 63900, 63961, 64018, 64075, 64129, 64185, 64237, 64289, 64338, 64388, 64436,
	64484, 64528, 64572, 64614, 64657, 64697, 64736, 64775, 64812, 64847, 64883, 64915, 64949, 64979, 65010, 65040,
	65069, 65094, 65121, 65146, 65171, 65193, 65216, 65236, 65258, 65277, 65296, 65313, 65331, 65346, 65361, 65375,
	65390, 65401, 65414, 65425, 65437, 65445, 65456, 65463, 65472, 65479, 65486, 65491, 65499, 65503, 65508, 65512,
	65516, 65518, 65522, 65524, 65527, 65528, 65530, 65531, 65532, 65533, 65534, 65534, 65534, 65534, 65534, 65534
};

/* Ken Perlin's quintic polynomial: 6t^5 - 15t^4 + 10t^3.
 * First and second derivatives are zero at 0 and 1, ensuring C^2 continuity.
 * Input f in [0, 2^x), returns value in [0, 2^x). */
static inline uint32_t
noise_wfade5(uint32_t f, unsigned x)
{
	if (x <= 9u && f < (1u << x))
		return (uint32_t)s_noise_fade5_q16_lut[f << (9u - x)] >> (16u - x);
	uint64_t t = ((uint64_t) f << 16) >> x;
	if (t > 65536ull) t = 65536ull;
	int64_t inner1 = ((int64_t) t * 6 - 983040ll);
	int64_t inner2 = (((inner1 * (int64_t) t) >> 16) + 655360ll);
	uint64_t t2 = (t * t) >> 16;
	uint64_t t3 = (t2 * t) >> 16;
	uint64_t faded_q16 = ((uint64_t) inner2 * t3) >> 16;
	return (uint32_t) ((faded_q16 << x) >> 16);
}

/* Multiply-then-shift fixed-point linear interpolation step:
 * Evaluates v0 + (((v1 - v0) * f) >> x) without premature division. */
static inline uint32_t
noise_wstep(uint32_t v0, uint32_t v1, uint32_t f, unsigned x)
{
	int64_t diff = (int64_t) v1 - (int64_t) v0;
	return (uint32_t) ((int64_t) v0 + ((diff * (int64_t) f) >> x));
}

/* Legacy slope helper for un-faded matrix steppers */
static inline int32_t
noise_wstep_slope(uint32_t v0, uint32_t v1, unsigned x)
{
	return (int32_t) (((int64_t) v1 - (int64_t) v0) >> x);
}

/* Collapse the first `levels` axes of the 2^dim vertex values in v,
 * axis 0 first, at in-quad offsets f[] (already faded if requested).
 * With NOISE_FADE/NOISE_FADE5, multiplies before shifting to eliminate slope truncation across cells.
 * Without NOISE_FADE, preserves exact bit-level noise_m calc_steps arithmetic. */
static inline void
noise_wreduce(uint32_t *v, const uint32_t *f, unsigned x, unsigned levels, uint8_t dim, unsigned flags)
{
	unsigned half = 1u << dim;
	for (unsigned j = 0; j < levels; j++) {
		half >>= 1;
		for (unsigned n = 0; n < half; n++) {
			if (flags & (NOISE_FADE | NOISE_FADE5)) {
				v[n] = noise_wstep(v[n], v[n + half], f[j], x);
			} else {
				int32_t st = noise_wstep_slope(v[n], v[n + half], x);
				v[n] = (uint32_t) ((int64_t) v[n] + ((int64_t) st * f[j]));
			}
		}
	}
}

/* Lattice cell origin and in-quad offsets of p */
static inline void
noise_wcell(int16_t *s, uint32_t *f, const int16_t *p, unsigned x, unsigned flags, uint8_t dim)
{
	const uint16_t mask = COORDMASK_LSHIFT(x);
	for (unsigned j = 0; j < dim; j++) {
		s[j] = (int16_t) (p[j] & mask);
		f[j] = (uint16_t) p[j] & (uint16_t) ~mask;
		if (flags & NOISE_FADE5)
			f[j] = noise_wfade5(f[j], x);
		else if (flags & NOISE_FADE)
			f[j] = noise_wfade(f[j], x);
	}
}

/* Uncached point query (2^dim hashes) */
static inline uint32_t
noise_wpoint_n(const int16_t *p, unsigned x, unsigned w, unsigned seed, uint8_t dim, unsigned flags)
{
	int16_t s[4];
	uint32_t f[4], v[16];
	noise_wcell(s, f, p, x, flags, dim);
	noise_get_v(v, s, x, w, seed, dim);
	noise_wreduce(v, f, x, dim, dim, flags);
	return v[0];
}

/* Uncached point query with optional periodic domain wrapping */
static inline uint32_t
noise_wpoint_n_periodic(const int16_t *p, unsigned x, unsigned w, unsigned seed,
                        uint8_t dim, unsigned flags, uint8_t period_shift)
{
	int16_t s[4];
	uint32_t f[4], v[16];
	noise_wcell(s, f, p, x, flags, dim);
	noise_get_v_periodic(v, s, (uint16_t)x, w, seed, dim, period_shift);
	noise_wreduce(v, f, x, dim, dim, flags);
	return v[0];
}

static inline uint32_t
noise_wpoint_2d(int16_t x0, int16_t x1, unsigned x, unsigned w, unsigned seed, unsigned flags)
{
	int16_t p[2] = { x0, x1 };
	return noise_wpoint_n(p, x, w, seed, 2, flags);
}

static inline uint32_t
noise_wpoint_3d(int16_t x0, int16_t x1, int16_t x2, unsigned x, unsigned w, unsigned seed, unsigned flags)
{
	int16_t p[3] = { x0, x1, x2 };
	return noise_wpoint_n(p, x, w, seed, 3, flags);
}

/* Corner-reuse cache: remembers the vertex values of the last lattice
 * cell, so slowly-varying (e.g. domain-warped) coordinates hash only when
 * they enter a new cell. `fetches` counts corner-set fetches (each one is
 * 2^dim hashes) for profiling; callers may reset it at will. */
typedef struct {
	int16_t s[4];
	uint32_t v[16];
	unsigned x, w, seed;
	uint8_t dim, valid, period_shift;
	uint32_t fetches;
} noise_wcache_t;

static inline void
noise_wcache_init(noise_wcache_t *c)
{
	memset(c, 0, sizeof(*c));
}

/* Vertex values of the cell at s, from the cache when it matches */
static inline const uint32_t *
noise_wcorners(noise_wcache_t *c, int16_t *s, unsigned x, unsigned w, unsigned seed, uint8_t dim)
{
	int hit = c->valid && c->x == x && c->w == w && c->seed == seed && c->dim == dim && c->period_shift == 0;
	for (unsigned j = 0; hit && j < dim; j++)
		hit = c->s[j] == s[j];
	if (!hit) {
		noise_get_v(c->v, s, x, w, seed, dim);
		memcpy(c->s, s, sizeof(int16_t) * dim);
		c->x = x; c->w = w; c->seed = seed; c->dim = dim; c->period_shift = 0; c->valid = 1;
		c->fetches++;
	}
	return c->v;
}

/* Vertex values of the cell at s with optional periodic domain wrapping */
static inline const uint32_t *
noise_wcorners_periodic(noise_wcache_t *c, const int16_t *s, unsigned x, unsigned w,
                        unsigned seed, uint8_t dim, uint8_t period_shift)
{
	if (period_shift == 0 || x > period_shift)
		return noise_wcorners(c, (int16_t *)s, x, w, seed, dim);
	uint16_t period_mask = (period_shift >= 16u) ? 0xFFFFu : (uint16_t)((1u << period_shift) - 1u);
	int16_t ws[4];
	for (unsigned j = 0; j < dim; j++)
		ws[j] = (int16_t)((uint16_t)s[j] & period_mask);
	int hit = c->valid && c->x == x && c->w == w && c->seed == seed && c->dim == dim && c->period_shift == period_shift;
	for (unsigned j = 0; hit && j < dim; j++)
		hit = c->s[j] == ws[j];
	if (!hit) {
		noise_get_v_periodic(c->v, ws, (uint16_t)x, w, seed, dim, period_shift);
		memcpy(c->s, ws, sizeof(int16_t) * dim);
		c->x = x; c->w = w; c->seed = seed; c->dim = dim; c->period_shift = period_shift; c->valid = 1;
		c->fetches++;
	}
	return c->v;
}

/* Cached point query: value-equal to noise_wpoint_n */
static inline uint32_t
noise_wpoint_c(noise_wcache_t *c, const int16_t *p, unsigned x, unsigned w, unsigned seed, uint8_t dim, unsigned flags)
{
	if (dim == 2u) {
		if (x == 0u) {
			int16_t p0[2] = { p[0], p[1] };
			return noise_r(p0, seed, w, 2);
		}
		const uint16_t mask = COORDMASK_LSHIFT(x);
		int16_t s[2] = { (int16_t)(p[0] & mask), (int16_t)(p[1] & mask) };
		uint32_t f0 = (uint16_t)p[0] & (uint16_t)~mask;
		uint32_t f1 = (uint16_t)p[1] & (uint16_t)~mask;
		const uint32_t *vc = noise_wcorners(c, s, x, w, seed, 2);
		if ((f0 | f1) == 0u)
			return vc[0];
		if (flags & NOISE_FADE5) {
			f0 = noise_wfade5(f0, x);
			f1 = noise_wfade5(f1, x);
			uint32_t b0 = noise_wstep(vc[0], vc[2], f0, x);
			uint32_t b1 = noise_wstep(vc[1], vc[3], f0, x);
			return noise_wstep(b0, b1, f1, x);
		}
		if (flags & NOISE_FADE) {
			f0 = noise_wfade(f0, x);
			f1 = noise_wfade(f1, x);
			uint32_t b0 = noise_wstep(vc[0], vc[2], f0, x);
			uint32_t b1 = noise_wstep(vc[1], vc[3], f0, x);
			return noise_wstep(b0, b1, f1, x);
		}
	}
	int16_t s[4];
	uint32_t f[4], v[16];
	noise_wcell(s, f, p, x, flags, dim);
	memcpy(v, noise_wcorners(c, s, x, w, seed, dim), sizeof(uint32_t) << dim);
	noise_wreduce(v, f, x, dim, dim, flags);
	return v[0];
}

/* Cached point query with optional periodic domain wrapping */
static inline uint32_t
noise_wpoint_c_periodic(noise_wcache_t *c, const int16_t *p, unsigned x, unsigned w,
                        unsigned seed, uint8_t dim, unsigned flags, uint8_t period_shift)
{
	if (period_shift == 0u || x > period_shift)
		return noise_wpoint_c(c, p, x, w, seed, dim, flags);
	int16_t s[4];
	uint32_t f[4], v[16];
	noise_wcell(s, f, p, x, flags, dim);
	memcpy(v, noise_wcorners_periodic(c, s, x, w, seed, dim, period_shift), sizeof(uint32_t) << dim);
	noise_wreduce(v, f, x, dim, dim, flags);
	return v[0];
}

#if defined(__wasm_simd128__)
/* Register-resident 4-wide WASM SIMD128 Perlin quintic fade matching noise_wfade5(f, x) */
static inline v128_t
noise_wfade5_v128(v128_t vf, unsigned x)
{
	v128_t vt;
	const v128_t v65536 = wasm_i32x4_splat(65536);
	if (x <= 16u) {
		uint32_t max_f = 1u << x;
		v128_t vmaxf = wasm_i32x4_splat((int32_t)max_f);
		v128_t vfc = wasm_u32x4_min(vf, vmaxf);
		vt = wasm_i32x4_shl(vfc, 16u - x);
	} else {
		vt = wasm_u32x4_shr(vf, x - 16u);
		vt = wasm_u32x4_min(vt, v65536);
	}
	/* neg_inner1 = 983040 - 6*t in [589824..983040] (strictly positive u32) */
	v128_t v6t = wasm_i32x4_add(wasm_i32x4_shl(vt, 2), wasm_i32x4_shl(vt, 1));
	v128_t vneg1 = wasm_i32x4_sub(wasm_i32x4_splat(983040), v6t);
	const v128_t v65535_64 = wasm_i64x2_splat(65535);

	/* pos = neg_inner1 * t (u64), inner2 = 655360 - ((pos + 65535) >> 16) */
	v128_t p_lo = wasm_u64x2_shr(wasm_i64x2_add(wasm_u64x2_extmul_low_u32x4(vneg1, vt), v65535_64), 16);
	v128_t p_hi = wasm_u64x2_shr(wasm_i64x2_add(wasm_u64x2_extmul_high_u32x4(vneg1, vt), v65535_64), 16);
	v128_t vpos = wasm_i32x4_shuffle(p_lo, p_hi, 0, 2, 4, 6);
	v128_t vinner2 = wasm_i32x4_sub(wasm_i32x4_splat(655360), vpos);

	/* t2 = (t * t) >> 16 */
	v128_t t2_lo = wasm_u64x2_shr(wasm_u64x2_extmul_low_u32x4(vt, vt), 16);
	v128_t t2_hi = wasm_u64x2_shr(wasm_u64x2_extmul_high_u32x4(vt, vt), 16);
	v128_t vt2 = wasm_i32x4_shuffle(t2_lo, t2_hi, 0, 2, 4, 6);

	/* t3 = (t2 * t) >> 16 */
	v128_t t3_lo = wasm_u64x2_shr(wasm_u64x2_extmul_low_u32x4(vt2, vt), 16);
	v128_t t3_hi = wasm_u64x2_shr(wasm_u64x2_extmul_high_u32x4(vt2, vt), 16);
	v128_t vt3 = wasm_i32x4_shuffle(t3_lo, t3_hi, 0, 2, 4, 6);

	/* faded_q16 = (inner2 * t3) >> 16 */
	v128_t fq_lo = wasm_u64x2_shr(wasm_u64x2_extmul_low_u32x4(vinner2, vt3), 16);
	v128_t fq_hi = wasm_u64x2_shr(wasm_u64x2_extmul_high_u32x4(vinner2, vt3), 16);
	v128_t vfq = wasm_i32x4_shuffle(fq_lo, fq_hi, 0, 2, 4, 6);

	return (x <= 16u) ? wasm_u32x4_shr(vfq, 16u - x) : wasm_i32x4_shl(vfq, x - 16u);
}

/* Register-resident 4-wide WASM SIMD128 linear interpolation matching noise_wstep(v0, v1, f, x) */
static inline v128_t
noise_wstep_v128(v128_t vv0, v128_t vv1, v128_t vf, unsigned x)
{
	v128_t neg = wasm_u32x4_lt(vv1, vv0);
	v128_t d   = wasm_v128_bitselect(wasm_i32x4_sub(vv0, vv1), wasm_i32x4_sub(vv1, vv0), neg);
	uint64_t rmask = (x < 64u) ? ((1ull << x) - 1ull) : ~0ull;
	v128_t vrm = wasm_i64x2_splat((int64_t)rmask);
	v128_t neg_lo = wasm_i64x2_extend_low_i32x4(neg);
	v128_t neg_hi = wasm_i64x2_extend_high_i32x4(neg);
	v128_t p_lo = wasm_u64x2_extmul_low_u32x4(d, vf);
	v128_t p_hi = wasm_u64x2_extmul_high_u32x4(d, vf);
	p_lo = wasm_u64x2_shr(wasm_i64x2_add(p_lo, wasm_v128_and(neg_lo, vrm)), x);
	p_hi = wasm_u64x2_shr(wasm_i64x2_add(p_hi, wasm_v128_and(neg_hi, vrm)), x);
	v128_t mag = wasm_i32x4_shuffle(p_lo, p_hi, 0, 2, 4, 6);
	return wasm_v128_bitselect(wasm_i32x4_sub(vv0, mag), wasm_i32x4_add(vv0, mag), neg);
}
/* Fast 4-wide WASM SIMD128 linear interpolation using single 32-bit i32x4.mul.
 * Exact when vv0, vv1 <= 0x00FFFFFF (i.e. w >= 8) and x <= 8 (vf < 2^x <= 255):
 * max(d * f + rmask) = 16777215 * 255 + 255 = 4278190080 <= UINT32_MAX. */
static inline v128_t
noise_wstep_v128_u24(v128_t vv0, v128_t vv1, v128_t vf, unsigned x)
{
	v128_t neg = wasm_u32x4_lt(vv1, vv0);
	v128_t d   = wasm_v128_bitselect(wasm_i32x4_sub(vv0, vv1), wasm_i32x4_sub(vv1, vv0), neg);
	v128_t p   = wasm_i32x4_mul(d, vf);
	v128_t vrm = wasm_i32x4_splat((int32_t)((1u << x) - 1u));
	v128_t mag = wasm_u32x4_shr(wasm_i32x4_add(p, wasm_v128_and(neg, vrm)), x);
	return wasm_v128_bitselect(wasm_i32x4_sub(vv0, mag), wasm_i32x4_add(vv0, mag), neg);
}
#endif

/* 4-wide bit-exact Perlin quintic fade matching noise_wfade5(f[i], x) */
static inline void
noise_wfade5_4x(const uint32_t f[4], unsigned x, uint32_t out[4])
{
	if (x <= 9u) {
		const uint32_t lim = 1u << x;
		const unsigned lsh = 9u - x;
		const unsigned rsh = 16u - x;
		if ((f[0] | f[1] | f[2] | f[3]) < lim) {
			out[0] = (uint32_t)s_noise_fade5_q16_lut[f[0] << lsh] >> rsh;
			out[1] = (uint32_t)s_noise_fade5_q16_lut[f[1] << lsh] >> rsh;
			out[2] = (uint32_t)s_noise_fade5_q16_lut[f[2] << lsh] >> rsh;
			out[3] = (uint32_t)s_noise_fade5_q16_lut[f[3] << lsh] >> rsh;
			return;
		}
	}
	for (int i = 0; i < 4; i++)
		out[i] = noise_wfade5(f[i], x);
}

/* 4-wide bit-exact fixed-point linear interpolation matching noise_wstep(v0[i], v1[i], f[i], x) */
static inline void
noise_wstep_4x(const uint32_t v0[4], const uint32_t v1[4], const uint32_t f[4], unsigned x, uint32_t out[4])
{
#if defined(__wasm_simd128__)
	if (x <= 8u && ((v0[0] | v0[1] | v0[2] | v0[3] | v1[0] | v1[1] | v1[2] | v1[3]) <= 0x00FFFFFFu) &&
	    ((f[0] | f[1] | f[2] | f[3]) < (1u << x))) {
		wasm_v128_store(out, noise_wstep_v128_u24(wasm_v128_load(v0), wasm_v128_load(v1), wasm_v128_load(f), x));
		return;
	}
#endif
	for (int i = 0; i < 4; i++)
		out[i] = noise_wstep(v0[i], v1[i], f[i], x);
}

/* 4-wide 2D wrapped point query: bit-exact with 4 calls to
 * noise_wpoint_c(c, {py[i], px[i]}, x, w, seed, 2, flags). */
static inline void
noise_wpoint_2d_4x(noise_wcache_t *c, const int16_t py[4], const int16_t px[4],
                   unsigned x, unsigned w, unsigned seed, unsigned flags, uint32_t out[4])
{
	if (x == 0u) {
#if NOISE_FAST_XXH32_4B && defined(__wasm_simd128__)
		v128_t vk = wasm_i32x4_make(
			(int32_t)noise_pack2(py[0], px[0]),
			(int32_t)noise_pack2(py[1], px[1]),
			(int32_t)noise_pack2(py[2], px[2]),
			(int32_t)noise_pack2(py[3], px[3]));
		wasm_v128_store(out, noise_hash2_v128(vk, seed, w));
#else
		for (int i = 0; i < 4; i++)
			out[i] = (w >= 32u) ? 0u : (noise_hash2_1(py[i], px[i], seed) >> w);
#endif
		return;
	}
	if ((flags & (NOISE_FADE | NOISE_FADE5)) != NOISE_FADE5) {
		for (int i = 0; i < 4; i++) {
			int16_t p[2] = { py[i], px[i] };
			out[i] = noise_wpoint_c(c, p, x, w, seed, 2, flags);
		}
		return;
	}

	const uint16_t mask = COORDMASK_LSHIFT(x);
	const uint16_t lo_mask = (uint16_t)~mask;
	int16_t sy[4], sx[4];
	uint32_t f0[4], f1[4];
	for (int i = 0; i < 4; i++) {
		sy[i] = (int16_t)(py[i] & mask);
		sx[i] = (int16_t)(px[i] & mask);
		f0[i] = (uint16_t)py[i] & lo_mask;
		f1[i] = (uint16_t)px[i] & lo_mask;
	}

#if defined(__wasm_simd128__)
	if (x <= 8u && w >= 8u) {
		v128_t vv00, vv01, vv10, vv11;
		if (sy[0] == sy[1] && sy[0] == sy[2] && sy[0] == sy[3] &&
		    sx[0] == sx[1] && sx[0] == sx[2] && sx[0] == sx[3]) {
			int16_t s[2] = { sy[0], sx[0] };
			const uint32_t *vc = noise_wcorners(c, s, x, w, seed, 2);
			vv00 = wasm_i32x4_splat((int32_t)vc[0]);
			vv01 = wasm_i32x4_splat((int32_t)vc[1]);
			vv10 = wasm_i32x4_splat((int32_t)vc[2]);
			vv11 = wasm_i32x4_splat((int32_t)vc[3]);
		} else {
			uint32_t v00[4], v01[4], v10[4], v11[4];
			for (int i = 0; i < 4; i++) {
				if (i > 0 && sy[i] == sy[i - 1] && sx[i] == sx[i - 1]) {
					v00[i] = v00[i - 1]; v01[i] = v01[i - 1];
					v10[i] = v10[i - 1]; v11[i] = v11[i - 1];
				} else {
					int16_t s[2] = { sy[i], sx[i] };
					const uint32_t *vc = noise_wcorners(c, s, x, w, seed, 2);
					v00[i] = vc[0]; v01[i] = vc[1]; v10[i] = vc[2]; v11[i] = vc[3];
				}
			}
			vv00 = wasm_v128_load(v00);
			vv01 = wasm_v128_load(v01);
			vv10 = wasm_v128_load(v10);
			vv11 = wasm_v128_load(v11);
		}
		const unsigned lsh = 9u - x;
		const unsigned rsh = 16u - x;
		v128_t ff0 = wasm_u32x4_shr(wasm_i32x4_make(
			(int32_t)s_noise_fade5_q16_lut[f0[0] << lsh],
			(int32_t)s_noise_fade5_q16_lut[f0[1] << lsh],
			(int32_t)s_noise_fade5_q16_lut[f0[2] << lsh],
			(int32_t)s_noise_fade5_q16_lut[f0[3] << lsh]), rsh);
		v128_t ff1 = wasm_u32x4_shr(wasm_i32x4_make(
			(int32_t)s_noise_fade5_q16_lut[f1[0] << lsh],
			(int32_t)s_noise_fade5_q16_lut[f1[1] << lsh],
			(int32_t)s_noise_fade5_q16_lut[f1[2] << lsh],
			(int32_t)s_noise_fade5_q16_lut[f1[3] << lsh]), rsh);
		v128_t b0  = noise_wstep_v128_u24(vv00, vv10, ff0, x);
		v128_t b1  = noise_wstep_v128_u24(vv01, vv11, ff0, x);
		wasm_v128_store(out, noise_wstep_v128_u24(b0, b1, ff1, x));
		return;
	}
#endif
	if (sy[0] == sy[1] && sy[0] == sy[2] && sy[0] == sy[3] &&
	    sx[0] == sx[1] && sx[0] == sx[2] && sx[0] == sx[3]) {
		int16_t s[2] = { sy[0], sx[0] };
		const uint32_t *vc = noise_wcorners(c, s, x, w, seed, 2);
		uint32_t c00 = vc[0], c01 = vc[1], c10 = vc[2], c11 = vc[3];
		for (int i = 0; i < 4; i++) {
			uint32_t ff0 = noise_wfade5(f0[i], x);
			uint32_t ff1 = noise_wfade5(f1[i], x);
			uint32_t b0 = noise_wstep(c00, c10, ff0, x);
			uint32_t b1 = noise_wstep(c01, c11, ff0, x);
			out[i] = noise_wstep(b0, b1, ff1, x);
		}
	} else {
		uint32_t c00 = 0, c01 = 0, c10 = 0, c11 = 0;
		for (int i = 0; i < 4; i++) {
			if (i == 0 || sy[i] != sy[i - 1] || sx[i] != sx[i - 1]) {
				int16_t s[2] = { sy[i], sx[i] };
				const uint32_t *vc = noise_wcorners(c, s, x, w, seed, 2);
				c00 = vc[0]; c01 = vc[1]; c10 = vc[2]; c11 = vc[3];
			}
			uint32_t ff0 = noise_wfade5(f0[i], x);
			uint32_t ff1 = noise_wfade5(f1[i], x);
			uint32_t b0 = noise_wstep(c00, c10, ff0, x);
			uint32_t b1 = noise_wstep(c01, c11, ff0, x);
			out[i] = noise_wstep(b0, b1, ff1, x);
		}
	}
}

/* Row stepper: out[k] = noise_wpoint_n(p0 + k * dstep along the LAST axis),
 * k < n, the coordinate wrapping in int16_t. Like noise_get_v + noise_quad,
 * it fetches the 2^dim corners once per lattice cell and collapses the
 * other axes once per cell; each sample is then one multiply-add on the
 * innermost axis. `c` may be NULL. */
static inline void
noise_wrow(uint32_t *out, unsigned n, const int16_t *p0, int16_t dstep, unsigned x,
           unsigned w, unsigned seed, uint8_t dim, unsigned flags, noise_wcache_t *c)
{
	const unsigned L = dim - 1u;
	const uint16_t mask = COORDMASK_LSHIFT(x), lo = (uint16_t) ~mask;
	noise_wcache_t local;
	int16_t p[4], s[4];
	uint32_t f[4], v[16], b0 = 0;
	int32_t st = 0;
	int have = 0;

	if (!c) {
		noise_wcache_init(&local);
		c = &local;
	}
	memcpy(p, p0, sizeof(int16_t) * dim);

	for (unsigned k = 0; k < n; k++, p[L] = (int16_t) (p[L] + dstep)) {
		int16_t sl = (int16_t) (p[L] & mask);
		uint32_t fl = (uint16_t) p[L] & lo;

		if (!have || sl != s[L]) {
			noise_wcell(s, f, p, x, flags, dim);
			memcpy(v, noise_wcorners(c, s, x, w, seed, dim), sizeof(uint32_t) << dim);
			noise_wreduce(v, f, x, L, dim, flags);
			b0 = v[0];
			st = (flags & (NOISE_FADE | NOISE_FADE5)) ? 0 : noise_wstep_slope(v[0], v[1], x);
			have = 1;
		}
		if (flags & NOISE_FADE5) {
			fl = noise_wfade5(fl, x);
			out[k] = noise_wstep(v[0], v[1], fl, x);
		} else if (flags & NOISE_FADE) {
			fl = noise_wfade(fl, x);
			out[k] = noise_wstep(v[0], v[1], fl, x);
		} else {
			out[k] = b0 + (uint32_t) ((int64_t) st * fl);
		}
	}
}

/* Plane stepper: out[r * stride + k] = noise_wpoint_n(p0 with the last two
 * axes advanced by r (rows, axis dim-2) and k (columns, axis dim-1)), for
 * r < rows, k < n; both coordinates wrap in int16_t. In noise_quad order it
 * fetches the 2^dim corners ONCE per 2D lattice cell (rows x cols / 4^x
 * fetches), collapses the fixed outer axes once per cell, the row axis once
 * per row, and spends one multiply-add per sample. Requires dim >= 2.
 * `c` may be NULL. */
static inline void
noise_wplane(uint32_t *out, unsigned n, unsigned rows, size_t stride, const int16_t *p0,
             unsigned x, unsigned w, unsigned seed, uint8_t dim, unsigned flags, noise_wcache_t *c)
{
	const unsigned R = dim - 2u, C = dim - 1u, side = 1u << x;
	const uint16_t lo = (uint16_t) (side - 1u);
	noise_wcache_t local;
	int16_t p[4], s[4];
	uint32_t f[4], v[16];

	if (!c) {
		noise_wcache_init(&local);
		c = &local;
	}
	memcpy(p, p0, sizeof(int16_t) * dim);

	for (unsigned r = 0; r < rows; ) {
		int16_t pr = (int16_t) (p0[R] + (int) r);
		unsigned rlen = side - ((uint16_t) pr & lo);
		if (rlen > rows - r)
			rlen = rows - r;
		p[R] = pr;

		for (unsigned k = 0; k < n; ) {
			int16_t pc = (int16_t) (p0[C] + (int) k);
			unsigned klen = side - ((uint16_t) pc & lo);
			if (klen > n - k)
				klen = n - k;
			p[C] = pc;

			/* one corner fetch per 2D cell; collapse outer axes */
			noise_wcell(s, f, p, x, flags, dim);
			memcpy(v, noise_wcorners(c, s, x, w, seed, dim), sizeof(uint32_t) << dim);
			noise_wreduce(v, f, x, R, dim, flags);
			{
				const uint32_t a00 = v[0], a01 = v[1], a10 = v[2], a11 = v[3];
				const int64_t d0 = (int64_t) a10 - (int64_t) a00;
				const int64_t d1 = (int64_t) a11 - (int64_t) a01;
				const int32_t s0 = (int32_t) (d0 >> x), s1 = (int32_t) (d1 >> x);
				uint32_t fr0 = (uint16_t) pr & lo, fc0 = (uint16_t) pc & lo;

				for (unsigned i = 0; i < rlen; i++) {
					uint32_t fr = fr0 + i, *o = out + (size_t) (r + i) * stride + k;
					uint32_t b0, b1;
					if (flags & NOISE_FADE5) {
						fr = noise_wfade5(fr, x);
						b0 = noise_wstep(a00, a10, fr, x);
						b1 = noise_wstep(a01, a11, fr, x);
					} else if (flags & NOISE_FADE) {
						fr = noise_wfade(fr, x);
						b0 = noise_wstep(a00, a10, fr, x);
						b1 = noise_wstep(a01, a11, fr, x);
					} else {
						b0 = a00 + (uint32_t) ((int64_t) s0 * fr);
						b1 = a01 + (uint32_t) ((int64_t) s1 * fr);
					}
					const int64_t dst = (int64_t) b1 - (int64_t) b0;
					const int32_t st = (int32_t) (dst >> x);
					for (unsigned j = 0; j < klen; j++) {
						uint32_t fc = fc0 + j;
						if (flags & NOISE_FADE5) {
							fc = noise_wfade5(fc, x);
							o[j] = noise_wstep(b0, b1, fc, x);
						} else if (flags & NOISE_FADE) {
							fc = noise_wfade(fc, x);
							o[j] = noise_wstep(b0, b1, fc, x);
						} else {
							o[j] = b0 + (uint32_t) ((int64_t) st * fc);
						}
					}
				}
			}
			k += klen;
		}
		r += rlen;
	}
}

/* Point-evaluation ergonomic helpers */
static inline int
noise_point_1d(int16_t x, uint32_t seed, uint8_t shift)
{
	int16_t p[1] = { x };
	return noise_point_n(p, seed, shift, 1);
}

static inline int
noise_point_2d(int16_t x, int16_t y, uint32_t seed, uint8_t shift)
{
	int16_t p[2] = { x, y };
	return noise_point_n(p, seed, shift, 2);
}

static inline int
noise_point_3d(int16_t x, int16_t y, int16_t z, uint32_t seed, uint8_t shift)
{
	int16_t p[3] = { x, y, z };
	return noise_point_n(p, seed, shift, 3);
}

static inline int
noise_point_4d(int16_t x, int16_t y, int16_t z, int16_t w, uint32_t seed, uint8_t shift)
{
	int16_t p[4] = { x, y, z, w };
	return noise_point_n(p, seed, shift, 4);
}

/* --- Canonical libchaff aliases (chaff_ prefix) --------------------------- */
#define CHAFF_MAX NOISE_MAX
#define CHAFF_FADE NOISE_FADE

typedef octave_t         chaff_oct_t;
typedef noise_wcache_t   chaff_wcache_t;

static inline int chaff_point_1d(int16_t x, uint32_t seed, uint8_t shift) {
	return noise_point_1d(x, seed, shift);
}
static inline int chaff_point_2d(int16_t x, int16_t y, uint32_t seed, uint8_t shift) {
	return noise_point_2d(x, y, seed, shift);
}
static inline int chaff_point_3d(int16_t x, int16_t y, int16_t z, uint32_t seed, uint8_t shift) {
	return noise_point_3d(x, y, z, seed, shift);
}
static inline int chaff_point_4d(int16_t x, int16_t y, int16_t z, int16_t w, uint32_t seed, uint8_t shift) {
	return noise_point_4d(x, y, z, w, seed, shift);
}
static inline int chaff_point_n(const int16_t *p, uint32_t seed, uint8_t shift, uint8_t dim) {
	return noise_point_n(p, seed, shift, dim);
}
static inline int chaff_point_n_ex(const int16_t *p, uint32_t seed, uint8_t shift, uint8_t dim, uint8_t vshift) {
	return noise_point_n_ex(p, seed, shift, dim, vshift);
}
static inline void chaff_oct(uint32_t *m, int16_t *s, size_t oct_n, octave_t *oct, unsigned seed, unsigned cy, uint8_t dim) {
	noise_oct(m, s, oct_n, oct, seed, cy, dim);
}
static inline uint32_t chaff_wpoint_n(const int16_t *p, unsigned x, unsigned w, unsigned seed, uint8_t dim, unsigned flags) {
	return noise_wpoint_n(p, x, w, seed, dim, flags);
}
static inline uint32_t chaff_wpoint_c(noise_wcache_t *c, const int16_t *p, unsigned x, unsigned w, unsigned seed, uint8_t dim, unsigned flags) {
	return noise_wpoint_c(c, p, x, w, seed, dim, flags);
}
static inline void chaff_wrow(uint32_t *out, unsigned n, const int16_t *p0, int16_t dstep, unsigned x, unsigned w, unsigned seed, uint8_t dim, unsigned flags, noise_wcache_t *c) {
	noise_wrow(out, n, p0, dstep, x, w, seed, dim, flags, c);
}
static inline void chaff_wplane(uint32_t *out, unsigned n, unsigned rows, size_t stride, const int16_t *p0, unsigned x, unsigned w, unsigned seed, uint8_t dim, unsigned flags, noise_wcache_t *c) {
	noise_wplane(out, n, rows, stride, p0, x, w, seed, dim, flags, c);
}

/* --- Unified Chaff Fixed-Point API & Declarative Cascades -------------- */

/* Fixed-point coordinate: cell (lattice/room units) + frac (1/65536 of a unit) */
typedef struct {
	int16_t  cell;
	uint16_t frac;
} chaff_coord_t;

#define CHAFF_MODE_SET 0u
#define CHAFF_MODE_ADD 1u

#define CHAFF_SEED_SHARED    0u    /* One seed for all layers */
#define CHAFF_SEED_PER_LAYER 1u    /* Seed derived per layer: seed ^ (l * CHAFF_SEED_PRIME + 1) */
#define CHAFF_LAYER_LINEAR   0u    /* Standard linear signed layer amplitude */
#define CHAFF_LAYER_RIDGED   0x80u /* Ridged multifractal folding bitflag on seed_mode */

#define CHAFF_FLAG_FADE   1u
#define CHAFF_FLAG_FADE5  2u
#define CHAFF_FLAG_LINEAR 0x80u

/** Identity contrast multiplier in Q8.8 fixed-point (256 = 1.0x). */
#define NOISE_CONTRAST_IDENTITY 256u

/** Per-layer seed decorrelation multiplier (Murmur3 finalizer constant). */
#define CHAFF_SEED_PRIME 0x85EBCA77u

/* Universal layer descriptor */
typedef struct {
	uint8_t  shift;        /* Lattice size 2^shift (e.g. 7 = 128, 2 = 4) */
	uint8_t  vshift;       /* Value right shift: hash >> vshift (default NOISE_DEFAULT_VSHIFT) */
	int16_t  amp;          /* Signed amplitude multiplier (0 = raw unsigned hash >> vshift) */
	uint8_t  seed_mode;    /* CHAFF_SEED_SHARED or CHAFF_SEED_PER_LAYER (| CHAFF_LAYER_RIDGED) */
	int8_t   skew[2];      /* Affine domain skew: skew[0]=x, skew[1]=y */
	uint16_t contrast;     /* Contrast multiplier: 256 = 1.0x (0 or 256 = identity) */
	uint8_t  period_shift; /* Optional periodic domain shift 2^period_shift (0 = unbounded) */
	uint32_t seed_salt;    /* Optional per-layer seed salt override (0 = use field->seed_salt or 1) */
} chaff_layer_t;

/* Universal field descriptor */
typedef struct {
	uint32_t seed;
	uint8_t  dim;
	uint8_t  flags;
	uint8_t  nlayers;
	const chaff_layer_t *layers;
	uint32_t seed_prime;   /* Optional per-layer seed prime (0 = CHAFF_SEED_PRIME) */
	uint32_t seed_salt;    /* Optional per-layer seed salt offset (0 = 1u) */
	int16_t  rot_cos_q15;  /* Optional inter-octave rotation cos in Q15 (0 = default in rugged_2d) */
	int16_t  rot_sin_q15;  /* Optional inter-octave rotation sin in Q15 (0 = default in rugged_2d) */
} chaff_field_t;

static inline uint32_t
chaff_layer_seed(const chaff_field_t *field, const chaff_layer_t *lay, uint8_t l)
{
	if (!field || !lay) return 0;
	if ((lay->seed_mode & 0x01u) == CHAFF_SEED_SHARED)
		return field->seed ^ lay->seed_salt;
	uint32_t prime = field->seed_prime ? field->seed_prime : CHAFF_SEED_PRIME;
	uint32_t salt  = lay->seed_salt ? lay->seed_salt : (field->seed_salt ? field->seed_salt : 1u);
	return field->seed ^ (uint32_t)(l * prime + salt);
}

/* Fractional fade helper with selectable C1 cubic Hermite or C2 quintic Hermite in 16.16 */
static inline uint32_t
chaff_fade_frac_ex(uint32_t fi, uint16_t frac, unsigned x, uint8_t flags)
{
	if (flags & CHAFF_FLAG_FADE5) {
		if (frac == 0)
			return noise_wfade5(fi, x) << 16;
		uint64_t t = (((uint64_t)fi << 16) | frac) >> x;
		if (t > 65536ull) t = 65536ull;
		uint64_t p1 = ((t * 6ull) >> 16);
		int64_t  p2 = (int64_t)p1 - (15ll << 16);
		int64_t  p3 = (((int64_t)t * p2) >> 16) + (10ll << 16);
		uint64_t t2 = (t * t) >> 16;
		uint64_t t3 = (t2 * t) >> 16;
		uint64_t faded_t = (t3 * (uint64_t)p3) >> 16;
		if (faded_t > 65536ull) faded_t = 65536ull;
		return (uint32_t)(faded_t << x);
	}
	if (frac == 0)
		return noise_wfade(fi, x) << 16;
	uint64_t t = (((uint64_t)fi << 16) | frac) >> x;
	if (t > 65536ull) t = 65536ull;
	uint64_t faded_t = (t * t * (3ull * 65536ull - 2ull * t)) >> 32;
	return (uint32_t)(faded_t << x);
}

static inline uint32_t
chaff_fade_frac(uint32_t fi, uint16_t frac, unsigned x)
{
	return chaff_fade_frac_ex(fi, frac, x, CHAFF_FLAG_FADE);
}

/* Primary multi-layer region generator (point, row, plane, volume).
 * Evaluates field layers [layer_lo, layer_hi) over a grid of size n[dim]
 * starting at origin[dim] with step_q16[dim] (rational 16.16 fixed-point steps).
 * Zero heap allocations; callers supply `out`. */
static inline void
chaff_fill(const chaff_field_t *field,
           uint8_t layer_lo, uint8_t layer_hi,
           const chaff_coord_t origin[],
           const int32_t step_q16[],
           const uint16_t n[],
           int32_t *out, size_t stride, unsigned mode)
{
	if (!field || !out || field->dim == 0 || field->dim > 4) return;
	if (layer_hi > field->nlayers) layer_hi = field->nlayers;

	size_t hypers = (field->dim == 4) ? n[0] : 1;
	size_t slices = (field->dim >= 3) ? n[field->dim - 3] : 1;
	size_t rows   = (field->dim >= 2) ? n[field->dim - 2] : 1;
	size_t cols   = n[field->dim - 1];
	if (stride < cols) stride = cols;
	size_t slice_stride = rows * stride;
	size_t hyper_stride = slices * slice_stride;

	if (mode == CHAFF_MODE_SET) {
		for (size_t h = 0; h < hypers; h++) {
			for (size_t s = 0; s < slices; s++) {
				for (size_t r = 0; r < rows; r++) {
					memset(out + h * hyper_stride + s * slice_stride + r * stride, 0, cols * sizeof(int32_t));
				}
			}
		}
	}
	if (layer_lo >= layer_hi) return;

	noise_wcache_t cache;
	int use_fade = (field->flags & (CHAFF_FLAG_FADE | CHAFF_FLAG_FADE5)) != 0 &&
	               (field->flags & CHAFF_FLAG_LINEAR) == 0;

	for (uint8_t l = layer_lo; l < layer_hi; l++) {
		const chaff_layer_t *lay = &field->layers[l];
		uint32_t seed = chaff_layer_seed(field, lay, l);
		unsigned x = lay->shift;
		unsigned vshift = (lay->vshift && lay->vshift < 16u) ? lay->vshift : NOISE_DEFAULT_VSHIFT;
		int32_t mid = (int32_t)(1u << (15u - vshift));
		int32_t ridge_fold = mid >> 1;
		unsigned amp_shift = 15u - vshift;
		uint16_t mask = COORDMASK_LSHIFT(x);
		uint8_t pshift = lay->period_shift;
		noise_wcache_init(&cache);

		int32_t base_w_q16 = (field->dim == 4) ?
		                     (((int32_t)origin[0].cell << 16) | (uint32_t)origin[0].frac) : 0;
		int32_t dw_q16     = (field->dim == 4) ? step_q16[0] : 0;

		int32_t base_z_q16 = (field->dim >= 3) ?
		                     (((int32_t)origin[field->dim - 3].cell << 16) | (uint32_t)origin[field->dim - 3].frac) : 0;
		int32_t dz_q16     = (field->dim >= 3) ? step_q16[field->dim - 3] : 0;

		int32_t base_y_q16 = (field->dim >= 2) ?
		                     (((int32_t)origin[field->dim - 2].cell << 16) | (uint32_t)origin[field->dim - 2].frac) : 0;
		int32_t dy_q16     = (field->dim >= 2) ? step_q16[field->dim - 2] : 0;

		int32_t base_x_q16 = (((int32_t)origin[field->dim - 1].cell << 16) | (uint32_t)origin[field->dim - 1].frac);
		int32_t dx_q16     = step_q16[field->dim - 1];

		for (size_t h = 0; h < hypers; h++) {
			int32_t raw_w_q16 = base_w_q16 + (int32_t)h * dw_q16;
			int32_t *hyper_out = out + h * hyper_stride;

			for (size_t s = 0; s < slices; s++) {
				int32_t raw_z_q16 = base_z_q16 + (int32_t)s * dz_q16;
				int32_t *slice_out = hyper_out + s * slice_stride;

				for (size_t r = 0; r < rows; r++) {
					int32_t raw_y_q16 = base_y_q16 + (int32_t)r * dy_q16;
					int32_t *row_out = slice_out + r * stride;

					for (size_t c = 0; c < cols; c++) {
						int32_t raw_x_q16 = base_x_q16 + (int32_t)c * dx_q16;
						int16_t cell_x, cell_y, cell_z = 0, cell_w = 0;
						uint16_t frac_x = 0, frac_y = 0, frac_z = 0, frac_w = 0;

						if ((raw_x_q16 & 0xffff) == 0 && (raw_y_q16 & 0xffff) == 0 &&
						    (raw_z_q16 & 0xffff) == 0 && (raw_w_q16 & 0xffff) == 0) {
							cell_x = (int16_t)(raw_x_q16 >> 16);
							cell_y = (int16_t)(raw_y_q16 >> 16);
							cell_z = (int16_t)(raw_z_q16 >> 16);
							cell_w = (int16_t)(raw_w_q16 >> 16);
							if (lay->skew[0]) cell_x = (int16_t)(cell_x + cell_y / lay->skew[0]);
							if (lay->skew[1]) cell_y = (int16_t)(cell_y - cell_x / lay->skew[1]);
						} else {
							int32_t qx = raw_x_q16, qy = raw_y_q16;
							if (lay->skew[0]) qx += qy / lay->skew[0];
							if (lay->skew[1]) qy -= qx / lay->skew[1];
							cell_x = (int16_t)(qx >> 16);
							frac_x = (uint16_t)(qx & 0xffff);
							cell_y = (int16_t)(qy >> 16);
							frac_y = (uint16_t)(qy & 0xffff);
							cell_z = (int16_t)(raw_z_q16 >> 16);
							frac_z = (uint16_t)(raw_z_q16 & 0xffff);
							cell_w = (int16_t)(raw_w_q16 >> 16);
							frac_w = (uint16_t)(raw_w_q16 & 0xffff);
						}

						uint32_t raw = 0;
						if (field->dim == 1) {
							int16_t pt_s[1] = { (int16_t)(cell_x & mask) };
							uint32_t fix = (uint16_t)cell_x & (uint16_t)~mask;
							uint32_t ux = use_fade ?
							              chaff_fade_frac_ex(fix, frac_x, x, field->flags) :
							              (((uint32_t)fix << 16) | frac_x);
							const uint32_t *v = noise_wcorners_periodic(&cache, pt_s, x, vshift, seed, 1, pshift);
							int32_t st = noise_wstep_slope(v[0], v[1], x);
							raw = v[0] + (uint32_t)(((int64_t)st * ux) >> 16);
						} else if (field->dim == 2) {
							int16_t pt_s[2] = { (int16_t)(cell_y & mask), (int16_t)(cell_x & mask) };
							uint32_t fiy = (uint16_t)cell_y & (uint16_t)~mask;
							uint32_t fix = (uint16_t)cell_x & (uint16_t)~mask;
							uint32_t uy = use_fade ?
							              chaff_fade_frac_ex(fiy, frac_y, x, field->flags) :
							              (((uint32_t)fiy << 16) | frac_y);
							uint32_t ux = use_fade ?
							              chaff_fade_frac_ex(fix, frac_x, x, field->flags) :
							              (((uint32_t)fix << 16) | frac_x);
							const uint32_t *v = noise_wcorners_periodic(&cache, pt_s, x, vshift, seed, 2, pshift);
							int32_t s0 = noise_wstep_slope(v[0], v[2], x);
							int32_t s1 = noise_wstep_slope(v[1], v[3], x);
							uint32_t b0 = v[0] + (uint32_t)(((int64_t)s0 * uy) >> 16);
							uint32_t b1 = v[1] + (uint32_t)(((int64_t)s1 * uy) >> 16);
							int32_t st = noise_wstep_slope(b0, b1, x);
							raw = b0 + (uint32_t)(((int64_t)st * ux) >> 16);
						} else if (field->dim == 3) {
							int16_t pt_s[3] = { (int16_t)(cell_z & mask), (int16_t)(cell_y & mask), (int16_t)(cell_x & mask) };
							uint32_t fiz = (uint16_t)cell_z & (uint16_t)~mask;
							uint32_t fiy = (uint16_t)cell_y & (uint16_t)~mask;
							uint32_t fix = (uint16_t)cell_x & (uint16_t)~mask;
							uint32_t uz = use_fade ?
							              chaff_fade_frac_ex(fiz, frac_z, x, field->flags) :
							              (((uint32_t)fiz << 16) | frac_z);
							uint32_t uy = use_fade ?
							              chaff_fade_frac_ex(fiy, frac_y, x, field->flags) :
							              (((uint32_t)fiy << 16) | frac_y);
							uint32_t ux = use_fade ?
							              chaff_fade_frac_ex(fix, frac_x, x, field->flags) :
							              (((uint32_t)fix << 16) | frac_x);
							const uint32_t *v = noise_wcorners_periodic(&cache, pt_s, x, vshift, seed, 3, pshift);
							uint32_t c_mid[4];
							for (int i = 0; i < 4; i++) {
								int32_t st_z = noise_wstep_slope(v[i], v[i + 4], x);
								c_mid[i] = v[i] + (uint32_t)(((int64_t)st_z * uz) >> 16);
							}
							int32_t s0 = noise_wstep_slope(c_mid[0], c_mid[2], x);
							int32_t s1 = noise_wstep_slope(c_mid[1], c_mid[3], x);
							uint32_t b0 = c_mid[0] + (uint32_t)(((int64_t)s0 * uy) >> 16);
							uint32_t b1 = c_mid[1] + (uint32_t)(((int64_t)s1 * uy) >> 16);
							int32_t st = noise_wstep_slope(b0, b1, x);
							raw = b0 + (uint32_t)(((int64_t)st * ux) >> 16);
						} else if (field->dim == 4) {
							int16_t pt_s[4] = {
								(int16_t)(cell_w & mask),
								(int16_t)(cell_z & mask),
								(int16_t)(cell_y & mask),
								(int16_t)(cell_x & mask)
							};
							uint32_t fiw = (uint16_t)cell_w & (uint16_t)~mask;
							uint32_t fiz = (uint16_t)cell_z & (uint16_t)~mask;
							uint32_t fiy = (uint16_t)cell_y & (uint16_t)~mask;
							uint32_t fix = (uint16_t)cell_x & (uint16_t)~mask;
							uint32_t uw = use_fade ?
							              chaff_fade_frac_ex(fiw, frac_w, x, field->flags) :
							              (((uint32_t)fiw << 16) | frac_w);
							uint32_t uz = use_fade ?
							              chaff_fade_frac_ex(fiz, frac_z, x, field->flags) :
							              (((uint32_t)fiz << 16) | frac_z);
							uint32_t uy = use_fade ?
							              chaff_fade_frac_ex(fiy, frac_y, x, field->flags) :
							              (((uint32_t)fiy << 16) | frac_y);
							uint32_t ux = use_fade ?
							              chaff_fade_frac_ex(fix, frac_x, x, field->flags) :
							              (((uint32_t)fix << 16) | frac_x);
							const uint32_t *v = noise_wcorners_periodic(&cache, pt_s, x, vshift, seed, 4, pshift);
							uint32_t d_mid[8];
							for (int i = 0; i < 8; i++) {
								int32_t st_w = noise_wstep_slope(v[i], v[i + 8], x);
								d_mid[i] = v[i] + (uint32_t)(((int64_t)st_w * uw) >> 16);
							}
							uint32_t c_mid[4];
							for (int i = 0; i < 4; i++) {
								int32_t st_z = noise_wstep_slope(d_mid[i], d_mid[i + 4], x);
								c_mid[i] = d_mid[i] + (uint32_t)(((int64_t)st_z * uz) >> 16);
							}
							int32_t s0 = noise_wstep_slope(c_mid[0], c_mid[2], x);
							int32_t s1 = noise_wstep_slope(c_mid[1], c_mid[3], x);
							uint32_t b0 = c_mid[0] + (uint32_t)(((int64_t)s0 * uy) >> 16);
							uint32_t b1 = c_mid[1] + (uint32_t)(((int64_t)s1 * uy) >> 16);
							int32_t st = noise_wstep_slope(b0, b1, x);
							raw = b0 + (uint32_t)(((int64_t)st * ux) >> 16);
						}

						int32_t sample;
						if (lay->amp == 0) {
							sample = (int32_t)raw;
						} else {
							int32_t val = (int32_t)(raw >> 16) - mid;
							if (lay->seed_mode & CHAFF_LAYER_RIDGED)
								val = ridge_fold - abs(val);
							if (lay->contrast != 0 && lay->contrast != NOISE_CONTRAST_IDENTITY)
								val = (val * (int32_t)lay->contrast) >> 8;
							sample = (val * (int32_t)lay->amp) >> amp_shift;
						}
						row_out[c] += sample;
					}
				}
			}
		}
	}
}

/* --- Declarative Multi-Scale Cascades & Universal Accessors -------------- */

/** Maximum number of octaves supported by noise_scanline_t and cascade evaluators. */
#define NOISE_MAX_OCTAVES 16

/** Flag for noise_cascade_t::flags to explicitly request raw linear interpolation (disabling NOISE_FADE). */
#define NOISE_CASCADE_LINEAR 0x80u

/* Single noise octave specification */
typedef struct {
	uint8_t  shift;        /* Lattice size 2^shift (e.g. 10 = 1024, 3 = 8) */
	uint16_t amp;          /* Amplitude weight in fixed-point world units */
	int8_t   skew_x;       /* Domain skew: px = x + y / skew_x (0 = disabled) */
	int8_t   skew_y;       /* Domain skew: py = y - x / skew_y (0 = disabled) */
	uint16_t contrast;     /* Fixed-point contrast multiplier (256 = 1.0x, 512 = 2.0x) */
	uint32_t seed_salt;    /* Optional per-octave seed salt override (0 = use cascade default) */
	uint8_t  period_shift; /* Optional periodic domain shift 2^period_shift (0 = unbounded) */
} noise_octave_t;

/* Multi-scale composite cascade */
typedef struct {
	uint32_t seed;
	uint8_t  count;
	const noise_octave_t *octaves;
	uint8_t  vshift;     /* Value bit-shift per octave (0 defaults to NOISE_DEFAULT_VSHIFT = 8) */
	uint8_t  flags;      /* Sampling flags (0 defaults to NOISE_FADE; NOISE_CASCADE_LINEAR for raw linear) */
	uint32_t seed_prime; /* Optional per-octave seed prime (0 = CHAFF_SEED_PRIME) */
	uint32_t seed_salt;  /* Optional per-octave seed salt offset (0 = 1u) */
} noise_cascade_t;

static inline unsigned
noise_cascade_vshift(const noise_cascade_t *cfg)
{
	return (cfg && cfg->vshift != 0) ? (unsigned)cfg->vshift : NOISE_DEFAULT_VSHIFT;
}

static inline unsigned
noise_cascade_flags(const noise_cascade_t *cfg)
{
	if (!cfg || cfg->flags == 0)
		return NOISE_FADE;
	if (cfg->flags & NOISE_CASCADE_LINEAR)
		return (unsigned)(cfg->flags & ~NOISE_CASCADE_LINEAR);
	return (unsigned)cfg->flags;
}

static inline uint32_t
noise_cascade_octave_seed(const noise_cascade_t *cfg, const noise_octave_t *oct, uint8_t idx)
{
	if (!cfg) return 0;
	uint32_t prime = cfg->seed_prime ? cfg->seed_prime : CHAFF_SEED_PRIME;
	uint32_t salt  = (oct && oct->seed_salt) ? oct->seed_salt : (cfg->seed_salt ? cfg->seed_salt : 1u);
	return cfg->seed ^ (uint32_t)(idx * prime + salt);
}

/* Corner cache: aliased to noise_wcache_t to eliminate layout drift and type punning */
typedef noise_wcache_t noise_cache_t;

typedef struct {
	noise_wcache_t caches[NOISE_MAX_OCTAVES];
	const noise_cascade_t *cfg;
	uint8_t dim;
	int16_t w;
} noise_scanline_t;

static inline void
noise_scanline_init(noise_scanline_t *scan, const noise_cascade_t *cfg, uint8_t dim)
{
	if (!scan) return;
	memset(scan, 0, sizeof(*scan));
	scan->cfg = cfg;
	scan->dim = dim;
	for (int i = 0; i < NOISE_MAX_OCTAVES; i++)
		noise_wcache_init(&scan->caches[i]);
}

/* 1. Universal Scanline Stepper with Corner Cache
 * Evaluates rows in-register, applying skew per sample so values match the point sampler.
 * Zero heap allocations. */
static inline void
noise_scanline_eval_row(noise_scanline_t *scan,
                        int16_t start_x, int16_t y, int16_t z,
                        size_t count, int32_t *out_row)
{
	if (!scan || !scan->cfg || count == 0 || !out_row) return;
	memset(out_row, 0, count * sizeof(int32_t));

	const noise_cascade_t *cfg = scan->cfg;
	uint8_t count_oct = cfg->count > NOISE_MAX_OCTAVES ? NOISE_MAX_OCTAVES : cfg->count;
	unsigned vshift = noise_cascade_vshift(cfg);
	unsigned fl = noise_cascade_flags(cfg);
	int32_t mid = (vshift < 16) ? (int32_t)(1u << (15 - vshift)) : 0;
	unsigned amp_shift = (vshift < 16) ? (unsigned)(15 - vshift) : 1u;

	for (uint8_t o = 0; o < count_oct; o++) {
		const noise_octave_t *oct = &cfg->octaves[o];
		uint32_t oct_seed = noise_cascade_octave_seed(cfg, oct, o);
		noise_wcache_t *c = &scan->caches[o];

		for (size_t i = 0; i < count; i++) {
			int16_t x = (int16_t)(start_x + (int)i);
			int16_t px = x, py = y;
			if (oct->skew_x) px = (int16_t)(px + py / oct->skew_x);
			if (oct->skew_y) py = (int16_t)(py - px / oct->skew_y);

			uint32_t raw;
			if (scan->dim == 2) {
				int16_t p[2] = { py, px };
				raw = noise_wpoint_c_periodic(c, p, oct->shift, vshift, oct_seed, 2, fl, oct->period_shift);
			} else if (scan->dim == 4) {
				int16_t p[4] = { scan->w, z, py, px };
				raw = noise_wpoint_c_periodic(c, p, oct->shift, vshift, oct_seed, 4, fl, oct->period_shift);
			} else {
				int16_t p[3] = { z, py, px };
				raw = noise_wpoint_c_periodic(c, p, oct->shift, vshift, oct_seed, 3, fl, oct->period_shift);
			}
			int32_t val = (int32_t)(raw >> 16) - mid;
			if (oct->contrast != 0 && oct->contrast != NOISE_CONTRAST_IDENTITY)
				val = (val * (int32_t)oct->contrast) >> 8;
			out_row[i] += (val * (int32_t)oct->amp) >> amp_shift;
		}
	}
}

/* 2. Universal Bulk Matrix Stepper
 * Fills 2^size_shift x 2^size_shift chunk by scanline rows.
 * Guarantees out_matrix[r * side + c] == noise_sample_2d(origin_x + c, origin_y + r).
 * Zero heap allocations. */
static inline void
noise_matrix_fill_2d(const noise_cascade_t *cfg,
                     int16_t origin_x, int16_t origin_y,
                     uint8_t size_shift, int32_t *out_matrix)
{
	if (!cfg || cfg->count == 0 || !out_matrix) return;
	size_t side = (size_t)1 << size_shift;
	noise_scanline_t scan;
	noise_scanline_init(&scan, cfg, 2);
	for (size_t r = 0; r < side; r++) {
		noise_scanline_eval_row(&scan, origin_x, (int16_t)(origin_y + (int)r), 0,
		                        side, out_matrix + r * side);
	}
}

/* 3. Sparse Point Sampler (Collision, Entrances, Caverns) */
static inline int32_t
noise_sample_2d(const noise_cascade_t *cfg, int16_t x, int16_t y)
{
	if (!cfg || cfg->count == 0) return 0;
	int32_t total = 0;
	unsigned vshift = noise_cascade_vshift(cfg);
	unsigned fl = noise_cascade_flags(cfg);
	int32_t mid = (vshift < 16) ? (int32_t)(1u << (15 - vshift)) : 0;
	unsigned amp_shift = (vshift < 16) ? (unsigned)(15 - vshift) : 1u;
	for (uint8_t i = 0; i < cfg->count; i++) {
		const noise_octave_t *oct = &cfg->octaves[i];
		uint32_t oct_seed = noise_cascade_octave_seed(cfg, oct, i);
		int16_t px = x, py = y;
		if (oct->skew_x) px = (int16_t)(px + py / oct->skew_x);
		if (oct->skew_y) py = (int16_t)(py - px / oct->skew_y);
		int16_t p[2] = { py, px };
		uint32_t raw = noise_wpoint_n_periodic(p, oct->shift, vshift, oct_seed, 2, fl, oct->period_shift);
		int32_t val = (int32_t)(raw >> 16) - mid;
		if (oct->contrast != 0 && oct->contrast != NOISE_CONTRAST_IDENTITY)
			val = (val * (int32_t)oct->contrast) >> 8;
		total += (val * (int32_t)oct->amp) >> amp_shift;
	}
	return total;
}

static inline int32_t
noise_sample_3d(const noise_cascade_t *cfg, int16_t x, int16_t y, int16_t z)
{
	if (!cfg || cfg->count == 0) return 0;
	int32_t total = 0;
	unsigned vshift = noise_cascade_vshift(cfg);
	unsigned fl = noise_cascade_flags(cfg);
	int32_t mid = (vshift < 16) ? (int32_t)(1u << (15 - vshift)) : 0;
	unsigned amp_shift = (vshift < 16) ? (unsigned)(15 - vshift) : 1u;
	for (uint8_t i = 0; i < cfg->count; i++) {
		const noise_octave_t *oct = &cfg->octaves[i];
		uint32_t oct_seed = noise_cascade_octave_seed(cfg, oct, i);
		int16_t px = x, py = y, pz = z;
		if (oct->skew_x) px = (int16_t)(px + py / oct->skew_x);
		if (oct->skew_y) py = (int16_t)(py - px / oct->skew_y);
		int16_t p[3] = { pz, py, px };
		uint32_t raw = noise_wpoint_n_periodic(p, oct->shift, vshift, oct_seed, 3, fl, oct->period_shift);
		int32_t val = (int32_t)(raw >> 16) - mid;
		if (oct->contrast != 0 && oct->contrast != NOISE_CONTRAST_IDENTITY)
			val = (val * (int32_t)oct->contrast) >> 8;
		total += (val * (int32_t)oct->amp) >> amp_shift;
	}
	return total;
}

static inline int32_t
noise_sample_4d(const noise_cascade_t *cfg, int16_t x, int16_t y, int16_t z, int16_t w)
{
	if (!cfg || cfg->count == 0) return 0;
	int32_t total = 0;
	unsigned vshift = noise_cascade_vshift(cfg);
	unsigned fl = noise_cascade_flags(cfg);
	int32_t mid = (vshift < 16) ? (int32_t)(1u << (15 - vshift)) : 0;
	unsigned amp_shift = (vshift < 16) ? (unsigned)(15 - vshift) : 1u;
	for (uint8_t i = 0; i < cfg->count; i++) {
		const noise_octave_t *oct = &cfg->octaves[i];
		uint32_t oct_seed = noise_cascade_octave_seed(cfg, oct, i);
		int16_t px = x, py = y, pz = z, pw = w;
		if (oct->skew_x) px = (int16_t)(px + py / oct->skew_x);
		if (oct->skew_y) py = (int16_t)(py - px / oct->skew_y);
		int16_t p[4] = { pw, pz, py, px };
		uint32_t raw = noise_wpoint_n_periodic(p, oct->shift, vshift, oct_seed, 4, fl, oct->period_shift);
		int32_t val = (int32_t)(raw >> 16) - mid;
		if (oct->contrast != 0 && oct->contrast != NOISE_CONTRAST_IDENTITY)
			val = (val * (int32_t)oct->contrast) >> 8;
		total += (val * (int32_t)oct->amp) >> amp_shift;
	}
	return total;
}

/* Canonical declarative libchaff aliases */
typedef noise_octave_t   chaff_octave_t;
typedef noise_cascade_t  chaff_cascade_t;
typedef noise_cache_t    chaff_cache_t;
typedef noise_scanline_t chaff_scanline_t;

static inline void
chaff_matrix_fill_2d(const chaff_cascade_t *cfg,
                     int16_t origin_x, int16_t origin_y,
                     uint8_t size_shift, int32_t *out_matrix)
{
	noise_matrix_fill_2d(cfg, origin_x, origin_y, size_shift, out_matrix);
}

static inline void
chaff_scanline_init(chaff_scanline_t *scan, const chaff_cascade_t *cfg, uint8_t dim)
{
	noise_scanline_init(scan, cfg, dim);
}

static inline void
chaff_scanline_eval_row(chaff_scanline_t *scan,
                        int16_t start_x, int16_t y, int16_t z,
                        size_t count, int32_t *out_row)
{
	noise_scanline_eval_row(scan, start_x, y, z, count, out_row);
}

static inline int32_t
chaff_sample_2d(const chaff_cascade_t *cfg, int16_t x, int16_t y)
{
	return noise_sample_2d(cfg, x, y);
}

static inline int32_t
chaff_sample_3d(const chaff_cascade_t *cfg, int16_t x, int16_t y, int16_t z)
{
	return noise_sample_3d(cfg, x, y, z);
}

static inline int32_t
chaff_sample_4d(const chaff_cascade_t *cfg, int16_t x, int16_t y, int16_t z, int16_t w)
{
	return noise_sample_4d(cfg, x, y, z, w);
}

#endif /* TTYPT_NOISE_H */

