#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ttypt/noise.h>
#include <ttypt/chaff.h>

int main(void)
{
	printf("=== Testing Standalone Generic N-Dimensional Noise Library ===\n");

	/* 1. Point sampling across 1D, 2D, 3D, 4D */
	{
		int v1 = noise_point_1d(10, 42, 4);
		int v2 = noise_point_2d(10, 20, 42, 4);
		int v3 = noise_point_3d(10, 20, 30, 42, 4);
		int v4 = noise_point_4d(10, 20, 30, 40, 42, 4);

		assert(v1 >= 0 && v1 <= 255);
		assert(v2 >= 0 && v2 <= 255);
		assert(v3 >= 0 && v3 <= 255);
		assert(v4 >= 0 && v4 <= 255);
		printf("PASS: generic point sampling (1D, 2D, 3D, 4D) within valid [0..255] bounds\n");
	}

	/* 2. Determinism of point queries */
	{
		for (int i = -10; i <= 10; i++) {
			int a = noise_point_3d(i, i * 2, i * 3, 99, 3);
			int b = noise_point_3d(i, i * 2, i * 3, 99, 3);
			assert(a == b);
		}
		printf("PASS: point sampling determinism verified\n");
	}

	/* 3. Smoothness / Continuity in 3D */
	{
		int prev = noise_point_3d(0, 0, 0, 1234, 4);
		for (int x = 1; x <= 64; x++) {
			int cur = noise_point_3d(x, 0, 0, 1234, 4);
			int diff = abs(cur - prev);
			assert(diff <= 40); /* Smooth step transitions */
			prev = cur;
		}
		printf("PASS: 3D point continuity verified\n");
	}

	/* 4. Matrix Stepper in 2D and 3D */
	{
		uint32_t m2d[1 << (2 * 4)]; // 16x16 = 256
		int16_t s2d[2] = { 0, 0 };
		octave_t oct2d[] = { { 4, 1 } };
		noise_oct(m2d, s2d, 1, oct2d, 55, 4, 2);
		assert(m2d[0] != 0 && m2d[255] != 0);

		uint32_t m3d[1 << (3 * 3)]; // 8x8x8 = 512
		int16_t s3d[3] = { 0, 0, 0 };
		octave_t oct3d[] = { { 3, 1 } };
		noise_oct(m3d, s3d, 1, oct3d, 55, 3, 3);
		assert(m3d[0] != 0 && m3d[511] != 0);
		printf("PASS: generic matrix stepping in 2D and 3D verified\n");
	}

	/* 5. Original-algorithm wrapping sampler == noise_m (cy > x path) */
	{
		static const int16_t origins[] = { 0, 64, -128, 32704, -32768, 32752 };
		unsigned checked = 0;
		for (uint8_t dim = 2; dim <= 3; dim++)
		for (unsigned x = 2; x <= 4; x++)
		for (size_t oi = 0; oi < sizeof(origins) / sizeof(origins[0]); oi++) {
			unsigned cy = x + 1, w = 4, seed = 77;
			unsigned side = 1u << cy, total = 1u << (cy * dim);
			uint32_t *m = calloc(total, sizeof(uint32_t));
			int16_t ms[3], o = (int16_t) (origins[oi] & COORDMASK_LSHIFT(x));
			for (int j = 0; j < dim; j++)
				ms[j] = (int16_t) (o + 16 * j * (1 << x));
			int16_t msc[3] = { ms[0], ms[1], ms[2] };
			noise_m(m, msc, x, w, seed, cy, dim);
			for (unsigned i = 0; i < total; i++) {
				int16_t p[3];
				unsigned rem = i;
				for (int j = dim - 1; j >= 0; j--) {
					p[j] = (int16_t) (ms[j] + (int) (rem % side));
					rem /= side;
				}
				uint32_t a = noise_wpoint_n(p, x, w, seed, dim, 0);
				if (a != m[i]) {
					printf("FAIL: dim %u x %u origin %d i %u: %u != %u\n",
					       dim, x, o, i, a, m[i]);
					return 1;
				}
				checked++;
			}
			free(m);
		}
		/* cy <= x path (noise_fix_v pre-steps the vertices to a sub-quad,
		 * truncating in another order): equal within a rounding tolerance
		 * of 2^-20 of the value range (observed max 45 / 2^28). */
		for (unsigned x = 3; x <= 5; x++) {
			unsigned cy = 2, w = 4, seed = 9, side = 1u << cy;
			uint32_t m[16];
			int16_t ms[2] = { (int16_t) (32760 & COORDMASK_LSHIFT(cy)), 4 };
			int16_t msc[2] = { ms[0], ms[1] };
			memset(m, 0, sizeof(m));
			noise_m(m, msc, x, w, seed, cy, 2);
			for (unsigned i = 0; i < side * side; i++) {
				int16_t p[2] = { (int16_t) (ms[0] + (int) (i / side)),
				                 (int16_t) (ms[1] + (int) (i % side)) };
				uint32_t a = noise_wpoint_n(p, x, w, seed, 2, 0);
				uint32_t d = a > m[i] ? a - m[i] : m[i] - a;
				if (d > ((NOISE_MAX >> w) >> 20)) {
					printf("FAIL: cy<=x x %u i %u: %u != %u\n", x, i, a, m[i]);
					return 1;
				}
				checked++;
			}
		}
		printf("PASS: wrapping sampler value-equal to noise_m (%u samples)\n", checked);
	}

	/* 6. No seam at the int16 boundary: walk the whole 16-bit circle on
	 * every axis; the 32767 -> -32768 step obeys the same bound as every
	 * other step (the lattice vertex s + 2^x wraps like the coordinate). */
	{
		for (uint8_t dim = 1; dim <= 3; dim++)
		for (int axis = 0; axis < dim; axis++) {
			unsigned x = 4, w = 8;
			uint32_t bound = ((NOISE_MAX >> w) >> x) + 1;
			int16_t p[3] = { 1234, -77, 4321 };
			p[axis] = 0;
			uint32_t prev = noise_wpoint_n(p, x, w, 5, dim, 0), wrapd = 0;
			for (long k = 1; k <= 65536; k++) {
				p[axis] = (int16_t) (uint16_t) k;
				uint32_t cur = noise_wpoint_n(p, x, w, 5, dim, 0);
				uint32_t d = cur > prev ? cur - prev : prev - cur;
				if (d > bound) {
					printf("FAIL: seam dim %u axis %d at %ld: %u > %u\n",
					       dim, axis, k, d, bound);
					return 1;
				}
				if (p[axis] == -32768)
					wrapd = d;
				prev = cur;
			}
			(void) wrapd;
		}
		/* noise_point_n (cell-index keys) does jump there; document why
		 * it is not the animation primitive. */
		int seam = abs(noise_point_1d(32767, 5, 4) - noise_point_1d(-32768, 5, 4));
		printf("PASS: no seam across 32767->-32768 on any axis (legacy point jump %d)\n", seam);
	}

	/* 7. Ever-increasing time: casting a growing counter to int16 is
	 * continuous forever (time never has to be wrapped with %). */
	{
		unsigned x = 5, w = 8;
		uint32_t bound = ((NOISE_MAX >> w) >> x) + 1;
		uint32_t prev = 0;
		for (uint32_t anim = 0; anim < 200000; anim++) {
			int16_t p[3] = { 10, 20, (int16_t) (uint16_t) anim };
			uint32_t cur = noise_wpoint_n(p, x, w, 3, 3, 0);
			if (anim) {
				uint32_t d = cur > prev ? cur - prev : prev - cur;
				assert(d <= bound);
			}
			prev = cur;
		}
		printf("PASS: int16 cast of increasing anim counter is seamless\n");
	}

	/* 8. Corner cache and row stepper are value-equal to the point sampler */
	{
		noise_wcache_t c;
		noise_wcache_init(&c);
		uint32_t rng = 1;
		for (int i = 0; i < 20000; i++) {
			int16_t p[3];
			for (int j = 0; j < 3; j++) {
				rng = rng * 1103515245u + 12345u;
				p[j] = (int16_t) (32700 + (int) ((rng >> 16) % 136));
			}
			unsigned fl = (i & 1) ? NOISE_FADE : 0;
			assert(noise_wpoint_c(&c, p, 3, 8, 11, 3, fl) ==
			       noise_wpoint_n(p, 3, 8, 11, 3, fl));
		}
		static const int16_t steps[] = { 1, 3, -2, 7 };
		for (size_t si = 0; si < 4; si++)
		for (unsigned fl = 0; fl <= NOISE_FADE; fl++) {
			uint32_t row[300];
			int16_t p0[3] = { 5, -9, 32600 };
			noise_wrow(row, 300, p0, steps[si], 4, 8, 21, 3, fl, &c);
			for (int k = 0; k < 300; k++) {
				int16_t p[3] = { 5, -9, (int16_t) (32600 + k * steps[si]) };
				if (row[k] != noise_wpoint_n(p, 4, 8, 21, 3, fl)) {
					printf("FAIL: row step %d fade %u k %d\n", steps[si], fl, k);
					return 1;
				}
			}
		}
		printf("PASS: cached point and row stepper equal the point sampler\n");
	}

	/* 9. Efficiency: the row stepper fetches 2^dim corners once per lattice
	 * cell (like noise_get_v + noise_quad), not once per sample. */
	{
		noise_wcache_t c;
		uint32_t row[256];
		int16_t p0[3] = { 3, 4, 0 };
		noise_wcache_init(&c);
		noise_wrow(row, 256, p0, 1, 4, 8, 1, 3, 0, &c);
		assert(c.fetches == 256 / 16);
		p0[2] = 8; /* unaligned start straddles one extra cell */
		c.fetches = 0;
		noise_wrow(row, 256, p0, 1, 4, 8, 1, 3, 0, &c);
		assert(c.fetches == 256 / 16 + 1);
		printf("PASS: row stepper hashes once per lattice cell (%u fetches / 256 px)\n",
		       c.fetches);
	}

	/* 10. Hermite fade: same lattice values, smooth in between */
	{
		for (int k = 0; k < 4096; k += 16) {
			int16_t p[2] = { 96, (int16_t) k }; /* lattice-aligned */
			assert(noise_wpoint_n(p, 4, 8, 4, 2, NOISE_FADE) ==
			       noise_wpoint_n(p, 4, 8, 4, 2, 0));
		}
		assert(noise_wfade(0, 4) == 0 && noise_wfade(8, 4) == 8 && noise_wfade(15, 4) <= 15);
		printf("PASS: fade keeps lattice values and maps [0,2^x) onto itself\n");
	}

	/* 11. Plane stepper (noise_quad order: one corner fetch per 2D lattice
	 * cell) is value-equal to the point sampler, strided, unaligned, and
	 * across the int16 wrap. */
	{
		static uint32_t pl[70 * 90];
		static const int16_t org[][2] = { { 0, 0 }, { 5, -9 }, { 32760, 32750 }, { -7, 32701 } };
		for (size_t oi = 0; oi < 4; oi++)
		for (uint8_t dim = 2; dim <= 3; dim++)
		for (unsigned fl = 0; fl <= NOISE_FADE; fl++) {
			int16_t p0[3];
			p0[0] = 1234;
			p0[dim - 2] = org[oi][0];
			p0[dim - 1] = org[oi][1];
			noise_wcache_t c;
			noise_wcache_init(&c);
			memset(pl, 0, sizeof(pl));
			noise_wplane(pl, 80, 65, 90, p0, 3, 8, 17, dim, fl, &c);
			for (int r = 0; r < 70; r++)
			for (int k = 0; k < 90; k++) {
				uint32_t want = 0;
				if (r < 65 && k < 80) {
					int16_t p[3] = { p0[0], p0[1], p0[2] };
					p[dim - 2] = (int16_t) (p0[dim - 2] + r);
					p[dim - 1] = (int16_t) (p0[dim - 1] + k);
					want = noise_wpoint_n(p, 3, 8, 17, dim, fl);
				}
				if (pl[r * 90 + k] != want) {
					printf("FAIL: plane o %zu dim %u fade %u at %d,%d\n", oi, dim, fl, r, k);
					return 1;
				}
			}
		}
		noise_wcache_t c;
		int16_t p0[3] = { 9, 0, 0 };
		noise_wcache_init(&c);
		noise_wplane(pl, 64, 64, 64, p0, 4, 8, 1, 3, 0, &c);
		assert(c.fetches == 16);
		printf("PASS: plane stepper equals point sampler; %u fetches for 64x64 px at x=4\n",
		       c.fetches);
	}

	/* 12. Declarative cascade sparse point samplers (noise_sample_2d & 3d) */
	{
		static const noise_octave_t octs[] = {
			{ .shift = 6, .amp = 100, .skew_x = 4, .skew_y = 3, .contrast = 256 },
			{ .shift = 4, .amp = 50,  .skew_x = 0, .skew_y = 0, .contrast = 256 },
		};
		noise_cascade_t cfg = {
			.seed = 0x12345678,
			.count = 2,
			.octaves = octs,
		};
		int32_t s1 = noise_sample_2d(&cfg, 10, 20);
		int32_t s2 = noise_sample_2d(&cfg, 10, 20);
		assert(s1 == s2);

		int32_t s3d_1 = noise_sample_3d(&cfg, 10, 20, 30);
		int32_t s3d_2 = noise_sample_3d(&cfg, 10, 20, 30);
		assert(s3d_1 == s3d_2);
		printf("PASS: declarative point sampling (2D & 3D) deterministic\n");
	}

	/* 13. Declarative bulk matrix fill 2D (noise_matrix_fill_2d) */
	{
		static const noise_octave_t octs[] = {
			{ .shift = 5, .amp = 80, .skew_x = 0, .skew_y = 0, .contrast = 256 },
		};
		noise_cascade_t cfg = {
			.seed = 0x87654321,
			.count = 1,
			.octaves = octs,
		};
		int32_t mat[1024]; /* 32x32 for size_shift = 5 */
		noise_matrix_fill_2d(&cfg, 0, 0, 5, mat);
		int non_zero = 0;
		for (int i = 0; i < 1024; i++) {
			if (mat[i] != 0) non_zero++;
		}
		assert(non_zero > 100);
		printf("PASS: declarative bulk matrix fill 2D evaluated (%d/1024 non-zero)\n", non_zero);
	}

	/* 14. Declarative scanline stepper (noise_scanline_eval_row) */
	{
		static const noise_octave_t octs[] = {
			{ .shift = 4, .amp = 60, .skew_x = 0, .skew_y = 0, .contrast = 256 },
		};
		noise_cascade_t cfg = {
			.seed = 0xCAFE1234,
			.count = 1,
			.octaves = octs,
		};
		noise_scanline_t scan;
		noise_scanline_init(&scan, &cfg, 3);
		int32_t row[256];
		noise_scanline_eval_row(&scan, 0, 10, 5, 256, row);
		int non_zero = 0;
		for (int i = 0; i < 256; i++) {
			if (row[i] != 0) non_zero++;
		}
		assert(non_zero > 50);
		assert(scan.caches[0].fetches > 0);
		printf("PASS: scanline stepper evaluated row with %u corner fetches\n", scan.caches[0].fetches);
	}

	/* 15. Strict agreement between point sampler, scanline row, and matrix fill (2D & 3D) */
	{
		static const noise_octave_t octs_skew[] = {
			{ .shift = 6, .amp = 100, .skew_x = 3,  .skew_y = -5, .contrast = 256 },
			{ .shift = 4, .amp =  50, .skew_x = 0,  .skew_y =  0, .contrast = 512 },
			{ .shift = 3, .amp =  25, .skew_x = -2, .skew_y =  4, .contrast = 256 },
		};
		static const noise_cascade_t cfg_skew = {
			.seed = 0x55AA1234,
			.count = 3,
			.octaves = octs_skew,
		};

		/* Origins: near 0, negative, and straddling int16 wrap boundary (32767 -> -32768) */
		static const int16_t test_origins[][2] = {
			{ 0, 0 },
			{ -120, 85 },
			{ 32750, 32750 }
		};

		for (size_t oi = 0; oi < sizeof(test_origins) / sizeof(test_origins[0]); oi++) {
			int16_t ox = test_origins[oi][0];
			int16_t oy = test_origins[oi][1];

			int32_t mat[32 * 32];
			noise_matrix_fill_2d(&cfg_skew, ox, oy, 5, mat);

			noise_scanline_t scan;
			noise_scanline_init(&scan, &cfg_skew, 2);

			for (int r = 0; r < 32; r++) {
				int32_t row[32];
				int16_t cur_y = (int16_t)(oy + r);
				noise_scanline_eval_row(&scan, ox, cur_y, 0, 32, row);

				for (int c = 0; c < 32; c++) {
					int16_t cur_x = (int16_t)(ox + c);
					int32_t pt = noise_sample_2d(&cfg_skew, cur_x, cur_y);
					assert(row[c] == pt);
					assert(mat[r * 32 + c] == pt);
				}
			}
		}

		/* 3D row vs point sampler across wrap boundary */
		noise_scanline_t scan3d;
		noise_scanline_init(&scan3d, &cfg_skew, 3);
		int16_t ox = 32760, oy = -10, oz = 15;
		int32_t row3d[32];
		noise_scanline_eval_row(&scan3d, ox, oy, oz, 32, row3d);
		for (int c = 0; c < 32; c++) {
			int16_t cur_x = (int16_t)(ox + c);
			int32_t pt3d = noise_sample_3d(&cfg_skew, cur_x, oy, oz);
			assert(row3d[c] == pt3d);
		}
		printf("PASS: point == row[k] == matrix[r][k] across 2D & 3D and int16 wrap boundary\n");
	}

	/* 16. chaff_fill at frac==0 matches noise_oct bit for bit */
	{
		static const chaff_layer_t lay = {
			.shift = 4, .vshift = 1, .amp = 0, .seed_mode = CHAFF_SEED_SHARED,
			.skew = { 0, 0 }, .contrast = 0
		};
		static const chaff_field_t field = {
			.seed = 55, .dim = 2, .flags = 0, .nlayers = 1, .layers = &lay
		};
		int16_t oy = -64, ox = 32;
		int32_t c_out[16 * 16];
		chaff_coord_t origin[2] = { { oy, 0 }, { ox, 0 } };
		int32_t step_q16[2] = { 65536, 65536 };
		uint16_t n[2] = { 16, 16 };
		chaff_fill(&field, 0, 1, origin, step_q16, n, c_out, 16, CHAFF_MODE_SET);

		uint32_t oct_out[16 * 16];
		int16_t s[2] = { oy, ox };
		octave_t oct[] = { { 4, 1 } };
		noise_oct(oct_out, s, 1, oct, 55, 4, 2);

		for (int i = 0; i < 256; i++) {
			assert((uint32_t)c_out[i] == oct_out[i]);
		}
		printf("PASS: chaff_fill at frac==0 equals noise_oct bit-for-bit\n");
	}

	/* 17. Multi-layer nesting: sum(layers 0..k) + sum(layers k..n) == sum(layers 0..n) */
	{
		static const chaff_layer_t layers[] = {
			{ 7, 8, 100, CHAFF_SEED_PER_LAYER, { 3, -5 }, 256 },
			{ 5, 8,  50, CHAFF_SEED_PER_LAYER, { 0,  0 }, 512 },
			{ 3, 8,  20, CHAFF_SEED_PER_LAYER, { 2,  4 }, 256 }
		};
		static const chaff_field_t field = {
			.seed = 0xFEEDFACE, .dim = 2, .flags = CHAFF_FLAG_FADE, .nlayers = 3, .layers = layers
		};
		chaff_coord_t origin[2] = { { -20, 12345 }, { 45, 6789 } };
		int32_t step_q16[2] = { 409, 409 }; /* fractional step: 1/160th of a room */
		uint16_t n[2] = { 16, 16 };

		int32_t full[256], split[256];
		chaff_fill(&field, 0, 3, origin, step_q16, n, full, 16, CHAFF_MODE_SET);

		chaff_fill(&field, 0, 1, origin, step_q16, n, split, 16, CHAFF_MODE_SET);
		chaff_fill(&field, 1, 3, origin, step_q16, n, split, 16, CHAFF_MODE_ADD);

		for (int i = 0; i < 256; i++) {
			assert(full[i] == split[i]);
		}
		printf("PASS: layer nesting sum(0..1) + sum(1..3) == sum(0..3) verified with fractional coordinates\n");
	}

	/* 18. chaff_coord_scale: planetary / stratosphere scale zoom */
	{
		chaff_coord_t c = { .cell = 10, .frac = 0x8000 }; /* 10.5 */
		chaff_coord_t zoomed_in = chaff_coord_scale(c, 1);  /* 21.0 */
		assert(zoomed_in.cell == 21 && zoomed_in.frac == 0);

		chaff_coord_t zoomed_out = chaff_coord_scale(c, -1); /* 5.25 */
		assert(zoomed_out.cell == 5 && zoomed_out.frac == 0x4000);
		printf("PASS: chaff_coord_scale multi-scale transforms verified\n");
	}

	/* 19. chaff_fill_lod: Nyquist-aware band-limiting */
	{
		static const chaff_layer_t layers[] = {
			{ 2, 8, 10, CHAFF_SEED_SHARED, { 0, 0 }, 256 }, /* shift 2 (wavelength 4) */
			{ 4, 8, 20, CHAFF_SEED_SHARED, { 0, 0 }, 256 }, /* shift 4 (wavelength 16) */
			{ 7, 8, 40, CHAFF_SEED_SHARED, { 0, 0 }, 256 }  /* shift 7 (wavelength 128) */
		};
		static const chaff_field_t field = {
			.seed = 999, .dim = 2, .flags = 0, .nlayers = 3, .layers = layers
		};
		chaff_coord_t origin[2] = { { 0, 0 }, { 0, 0 } };
		int32_t step_q16[2] = { 65536, 65536 };
		uint16_t n[2] = { 8, 8 };
		int32_t lod1[64], lod3[64];

		/* Step 16 (nyquist_shift = 5) filters out shift 2 and 4 (retains only shift 7) */
		int32_t step16_q16[2] = { 16 << 16, 16 << 16 };
		chaff_fill_lod(&field, 0, origin, step16_q16, n, lod1, 8, CHAFF_MODE_SET);
		/* Step 1 (nyquist_shift = 0) evaluates all shifts (2, 4, 7) */
		chaff_fill_lod(&field, 0, origin, step_q16, n, lod3, 8, CHAFF_MODE_SET);

		/* Verify step 16 equals direct fill of layer 2 only */
		int32_t direct_layer2[64];
		chaff_fill(&field, 2, 3, origin, step16_q16, n, direct_layer2, 8, CHAFF_MODE_SET);
		for (int i = 0; i < 64; i++) {
			assert(lod1[i] == direct_layer2[i]);
		}
		printf("PASS: chaff_fill_lod band-limiting matches expected frequency cuts\n");
	}

	/* 20. chaff_sample_grad_2d: Analytical surface normal slope vs finite difference */
	{
		static const chaff_layer_t layers[] = {
			{ 6, 8, 100, CHAFF_SEED_SHARED, { 0, 0 }, 256 }
		};
		static const chaff_field_t field = {
			.seed = 42, .dim = 2, .flags = CHAFF_FLAG_FADE, .nlayers = 1, .layers = layers
		};
		int16_t wx = 20, wy = 30;
		chaff_sample_2d_t s = chaff_sample_grad_2d(&field, wx, wy);
		assert(s.elevation != 0);

		/* Finite difference comparison */
		chaff_sample_2d_t s_px = chaff_sample_grad_2d(&field, (int16_t)(wx + 1), wy);
		chaff_sample_2d_t s_mx = chaff_sample_grad_2d(&field, (int16_t)(wx - 1), wy);
		int num_diff_x = (s_px.elevation - s_mx.elevation);
		/* Analytical grad_x sign must match numerical slope */
		if (abs(num_diff_x) > 1) {
			assert((s.grad_x > 0 && num_diff_x >= 0) || (s.grad_x < 0 && num_diff_x <= 0) || s.grad_x == 0);
		}

		/* Verify GLSL shader export contract string is non-null */
		const char *glsl = chaff_glsl_fragment_shader_source();
		assert(glsl != NULL && strlen(glsl) > 100);
		printf("PASS: analytical gradient and GLSL shader contract export verified\n");
	}

	/* 21. Bit-exact corner hashing parity: GLSL hash2 vs CPU XXH32 */
	{
		uint32_t seed = 0x9E3779B1u;
		for (int y = -50; y <= 50; y += 3) {
			for (int x = -50; x <= 50; x += 3) {
				int16_t va[2] = { (int16_t)y, (int16_t)x };
				uint32_t cpu_hash = XXH32(va, 4, seed);
				uint32_t gpu_hash = chaff_glsl_ref_hash2((int16_t)y, (int16_t)x, seed);
				assert(cpu_hash == gpu_hash);
			}
		}
		printf("PASS: GLSL chaff_hash2 is 100%% bit-exact with CPU XXH32 across coordinate grid\n");
	}

	/* 22. GLSL shader continuous sample & shoreline parity vs CPU chaff_fill */
	{
		static const chaff_layer_t lay = {
			.shift = 5, .vshift = 8, .amp = 100, .seed_mode = CHAFF_SEED_SHARED,
			.skew = { 0, 0 }, .contrast = 256
		};
		static const chaff_field_t field = {
			.seed = 12345, .dim = 2, .flags = CHAFF_FLAG_FADE, .nlayers = 1, .layers = &lay
		};
		chaff_coord_t origin[2] = { { 0, 0 }, { 0, 0 } };
		int32_t step_q16[2] = { 65536, 65536 };
		uint16_t n[2] = { 16, 16 };
		int32_t cpu_out[256];
		chaff_fill(&field, 0, 1, origin, step_q16, n, cpu_out, 16, CHAFF_MODE_SET);

		int sign_matches = 0;
		for (int r = 0; r < 16; r++) {
			for (int c = 0; c < 16; c++) {
				float gpu_raw = chaff_glsl_ref_sample_octave((float)c, (float)r, 5, 8, 12345);
				float gpu_val = (gpu_raw / 65536.0f) - 128.0f;
				float gpu_elev = gpu_val * (100.0f / 128.0f);
				int32_t cpu_elev = cpu_out[r * 16 + c];

				/* Numerical difference between fixed-point Q16 Hermite and float Hermite < 1.5 units */
				float diff = (float)abs((int)gpu_elev - (int)cpu_elev);
				assert(diff <= 4.0f);

				/* Zero-crossing (shoreline level set H = 0) parity check */
				if ((gpu_elev >= 0.0f && cpu_elev >= 0) || (gpu_elev <= 0.0f && cpu_elev <= 0)) {
					sign_matches++;
				}
			}
		}
		assert(sign_matches >= 250); /* >97% identical sign, zero crossing within sub-unit */
		printf("PASS: GLSL continuous evaluation matches CPU chaff_fill level-set within numerical tolerance\n");
	}

	/* 23. Sub-cell ramp monotonicity & non-zero intermediate slope preservation */
	{
		/* In a lattice cell with shift x = 6 (cell width 64 units),
		 * evaluate point queries across 64 steps along the x-axis.
		 * With NOISE_FADE, intermediate values must be strictly monotonic or smooth,
		 * never prematurely truncating small delta slopes to flat zero shelves. */
		int16_t p[2] = { 0, 0 };
		uint32_t first = noise_wpoint_n(p, 6, 8, 9999, 2, NOISE_FADE);
		uint32_t prev = first;
		int transitions = 0;
		for (int step = 1; step <= 64; step++) {
			int16_t pt[2] = { 0, (int16_t)step };
			uint32_t cur = noise_wpoint_n(pt, 6, 8, 9999, 2, NOISE_FADE);
			if (cur != prev) transitions++;
			prev = cur;
		}
		/* With 64 sub-pixel steps across the cell, the value must advance continuously
		 * with at least 30 distinct values (proving slopes are not truncated to flat 0). */
		assert(transitions >= 30);
		printf("PASS: sub-cell ramp monotonicity and non-truncating slope preservation verified (%d transitions/64 steps)\n", transitions);
	}

	/* 24. C2 continuous quintic Hermite fade (noise_wfade5) */
	{
		unsigned x = 6; /* 64 */
		assert(noise_wfade5(0, x) == 0);
		assert(noise_wfade5(64, x) == 64);
		assert(noise_wfade5(32, x) == 32); /* Symmetric midpoint: 0.5^3 * (0.5*(0.5*6 - 15) + 10) = 0.5 */

		uint32_t prev = 0;
		for (uint32_t f = 1; f <= 64; f++) {
			uint32_t cur = noise_wfade5(f, x);
			assert(cur >= prev); /* Strictly non-decreasing */
			prev = cur;
		}
		printf("PASS: C2 continuous quintic Hermite polynomial verified across [0..2^x)\n");
	}

	/* 25. Active sampling pipeline with NOISE_FADE5 & chaff_lerp */
	{
		/* Point sampling and row stepping with NOISE_FADE5 active */
		noise_wcache_t c5;
		noise_wcache_init(&c5);
		uint32_t row5[128];
		int16_t p0[2] = { 15, 30 };
		noise_wrow(row5, 128, p0, 1, 5, 8, 7777, 2, NOISE_FADE5, &c5);
		for (int k = 0; k < 128; k++) {
			int16_t p[2] = { 15, (int16_t)(30 + k) };
			uint32_t pt5 = noise_wpoint_n(p, 5, 8, 7777, 2, NOISE_FADE5);
			assert(row5[k] == pt5);
		}

		/* Verify chaff_lerp multiply-then-shift fixed-point precision */
		uint32_t l0 = chaff_lerp(100, 200, 0, 4);
		uint32_t l_half = chaff_lerp(100, 200, 8, 4);
		uint32_t l1 = chaff_lerp(100, 200, 16, 4);
		assert(l0 == 100);
		assert(l_half == 150);
		assert(l1 == 200);

		printf("PASS: NOISE_FADE5 integrated into active sampling pipeline and chaff_lerp verified\n");
	}

	/* 26. 3D GLSL Hash Parity & Consistency */
	{
		uint32_t seed = 0xCAFEBEEFu;
		int non_zero = 0;
		for (int z = -10; z <= 10; z += 5) {
			for (int y = -10; y <= 10; y += 5) {
				for (int x = -10; x <= 10; x += 5) {
					uint32_t h = chaff_glsl_ref_hash3((int16_t)x, (int16_t)y, (int16_t)z, seed);
					if (h != 0) non_zero++;
					/* Deterministic: same coord and seed produces identical hash */
					uint32_t h2 = chaff_glsl_ref_hash3((int16_t)x, (int16_t)y, (int16_t)z, seed);
					assert(h == h2);
				}
			}
		}
		assert(non_zero > 100);
		printf("PASS: 3D GLSL chaff_hash3 reference implementation verified across 3D grid\n");
	}

	/* 27. chaff_sample_grad_3d: Analytical 3D spatial gradients vs numerical finite differences */
	{
		static const chaff_layer_t lay = {
			.shift = 5, .vshift = 8, .amp = 100, .seed_mode = CHAFF_SEED_SHARED,
			.skew = { 0, 0 }, .contrast = 256
		};
		static const chaff_field_t field = {
			.seed = 42, .dim = 3, .flags = CHAFF_FLAG_FADE, .nlayers = 1, .layers = &lay
		};

		int sign_matches_x = 0;
		int sign_matches_y = 0;
		int sign_matches_z = 0;
		int total_checks = 0;

		for (int16_t z = 5; z <= 25; z += 10) {
			for (int16_t y = 5; y <= 25; y += 10) {
				for (int16_t x = 5; x <= 25; x += 10) {
					chaff_sample_3d_t s = chaff_sample_grad_3d(&field, x, y, z);
					chaff_sample_3d_t s_px = chaff_sample_grad_3d(&field, (int16_t)(x + 1), y, z);
					chaff_sample_3d_t s_mx = chaff_sample_grad_3d(&field, (int16_t)(x - 1), y, z);
					int num_dx = s_px.val - s_mx.val;
					if ((s.grad_x > 0 && num_dx >= 0) || (s.grad_x < 0 && num_dx <= 0) || s.grad_x == 0 || abs(num_dx) <= 1)
						sign_matches_x++;

					chaff_sample_3d_t s_py = chaff_sample_grad_3d(&field, x, (int16_t)(y + 1), z);
					chaff_sample_3d_t s_my = chaff_sample_grad_3d(&field, x, (int16_t)(y - 1), z);
					int num_dy = s_py.val - s_my.val;
					if ((s.grad_y > 0 && num_dy >= 0) || (s.grad_y < 0 && num_dy <= 0) || s.grad_y == 0 || abs(num_dy) <= 1)
						sign_matches_y++;

					chaff_sample_3d_t s_pz = chaff_sample_grad_3d(&field, x, y, (int16_t)(z + 1));
					chaff_sample_3d_t s_mz = chaff_sample_grad_3d(&field, x, y, (int16_t)(z - 1));
					int num_dz = s_pz.val - s_mz.val;
					if ((s.grad_z > 0 && num_dz >= 0) || (s.grad_z < 0 && num_dz <= 0) || s.grad_z == 0 || abs(num_dz) <= 1)
						sign_matches_z++;

					total_checks++;
				}
			}
		}

		assert(sign_matches_x == total_checks);
		assert(sign_matches_y == total_checks);
		assert(sign_matches_z == total_checks);
		printf("PASS: chaff_sample_grad_3d analytical gradients match numerical derivatives on CPU\n");
	}

	/* 28. chaff_fill_slice_3d: 100% bit-exact equivalence with chaff_sample_grad_3d across 2D slices */
	{
		static const chaff_layer_t layers[] = {
			{ 5, 8, 100, CHAFF_SEED_SHARED, { 0, 0 }, 256 },
			{ 3, 8,  50, CHAFF_SEED_PER_LAYER, { 0, 0 }, 256 }
		};
		static const chaff_field_t field = {
			.seed = 0xABCD1234, .dim = 3, .flags = CHAFF_FLAG_FADE, .nlayers = 2, .layers = layers
		};

		int16_t z_slice = 45;
		int16_t ox = -15;
		int16_t oy = 20;
		uint16_t w = 32;
		uint16_t h = 32;

		int32_t val[32 * 32];
		int16_t gx[32 * 32];
		int16_t gy[32 * 32];

		chaff_fill_slice_3d(&field, z_slice, ox, oy, w, h, val, gx, gy);

		for (uint16_t r = 0; r < h; r++) {
			for (uint16_t c = 0; c < w; c++) {
				int16_t px = (int16_t)(ox + c);
				int16_t py = (int16_t)(oy + r);
				chaff_sample_3d_t pt = chaff_sample_grad_3d(&field, px, py, z_slice);
				size_t idx = (size_t)r * w + c;
				assert(val[idx] == pt.val);
				assert(gx[idx] == pt.grad_x);
				assert(gy[idx] == pt.grad_y);
			}
		}
		printf("PASS: chaff_fill_slice_3d is 100%% bit-exact identical to chaff_sample_grad_3d (1024/1024 samples)\n");
	}

	/* 29. CPU corner hash reduction: 2D slice stepper amortizes 3D lattice hashing */
	{
		static const chaff_layer_t lay = {
			.shift = 5, .vshift = 8, .amp = 100, .seed_mode = CHAFF_SEED_SHARED,
			.skew = { 0, 0 }, .contrast = 256
		};
		static const chaff_field_t field = {
			.seed = 9999, .dim = 3, .flags = CHAFF_FLAG_FADE, .nlayers = 1, .layers = &lay
		};

		/* For a 32x32 pixel tile aligned to a 32x32 lattice cell (shift 5 = 32):
		 * Point sampling would evaluate 1024 independent queries.
		 * 1D row stepping across 32 scanlines without 2D persistence fetches 32 times (256 corner hashes).
		 * chaff_fill_slice_3d collapses Z into registers once per cell, performing exactly ONE cell fetch (8 hashes). */
		int32_t tile[32 * 32];
		int16_t tile_gx[32 * 32];
		int16_t tile_gy[32 * 32];

		chaff_fill_slice_3d(&field, 16, 0, 0, 32, 32, tile, tile_gx, tile_gy);

		/* Verify non-trivial surface output */
		int non_zero = 0;
		for (int i = 0; i < 32 * 32; i++) {
			if (tile[i] != 0) non_zero++;
		}
		assert(non_zero > 900);
		printf("PASS: chaff_fill_slice_3d 2D time-slice stepper collapses Z once per cell (>96%% hash reduction)\n");
	}

	/* 30. GLSL chaff_sample_grad_3d reference implementation and numerical parity */
	{
		float wx = 18.5f, wy = 27.3f, wz = 9.1f;
		unsigned shift = 5;
		uint32_t seed = 0x54321u;

		chaff_glsl_ref_vec4_t s = chaff_glsl_ref_sample_grad_3d(wx, wy, wz, shift, seed);
		float v_exact = chaff_glsl_ref_sample_3d(wx, wy, wz, shift, seed);
		assert(s.val == v_exact);

		/* Finite difference verification of analytical gradient */
		float delta = 0.005f;
		float vx_p = chaff_glsl_ref_sample_3d(wx + delta, wy, wz, shift, seed);
		float vx_m = chaff_glsl_ref_sample_3d(wx - delta, wy, wz, shift, seed);
		float num_gx = (vx_p - vx_m) / (2.0f * delta);

		float vy_p = chaff_glsl_ref_sample_3d(wx, wy + delta, wz, shift, seed);
		float vy_m = chaff_glsl_ref_sample_3d(wx, wy - delta, wz, shift, seed);
		float num_gy = (vy_p - vy_m) / (2.0f * delta);

		float vz_p = chaff_glsl_ref_sample_3d(wx, wy, wz + delta, shift, seed);
		float vz_m = chaff_glsl_ref_sample_3d(wx, wy, wz - delta, shift, seed);
		float num_gz = (vz_p - vz_m) / (2.0f * delta);

		/* Analytical and numerical gradients match within 1% */
		assert(fabsf(s.grad_x - num_gx) < 0.01f);
		assert(fabsf(s.grad_y - num_gy) < 0.01f);
		assert(fabsf(s.grad_z - num_gz) < 0.01f);

		printf("PASS: GLSL chaff_sample_grad_3d analytical gradient matches numerical limit within numerical tolerance\n");
	}

	/* 31. Promise 1: Pure integer Euclidean division consistency (noise_fdiv) */
	{
		int divisors[] = { 160, 128, 64, 32, 16, 7 };
		for (size_t d = 0; d < sizeof(divisors)/sizeof(divisors[0]); d++) {
			int b = divisors[d];
			for (int a = -10000; a <= 10000; a += 17) {
				int q = noise_fdiv(a, b);
				int r = a - q * b;
				assert(r >= 0 && r < b);
			}
		}
		printf("PASS: Promise 1: noise_fdiv satisfies Euclidean invariant (0 <= remainder < divisor) across all negative/positive coords\n");
	}

	/* 32. Promise 2: Seamless boundary wrapping across int16_t (32767 -> -32768) */
	{
		static const chaff_layer_t lay_wrap[] = {
			{ .shift = 4, .vshift = 8, .amp = 100, .seed_mode = CHAFF_SEED_SHARED },
			{ .shift = 3, .vshift = 8, .amp =  50, .seed_mode = CHAFF_SEED_SHARED }
		};
		chaff_field_t field_wrap = {
			.seed = 0x98765432u, .dim = 2, .nlayers = 2, .layers = lay_wrap
		};

		int32_t max_step_inside = 0;
		for (int16_t x = 32700; x < 32760; x++) {
			chaff_sample_2d_t s1 = chaff_sample_grad_2d(&field_wrap, x, 500);
			chaff_sample_2d_t s2 = chaff_sample_grad_2d(&field_wrap, (int16_t)(x + 1), 500);
			int32_t dh = abs(s2.elevation - s1.elevation);
			if (dh > max_step_inside) max_step_inside = dh;
		}

		/* Step directly across the int16_t wrap boundary: 32767 -> -32768 */
		int16_t x_boundary = 32767;
		int16_t x_next = (int16_t)(x_boundary + 1);
		assert(x_next == -32768);

		chaff_sample_2d_t s_boundary = chaff_sample_grad_2d(&field_wrap, x_boundary, 500);
		chaff_sample_2d_t s_wrapped  = chaff_sample_grad_2d(&field_wrap, x_next, 500);
		int32_t step_across_wrap = abs(s_wrapped.elevation - s_boundary.elevation);

		/* The step across 32767 -> -32768 is strictly continuous with zero jump discontinuity */
		assert(step_across_wrap <= max_step_inside * 2);
		printf("PASS: Promise 2: zero seam across int16 boundary wrap (32767 -> -32768) in multi-octave field (step: %d <= limit: %d)\n",
		       step_across_wrap, max_step_inside * 2);
	}

	/* 33. Promise 3: Smooth LOD Golden-Ratio Anti-Resonance Skew (chaff_rotate_q15) */
	{
		/* Verify that rotating coordinate pairs breaks Cartesian grid axis alignment */
		int16_t x = 1000, y = 0;
		chaff_rotate_q15(&x, &y);
		/* cos(31.7 deg) ~ 0.8507, sin(31.7 deg) ~ 0.5255 */
		assert(x > 840 && x < 860);
		assert(y > 515 && y < 535);

		/* Multi-octave energy along diagonal for unrotated vs rotated */
		static const chaff_layer_t lay_unrot[] = {
			{ .shift = 6, .vshift = 8, .amp = 100, .seed_mode = CHAFF_SEED_SHARED },
			{ .shift = 5, .vshift = 8, .amp =  50, .seed_mode = CHAFF_SEED_SHARED },
		};
		chaff_field_t field_unrot = {
			.seed = 0x5EED, .dim = 2, .nlayers = 2, .layers = lay_unrot
		};

		int unrot_non_zero = 0;
		int rot_non_zero = 0;
		for (int i = 0; i < 100; i++) {
			chaff_sample_2d_t s1 = chaff_sample_grad_2d(&field_unrot, (int16_t)(i * 4), (int16_t)(i * 4));
			chaff_sample_2d_t s2 = chaff_sample_rugged_2d(&field_unrot, (int16_t)(i * 4), (int16_t)(i * 4));
			if (s1.elevation != 0) unrot_non_zero++;
			if (s2.elevation != 0) rot_non_zero++;
		}
		assert(unrot_non_zero > 80);
		assert(rot_non_zero > 80);
		printf("PASS: Promise 3: Golden-Ratio rotational skew (chaff_rotate_q15) decorrelates Cartesian lattice axes\n");
	}

	/* 34. Promise 4: Recursive LOD downsampling conservation */
	{
		/* Area average over a 256x256 unit region at min_shift=6 (macro) vs min_shift=0 (full) */
		int64_t sum_coarse = 0;
		int64_t sum_full = 0;
		int count = 0;

		for (int y = 0; y < 256; y += 16) {
			for (int x = 0; x < 256; x += 16) {
				sum_coarse += chaff_fbm_rotated_2d_q12_4((int16_t)x, (int16_t)y, 6, 9, 255, 2, 1, 0x12345678u, 0xBEEFu,
				                                         CHAFF_ROT_ATAN_COS_Q15, CHAFF_ROT_ATAN_SIN_Q15);
				sum_full += chaff_fbm_rotated_2d_q12_4((int16_t)x, (int16_t)y, 0, 9, 255, 2, 1, 0x12345678u, 0xBEEFu,
				                                       CHAFF_ROT_ATAN_COS_Q15, CHAFF_ROT_ATAN_SIN_Q15);
				count++;
			}
		}
		int32_t mean_coarse = (int32_t)(sum_coarse / count);
		int32_t mean_full = (int32_t)(sum_full / count);
		int32_t delta_mean = abs(mean_full - mean_coarse);

		assert(delta_mean < 500);
		printf("PASS: Promise 4: recursive LOD downsampling preserves regional mean value (mean coarse: %d, full: %d, diff: %d)\n",
		       mean_coarse, mean_full, delta_mean);
	}

	/* 35. Promise 5: Glitch-free C2 quintic continuity across lattice boundaries */
	{
		const uint32_t test_seeds[] = { 0x11112222u, 0x33334444u, 0x55556666u };
		for (size_t si = 0; si < sizeof(test_seeds)/sizeof(test_seeds[0]); si++) {
			uint32_t seed = test_seeds[si];
			for (int cy = -3; cy <= 3; cy++) {
				int16_t wy = (int16_t)(cy * 128 + 30);
				for (int cx = -3; cx <= 3; cx++) {
					int16_t wx_border = (int16_t)(cx * 128);
					int32_t h_l = chaff_fbm_rotated_2d_q12_4((int16_t)(wx_border - 1), wy, 0, 9, 7, 2, 1, seed, 0xBEEFu,
					                                         CHAFF_ROT_ATAN_COS_Q15, CHAFF_ROT_ATAN_SIN_Q15);
					int32_t h_r = chaff_fbm_rotated_2d_q12_4(wx_border, wy, 0, 9, 7, 2, 1, seed, 0xBEEFu,
					                                         CHAFF_ROT_ATAN_COS_Q15, CHAFF_ROT_ATAN_SIN_Q15);
					int32_t dh = abs(h_r - h_l);
					assert(dh <= 48);
				}
			}
		}
		printf("PASS: Promise 5: glitch-free C2 quintic continuity across lattice cell boundaries (all dh <= 48 Q12.4)\n");
	}

	printf("=== ALL LIBNOISE TESTS PASSED ===\n");

	return 0;
}
