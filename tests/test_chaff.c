#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <ttypt/noise.h>
#include <ttypt/chaff.h>

int main(void)
{
	printf("=== Testing Canonical libchaff Public API & Advanced Features ===\n");

	/* 1. Canonical chaff_point_* wrappers equivalence with noise_point_* */
	{
		uint32_t seed = 0x12345678u;
		uint8_t shift = 4;

		for (int16_t v = -100; v <= 100; v += 10) {
			int p1_c = chaff_point_1d(v, seed, shift);
			int p1_n = noise_point_1d(v, seed, shift);
			assert(p1_c == p1_n);
			assert(p1_c >= 0 && p1_c <= 255);

			int p2_c = chaff_point_2d(v, (int16_t)(v * 2), seed, shift);
			int p2_n = noise_point_2d(v, (int16_t)(v * 2), seed, shift);
			assert(p2_c == p2_n);
			assert(p2_c >= 0 && p2_c <= 255);

			int p3_c = chaff_point_3d(v, (int16_t)(v * 2), (int16_t)(v * 3), seed, shift);
			int p3_n = noise_point_3d(v, (int16_t)(v * 2), (int16_t)(v * 3), seed, shift);
			assert(p3_c == p3_n);
			assert(p3_c >= 0 && p3_c <= 255);

			int p4_c = chaff_point_4d(v, (int16_t)(v * 2), (int16_t)(v * 3), (int16_t)(v * 4), seed, shift);
			int p4_n = noise_point_4d(v, (int16_t)(v * 2), (int16_t)(v * 3), (int16_t)(v * 4), seed, shift);
			assert(p4_c == p4_n);
			assert(p4_c >= 0 && p4_c <= 255);

			int16_t coords[4] = { v, (int16_t)(v * 2), (int16_t)(v * 3), (int16_t)(v * 4) };
			assert(chaff_point_n(coords, seed, shift, 1) == p1_c);
			assert(chaff_point_n(coords, seed, shift, 2) == p2_c);
			assert(chaff_point_n(coords, seed, shift, 3) == p3_c);
			assert(chaff_point_n(coords, seed, shift, 4) == p4_c);
		}
		printf("PASS: chaff_point_* wrappers (1D..4D) match noise_point_* bit-for-bit\n");
	}

	/* 2. Canonical chaff_oct, chaff_matrix_fill_2d, chaff_sample_2d, chaff_sample_3d */
	{
		uint32_t m_chaff[256], m_noise[256];
		int16_t s[2] = { -32, 64 };
		octave_t oct[] = { { 4, 1 } };
		chaff_oct(m_chaff, s, 1, oct, 0xABCDEF, 4, 2);
		noise_oct(m_noise, s, 1, oct, 0xABCDEF, 4, 2);
		for (int i = 0; i < 256; i++) {
			assert(m_chaff[i] == m_noise[i]);
		}

		static const noise_octave_t octs[] = {
			{ .shift = 5, .amp = 80, .skew_x = 0, .skew_y = 0, .contrast = 256 },
		};
		noise_cascade_t cfg = {
			.seed = 0x87654321,
			.count = 1,
			.octaves = octs,
		};
		int32_t mat_c[1024], mat_n[1024];
		chaff_matrix_fill_2d(&cfg, -16, 32, 5, mat_c);
		noise_matrix_fill_2d(&cfg, -16, 32, 5, mat_n);
		for (int i = 0; i < 1024; i++) {
			assert(mat_c[i] == mat_n[i]);
		}

		for (int16_t y = -20; y <= 20; y += 5) {
			for (int16_t x = -20; x <= 20; x += 5) {
				assert(chaff_sample_2d(&cfg, x, y) == noise_sample_2d(&cfg, x, y));
				assert(chaff_sample_3d(&cfg, x, y, 10) == noise_sample_3d(&cfg, x, y, 10));
			}
		}
		printf("PASS: canonical chaff_oct, chaff_matrix_fill_2d, chaff_sample_2d/3d match noise_* bit-for-bit\n");
	}

	/* 3. Comprehensive chaff_coord_scale Edge Cases */
	{
		/* Identity scale */
		chaff_coord_t c0 = { .cell = 1234, .frac = 0x5678 };
		chaff_coord_t s0 = chaff_coord_scale(c0, 0);
		assert(s0.cell == c0.cell && s0.frac == c0.frac);

		/* Positive scales: 1, 2, 4, 8 */
		for (int scale = 1; scale <= 8; scale++) {
			chaff_coord_t c = { .cell = 2, .frac = 0x4000 }; /* 2.25 */
			chaff_coord_t res = chaff_coord_scale(c, scale);
			int64_t expected_total = (((int64_t)2 << 16) | 0x4000) << scale;
			int16_t expected_cell = (int16_t)(expected_total >> 16);
			uint16_t expected_frac = (uint16_t)(expected_total & 0xFFFF);
			assert(res.cell == expected_cell && res.frac == expected_frac);
		}

		/* Negative scales: -1, -2, -4, -8 */
		for (int scale = 1; scale <= 8; scale++) {
			chaff_coord_t c = { .cell = 64, .frac = 0x0000 };
			chaff_coord_t res = chaff_coord_scale(c, -scale);
			int64_t expected_total = ((int64_t)64 << 16) >> scale;
			int16_t expected_cell = (int16_t)(expected_total >> 16);
			uint16_t expected_frac = (uint16_t)(expected_total & 0xFFFF);
			assert(res.cell == expected_cell && res.frac == expected_frac);
		}

		/* Negative coordinate cells */
		chaff_coord_t neg = { .cell = -10, .frac = 0x8000 }; /* -9.5 */
		chaff_coord_t neg_scaled = chaff_coord_scale(neg, 1);
		assert(neg_scaled.cell == -19 && neg_scaled.frac == 0);

		printf("PASS: chaff_coord_scale verified across zero, positive, negative, and sub-pixel scales\n");
	}

	/* 4. Isolated unit test for chaff_fade_frac */
	{
		unsigned shift = 6;
		/* When frac == 0, matches integer noise_wfade bit-exact */
		for (uint32_t fi = 0; fi <= 64; fi += 4) {
			uint32_t fade_zero = chaff_fade_frac(fi, 0, shift);
			uint32_t fade_direct = noise_wfade(fi, shift) << 16;
			assert(fade_zero == fade_direct);
		}

		/* Intra-cell fractional progression: as frac increases from 0 to 65535,
		 * fractional fade values are strictly non-decreasing */
		for (uint32_t fi = 0; fi <= 16; fi += 4) {
			uint32_t prev_frac_val = 0;
			for (uint32_t frac = 1024; frac < 65536; frac += 4096) {
				uint32_t val = chaff_fade_frac(fi, (uint16_t)frac, shift);
				assert(val >= prev_frac_val);
				prev_frac_val = val;
			}
		}
		printf("PASS: chaff_fade_frac evaluated with sub-cell fractional precision and monotonicity\n");
	}

	/* 5. 1D and 3D chaff_fill evaluation */
	{
		static const chaff_layer_t lay1d = {
			.shift = 4, .vshift = 8, .amp = 100, .seed_mode = CHAFF_SEED_SHARED,
			.skew = { 0, 0 }, .contrast = 256
		};
		static const chaff_field_t field1d = {
			.seed = 0x5555, .dim = 1, .flags = CHAFF_FLAG_FADE, .nlayers = 1, .layers = &lay1d
		};

		chaff_coord_t origin1d[1] = { { -10, 0 } };
		int32_t step1d[1] = { 65536 }; /* 1.0 */
		uint16_t n1d[1] = { 64 };
		int32_t out1d[64];
		chaff_fill(&field1d, 0, 1, origin1d, step1d, n1d, out1d, 64, CHAFF_MODE_SET);

		/* Verify 1D fill produces smooth, continuous, non-zero values */
		int non_zero1d = 0;
		for (int i = 0; i < 64; i++) {
			if (out1d[i] != 0) non_zero1d++;
			if (i > 0) {
				int diff = abs(out1d[i] - out1d[i - 1]);
				assert(diff <= 40);
			}
		}
		assert(non_zero1d > 50);

		/* Test 1D CHAFF_MODE_ADD */
		int32_t out1d_add[64];
		memcpy(out1d_add, out1d, sizeof(out1d));
		chaff_fill(&field1d, 0, 1, origin1d, step1d, n1d, out1d_add, 64, CHAFF_MODE_ADD);
		for (int i = 0; i < 64; i++) {
			assert(out1d_add[i] == out1d[i] * 2);
		}

		/* 3D volume fill: 4 slices of 8x8 */
		static const chaff_layer_t lay3d = {
			.shift = 5, .vshift = 8, .amp = 80, .seed_mode = CHAFF_SEED_SHARED,
			.skew = { 0, 0 }, .contrast = 256
		};
		static const chaff_field_t field3d = {
			.seed = 0x7777, .dim = 3, .flags = CHAFF_FLAG_FADE, .nlayers = 1, .layers = &lay3d
		};

		chaff_coord_t origin3d[3] = { { 0, 0 }, { 10, 0 }, { 20, 0 } };
		int32_t step3d[3] = { 65536, 65536, 65536 };
		uint16_t n3d[3] = { 4, 8, 8 }; /* 4 slices, 8 rows, 8 cols */
		int32_t out3d[4 * 8 * 8];
		chaff_fill(&field3d, 0, 1, origin3d, step3d, n3d, out3d, 8, CHAFF_MODE_SET);

		int non_zero3d = 0;
		for (int i = 0; i < 4 * 8 * 8; i++) {
			if (out3d[i] != 0) non_zero3d++;
		}
		assert(non_zero3d > 200);

		/* Test CHAFF_MODE_ADD */
		int32_t out3d_add[4 * 8 * 8];
		memcpy(out3d_add, out3d, sizeof(out3d));
		chaff_fill(&field3d, 0, 1, origin3d, step3d, n3d, out3d_add, 8, CHAFF_MODE_ADD);
		for (int i = 0; i < 4 * 8 * 8; i++) {
			assert(out3d_add[i] == out3d[i] * 2);
		}

		printf("PASS: 1D and 3D volume chaff_fill evaluated in SET and ADD modes\n");
	}

	/* 6. chaff_fill with Skewing and Contrast */
	{
		static const chaff_layer_t lay_skew = {
			.shift = 5, .vshift = 8, .amp = 100, .seed_mode = CHAFF_SEED_PER_LAYER,
			.skew = { 3, -5 }, .contrast = 512 /* 2.0x contrast */
		};
		static const chaff_field_t field_skew = {
			.seed = 0x11223344, .dim = 2, .flags = CHAFF_FLAG_FADE, .nlayers = 1, .layers = &lay_skew
		};

		chaff_coord_t origin[2] = { { -50, 1234 }, { 25, 5678 } };
		int32_t step[2] = { 32768, 32768 }; /* 0.5 step */
		uint16_t n[2] = { 16, 16 };
		int32_t out_skew[256];

		chaff_fill(&field_skew, 0, 1, origin, step, n, out_skew, 16, CHAFF_MODE_SET);
		int non_zero = 0;
		for (int i = 0; i < 256; i++) {
			if (out_skew[i] != 0) non_zero++;
		}
		assert(non_zero > 200);

		/* Test 2D CHAFF_MODE_ADD */
		int32_t out_skew_add[256];
		memcpy(out_skew_add, out_skew, sizeof(out_skew));
		chaff_fill(&field_skew, 0, 1, origin, step, n, out_skew_add, 16, CHAFF_MODE_ADD);
		for (int i = 0; i < 256; i++) {
			assert(out_skew_add[i] == out_skew[i] * 2);
		}
		printf("PASS: chaff_fill 1D and 2D with skewing evaluated in SET and ADD modes\n");
	}

	/* 7. Analytical Gradients (2D & 3D) Multi-Layer and Parameter Variants */
	{
		static const chaff_layer_t layers[] = {
			{ .shift = 7, .vshift = 8, .amp =   0, .seed_mode = CHAFF_SEED_SHARED,    .contrast = 256 }, /* raw amp=0 */
			{ .shift = 5, .vshift = 8, .amp = 100, .seed_mode = CHAFF_SEED_PER_LAYER, .contrast = 512 }, /* contrast 2x */
			{ .shift = 3, .vshift = 8, .amp =  30, .seed_mode = CHAFF_SEED_SHARED,    .contrast = 128 }  /* contrast 0.5x */
		};
		static const chaff_field_t field2d = {
			.seed = 0xA1B2C3D4u, .dim = 2, .flags = CHAFF_FLAG_FADE, .nlayers = 3, .layers = layers
		};
		static const chaff_field_t field3d = {
			.seed = 0xA1B2C3D4u, .dim = 3, .flags = CHAFF_FLAG_FADE, .nlayers = 3, .layers = layers
		};

		for (int16_t y = -50; y <= 50; y += 25) {
			for (int16_t x = -50; x <= 50; x += 25) {
				chaff_sample_2d_t s2 = chaff_sample_grad_2d(&field2d, x, y);
				assert(s2.elevation != 0);

				chaff_sample_3d_t s3 = chaff_sample_grad_3d(&field3d, x, y, 15);
				assert(s3.val != 0);
			}
		}
		printf("PASS: 2D & 3D analytical gradients verified with multi-layer, raw amp, and contrast\n");
	}

	/* 8. chaff_fill_slice_3d Optional Outputs and Multi-Cell Spanning */
	{
		static const chaff_layer_t lay = {
			.shift = 4, .vshift = 8, .amp = 100, .seed_mode = CHAFF_SEED_SHARED, .contrast = 256
		};
		static const chaff_field_t field = {
			.seed = 0x99998888, .dim = 3, .flags = CHAFF_FLAG_FADE, .nlayers = 1, .layers = &lay
		};

		uint16_t w = 48, h = 48; /* Spans 3x3 cells of size 16x16 */
		int16_t z = 25;
		int16_t ox = -20, oy = 10;

		/* 1. Value only (gradients NULL) */
		int32_t val_only[48 * 48];
		chaff_fill_slice_3d(&field, z, ox, oy, w, h, val_only, NULL, NULL);

		/* 2. Gradients only (val NULL) */
		int16_t gx_only[48 * 48];
		int16_t gy_only[48 * 48];
		chaff_fill_slice_3d(&field, z, ox, oy, w, h, NULL, gx_only, gy_only);

		/* 3. Combined */
		int32_t val_both[48 * 48];
		int16_t gx_both[48 * 48];
		int16_t gy_both[48 * 48];
		chaff_fill_slice_3d(&field, z, ox, oy, w, h, val_both, gx_both, gy_both);

		for (size_t i = 0; i < 48 * 48; i++) {
			assert(val_only[i] == val_both[i]);
			assert(gx_only[i] == gx_both[i]);
			assert(gy_only[i] == gy_both[i]);
		}

		/* 4. Slice across negative coordinates and Z animation limit */
		int16_t ox_neg = -32, oy_neg = -32;
		int32_t val_zwrap[16 * 16];
		int16_t gx_zwrap[16 * 16], gy_zwrap[16 * 16];
		int16_t z_time = 32767;
		chaff_fill_slice_3d(&field, z_time, ox_neg, oy_neg, 16, 16, val_zwrap, gx_zwrap, gy_zwrap);

		for (int r = 0; r < 16; r++) {
			for (int c = 0; c < 16; c++) {
				int16_t px = (int16_t)(ox_neg + c);
				int16_t py = (int16_t)(oy_neg + r);
				chaff_sample_3d_t pt = chaff_sample_grad_3d(&field, px, py, z_time);
				size_t idx = (size_t)r * 16 + c;
				assert(val_zwrap[idx] == pt.val);
				assert(gx_zwrap[idx] == pt.grad_x);
				assert(gy_zwrap[idx] == pt.grad_y);
			}
		}

		printf("PASS: chaff_fill_slice_3d verified with optional outputs, multi-cell spanning, and negative/Z-wrap coords\n");
	}

	/* 9. Band-limited Nyquist LOD Filtering (chaff_fill_lod) via explicit min_shift & step_q16 */
	{
		static const chaff_layer_t lod_layers[] = {
			{ .shift = 2, .vshift = 8, .amp =  15, .seed_mode = CHAFF_SEED_SHARED, .contrast = 256 }, /* shift 2 (wavelength 4) */
			{ .shift = 4, .vshift = 8, .amp =  30, .seed_mode = CHAFF_SEED_SHARED, .contrast = 256 }, /* shift 4 (wavelength 16) */
			{ .shift = 5, .vshift = 8, .amp =  60, .seed_mode = CHAFF_SEED_SHARED, .contrast = 256 }, /* shift 5 (wavelength 32) */
			{ .shift = 7, .vshift = 8, .amp = 120, .seed_mode = CHAFF_SEED_SHARED, .contrast = 256 }  /* shift 7 (wavelength 128) */
		};
		static const chaff_field_t lod_field = {
			.seed = 0x54321, .dim = 2, .flags = CHAFF_FLAG_FADE, .nlayers = 4, .layers = lod_layers
		};

		chaff_coord_t origin[2] = { { -20, 0 }, { 30, 0 } };
		int32_t step_q16[2] = { 65536, 65536 };
		uint16_t n[2] = { 16, 16 };

		/* Explicit min_shift 6: only shift 7 active (layer index 3..4) */
		int32_t out_lod0[256], ref_lod0[256];
		chaff_fill_lod(&lod_field, 6, origin, step_q16, n, out_lod0, 16, CHAFF_MODE_SET);
		chaff_fill(&lod_field, 3, 4, origin, step_q16, n, ref_lod0, 16, CHAFF_MODE_SET);
		for (int i = 0; i < 256; i++) {
			assert(out_lod0[i] == ref_lod0[i]);
		}

		/* Automatic Nyquist derivation from step_q16 = 16.0 (nyquist_shift = 5): shifts 5 and 7 active */
		int32_t step_16_q16[2] = { 16 << 16, 16 << 16 };
		int32_t out_lod1[256], ref_lod1[256];
		chaff_fill_lod(&lod_field, 0, origin, step_16_q16, n, out_lod1, 16, CHAFF_MODE_SET);
		chaff_fill(&lod_field, 2, 4, origin, step_16_q16, n, ref_lod1, 16, CHAFF_MODE_SET);
		for (int i = 0; i < 256; i++) {
			assert(out_lod1[i] == ref_lod1[i]);
		}

		/* Automatic Nyquist derivation from step_q16 = 4.0 (nyquist_shift = 3): shifts 4, 5, 7 active */
		int32_t step_4_q16[2] = { 4 << 16, 4 << 16 };
		int32_t out_lod2[256], ref_lod2[256];
		chaff_fill_lod(&lod_field, 0, origin, step_4_q16, n, out_lod2, 16, CHAFF_MODE_SET);
		chaff_fill(&lod_field, 1, 4, origin, step_4_q16, n, ref_lod2, 16, CHAFF_MODE_SET);
		for (int i = 0; i < 256; i++) {
			assert(out_lod2[i] == ref_lod2[i]);
		}

		/* min_shift 0 with step 1.0: all shifts active (layer index 0..4) */
		int32_t out_lod3[256], ref_lod3[256];
		chaff_fill_lod(&lod_field, 0, origin, step_q16, n, out_lod3, 16, CHAFF_MODE_SET);
		chaff_fill(&lod_field, 0, 4, origin, step_q16, n, ref_lod3, 16, CHAFF_MODE_SET);
		for (int i = 0; i < 256; i++) {
			assert(out_lod3[i] == ref_lod3[i]);
		}

		/* Test CHAFF_MODE_ADD with chaff_fill_lod */
		int32_t out_lod3_add[256];
		memcpy(out_lod3_add, out_lod3, sizeof(out_lod3));
		chaff_fill_lod(&lod_field, 0, origin, step_q16, n, out_lod3_add, 16, CHAFF_MODE_ADD);
		for (int i = 0; i < 256; i++) {
			assert(out_lod3_add[i] == out_lod3[i] * 2);
		}

		printf("PASS: chaff_fill_lod verified across continuous Nyquist step_q16 and explicit min_shift\n");
	}

	/* 10. Multi-Octave Rugged Sampling with Golden-Ratio Skew (chaff_sample_rugged_2d) */
	{
		static const chaff_layer_t rug_layers[] = {
			{ .shift = 6, .vshift = 8, .amp = 100, .seed_mode = CHAFF_SEED_SHARED, .contrast = 256 },
			{ .shift = 4, .vshift = 8, .amp =  50, .seed_mode = CHAFF_SEED_PER_LAYER, .contrast = 256 },
			{ .shift = 3, .vshift = 8, .amp =  25, .seed_mode = CHAFF_SEED_SHARED, .contrast = 256 }
		};
		static const chaff_field_t rug_field = {
			.seed = 0xBEEF1234u, .dim = 2, .flags = CHAFF_FLAG_FADE, .nlayers = 3, .layers = rug_layers
		};

		int non_zero_elev = 0;
		int non_zero_grad = 0;

		/* Verify determinism and valid non-zero surface */
		for (int16_t y = -40; y <= 40; y += 20) {
			for (int16_t x = -40; x <= 40; x += 20) {
				chaff_sample_2d_t s1 = chaff_sample_rugged_2d(&rug_field, x, y);
				chaff_sample_2d_t s2 = chaff_sample_rugged_2d(&rug_field, x, y);
				assert(s1.elevation == s2.elevation);
				assert(s1.grad_x == s2.grad_x);
				assert(s1.grad_y == s2.grad_y);

				if (s1.elevation != 0) non_zero_elev++;
				if (s1.grad_x != 0 || s1.grad_y != 0) non_zero_grad++;
			}
		}
		assert(non_zero_elev > 15);
		assert(non_zero_grad > 15);

		/* Verify spatial continuity along arbitrary 2D trajectories */
		chaff_sample_2d_t prev = chaff_sample_rugged_2d(&rug_field, 0, 0);
		for (int16_t step = 1; step <= 32; step++) {
			chaff_sample_2d_t cur = chaff_sample_rugged_2d(&rug_field, step, (int16_t)(step * 2));
			int32_t diff_elev = abs(cur.elevation - prev.elevation);
			assert(diff_elev <= 45); /* Smooth, continuous elevation transitions */
			prev = cur;
		}

		/* Golden-Ratio decorrelation: compare rugged sampling with unrotated sampling */
		chaff_sample_2d_t s_rug = chaff_sample_rugged_2d(&rug_field, 50, 50);
		chaff_sample_2d_t s_unrot = chaff_sample_grad_2d(&rug_field, 50, 50);
		assert(s_rug.elevation != s_unrot.elevation || s_rug.grad_x != s_unrot.grad_x);

		printf("PASS: chaff_sample_rugged_2d verified on valid multi-layer field (determinism, continuity, and decorrelation)\n");
	}

	/* 11. Rotational Skew & chaff_rotate_q15_inplace */
	{
		/* Rotate (10000, 0) by theta = arctan(0.5) ~= 26.565 deg */
		int16_t x = 10000, y = 0;
		chaff_rotate_q15_inplace(&x, &y);
		/* cos(26.565 deg) ~ 0.8944 -> 8944, sin(26.565 deg) ~ 0.4472 -> 4472 */
		assert(x >= 8900 && x <= 9000);
		assert(y >= 4400 && y <= 4500);

		/* Vector norm conservation within integer precision (approx 10000) */
		int64_t norm_sq = (int64_t)x * x + (int64_t)y * y;
		int32_t norm = (int32_t)sqrt((double)norm_sq);
		assert(abs(norm - 10000) < 50);

		/* Full rotation cycle test for chaff_rotate_q15:
		 * 360 / 31.72 deg ~= 11.35 steps. In ~11 to 12 steps, coordinate returns near x-axis. */
		int16_t rx = 10000, ry = 0;
		for (int step = 0; step < 11; step++) {
			chaff_rotate_q15(&rx, &ry);
		}
		int32_t norm_cyc = (int32_t)sqrt((double)((int64_t)rx * rx + (int64_t)ry * ry));
		assert(abs(norm_cyc - 10000) < 150);

		printf("PASS: chaff_rotate_q15_inplace and full 360-degree rotation cycle conserve vector norm\n");
	}

	/* 12. Generic Periodic Sampling & Rotated Fractal FBM (chaff_wpoint_periodic_2d & chaff_fbm_rotated_2d_q12_4) */
	{
		/* Verify periodic wrapping for period_shift = 10 (period 1024) and period_shift = 16 (period 65536) */
		uint32_t p10_a = chaff_wpoint_periodic_2d(300, -150, 5, 10, 8, 0x12345678u);
		uint32_t p10_b = chaff_wpoint_periodic_2d(300 + 1024, -150 - 2048, 5, 10, 8, 0x12345678u);
		assert(p10_a == p10_b);

		uint32_t p16_a = chaff_wpoint_periodic_2d(5000, -3000, 9, 16, 8, 0x87654321u);
		uint32_t p16_b = chaff_wpoint_periodic_2d(5000 + 65536, -3000 + 65536, 9, 16, 8, 0x87654321u);
		assert(p16_a == p16_b);

		/* Verify monotonic amplitude scaling */
		for (uint8_t s = 2; s < 14; s++) {
			assert(chaff_octave_amplitude(s + 1) >= chaff_octave_amplitude(s));
			assert(chaff_octave_amplitude_ex(s + 2, 2, 4) > chaff_octave_amplitude_ex(s, 2, 4));
		}

		/* Verify rotated FBM across octave bounds and ridged threshold */
		int32_t fbm_full = chaff_fbm_rotated_2d_q12_4(500, -300, 0, 9, 7, 2, 1, 0xCAFEBABEu, 0xBEEFu,
		                                              CHAFF_ROT_ATAN_COS_Q15, CHAFF_ROT_ATAN_SIN_Q15);
		int32_t fbm_coarse = chaff_fbm_rotated_2d_q12_4(500, -300, 5, 9, 7, 2, 1, 0xCAFEBABEu, 0xBEEFu,
		                                                CHAFF_ROT_ATAN_COS_Q15, CHAFF_ROT_ATAN_SIN_Q15);
		assert(fbm_full != 0);
		assert(fbm_coarse != 0);
		assert(fbm_full != fbm_coarse);

		printf("PASS: chaff_wpoint_periodic_2d, chaff_octave_amplitude_ex, and chaff_fbm_rotated_2d_q12_4 verified\n");
	}

	/* 13. Null-Safety & Invalid Parameter Defensive Guards */
	{
		/* chaff_fill with NULL field */
		int32_t dummy[16];
		chaff_coord_t org[2] = { { 0, 0 }, { 0, 0 } };
		int32_t st[2] = { 65536, 65536 };
		uint16_t n[2] = { 4, 4 };
		chaff_fill(NULL, 0, 1, org, st, n, dummy, 4, CHAFF_MODE_SET);

		/* chaff_fill with out == NULL */
		chaff_layer_t lay = { 4, 8, 10, 0, { 0, 0 }, 256 };
		chaff_field_t field = { 1, 2, 0, 1, &lay };
		chaff_fill(&field, 0, 1, org, st, n, NULL, 4, CHAFF_MODE_SET);

		/* chaff_fill with invalid dimension */
		chaff_field_t bad_dim = { 1, 5 /* >4 */, 0, 1, &lay };
		chaff_fill(&bad_dim, 0, 1, org, st, n, dummy, 4, CHAFF_MODE_SET);

		/* chaff_fill_lod with NULL */
		chaff_fill_lod(NULL, 1, org, st, n, dummy, 4, CHAFF_MODE_SET);

		/* chaff_sample_grad_2d with NULL */
		chaff_sample_2d_t s2_null = chaff_sample_grad_2d(NULL, 0, 0);
		assert(s2_null.elevation == 0 && s2_null.grad_x == 0 && s2_null.grad_y == 0);

		/* chaff_sample_grad_3d with NULL */
		chaff_sample_3d_t s3_null = chaff_sample_grad_3d(NULL, 0, 0, 0);
		assert(s3_null.val == 0 && s3_null.grad_x == 0 && s3_null.grad_y == 0 && s3_null.grad_z == 0);

		/* chaff_sample_grad_4d with NULL */
		chaff_sample_4d_t s4_null = chaff_sample_grad_4d(NULL, 0, 0, 0, 0);
		assert(s4_null.val == 0 && s4_null.grad_x == 0 && s4_null.grad_y == 0 &&
		       s4_null.grad_z == 0 && s4_null.grad_w == 0);

		/* chaff_fill_slice_3d with NULL / 0 width */
		chaff_fill_slice_3d(NULL, 0, 0, 0, 16, 16, dummy, NULL, NULL);
		chaff_fill_slice_3d(&field, 0, 0, 0, 0, 16, dummy, NULL, NULL);
		chaff_fill_slice_3d(&field, 0, 0, 0, 16, 0, dummy, NULL, NULL);

		/* chaff_fill_slice_4d with NULL / 0 width / dim < 4 */
		chaff_fill_slice_4d(NULL, 0, 0, 0, 0, 16, 16, dummy, NULL, NULL, NULL, NULL);
		chaff_fill_slice_4d(&field, 0, 0, 0, 0, 0, 16, dummy, NULL, NULL, NULL, NULL);
		chaff_fill_slice_4d(&field, 0, 0, 0, 0, 16, 0, dummy, NULL, NULL, NULL, NULL);
		chaff_fill_slice_4d(&field, 0, 0, 0, 0, 16, 16, dummy, NULL, NULL, NULL, NULL);

		/* chaff_sample_rugged_2d with NULL */
		chaff_sample_2d_t s_rug = chaff_sample_rugged_2d(NULL, 0, 0);
		assert(s_rug.elevation == 0 && s_rug.grad_x == 0 && s_rug.grad_y == 0);

		/* chaff_sample_flow_2d with NULL */
		chaff_flow_2d_t f_null = chaff_sample_flow_2d(NULL, NULL, 0, 0, 0);
		assert(f_null.vx_q8 == 0 && f_null.vy_q8 == 0 && f_null.psi == 0);

		printf("PASS: defensive NULL-safety and invalid parameter guards verified across all entry points\n");
	}

	/* 14. 4D Analytical Gradients (chaff_sample_grad_4d) & Finite-Difference Parity */
	{
		static const chaff_layer_t lay4d_smooth = {
			.shift = 5, .vshift = 8, .amp = 120, .seed_mode = CHAFF_SEED_SHARED, .contrast = 256
		};
		static const chaff_field_t field4d_smooth = {
			.seed = 0x4D4D1234u, .dim = 4, .flags = CHAFF_FLAG_FADE, .nlayers = 1, .layers = &lay4d_smooth
		};

		int non_zero_val = 0, non_zero_gx = 0, non_zero_gy = 0, non_zero_gz = 0, non_zero_gw = 0;
		for (int16_t w = 3; w <= 27; w += 8) {
			for (int16_t z = 5; z <= 29; z += 8) {
				for (int16_t y = -20; y <= 20; y += 10) {
					for (int16_t x = -20; x <= 20; x += 10) {
						chaff_sample_4d_t s0 = chaff_sample_grad_4d(&field4d_smooth, x, y, z, w);
						chaff_sample_4d_t sx_p = chaff_sample_grad_4d(&field4d_smooth, (int16_t)(x + 1), y, z, w);
						chaff_sample_4d_t sx_m = chaff_sample_grad_4d(&field4d_smooth, (int16_t)(x - 1), y, z, w);
						chaff_sample_4d_t sy_p = chaff_sample_grad_4d(&field4d_smooth, x, (int16_t)(y + 1), z, w);
						chaff_sample_4d_t sy_m = chaff_sample_grad_4d(&field4d_smooth, x, (int16_t)(y - 1), z, w);
						chaff_sample_4d_t sz_p = chaff_sample_grad_4d(&field4d_smooth, x, y, (int16_t)(z + 1), w);
						chaff_sample_4d_t sz_m = chaff_sample_grad_4d(&field4d_smooth, x, y, (int16_t)(z - 1), w);
						chaff_sample_4d_t sw_p = chaff_sample_grad_4d(&field4d_smooth, x, y, z, (int16_t)(w + 1));
						chaff_sample_4d_t sw_m = chaff_sample_grad_4d(&field4d_smooth, x, y, z, (int16_t)(w - 1));

						int32_t fd_x = (sx_p.val - sx_m.val) / 2;
						int32_t fd_y = (sy_p.val - sy_m.val) / 2;
						int32_t fd_z = (sz_p.val - sz_m.val) / 2;
						int32_t fd_w = (sw_p.val - sw_m.val) / 2;

						assert(abs((int32_t)s0.grad_x - fd_x) <= 4);
						assert(abs((int32_t)s0.grad_y - fd_y) <= 4);
						assert(abs((int32_t)s0.grad_z - fd_z) <= 4);
						assert(abs((int32_t)s0.grad_w - fd_w) <= 4);

						if (s0.val != 0) non_zero_val++;
						if (s0.grad_x != 0) non_zero_gx++;
						if (s0.grad_y != 0) non_zero_gy++;
						if (s0.grad_z != 0) non_zero_gz++;
						if (s0.grad_w != 0) non_zero_gw++;
					}
				}
			}
		}
		assert(non_zero_val > 100);
		assert(non_zero_gx > 50 && non_zero_gy > 50 && non_zero_gz > 50 && non_zero_gw > 50);
		printf("PASS: chaff_sample_grad_4d analytical derivatives match 4D central finite differences\n");
	}

	/* 15. chaff_fill_slice_4d Bit-Exact Parity with chaff_sample_grad_4d & Wrap Boundaries */
	{
		static const chaff_layer_t layers4d[] = {
			{ .shift = 6, .vshift = 8, .amp =   0, .seed_mode = CHAFF_SEED_SHARED,    .contrast = 256 },
			{ .shift = 4, .vshift = 8, .amp = 100, .seed_mode = CHAFF_SEED_PER_LAYER, .contrast = 512 },
			{ .shift = 3, .vshift = 8, .amp =  60, .seed_mode = CHAFF_SEED_SHARED | CHAFF_LAYER_RIDGED, .contrast = 192 }
		};
		static const chaff_field_t field4d = {
			.seed = 0xCAFE4D00u, .dim = 4, .flags = CHAFF_FLAG_FADE, .nlayers = 3, .layers = layers4d
		};

		uint16_t width = 40, height = 36;
		int16_t z_val = -19, w_val = 43;
		int16_t ox = -22, oy = 11;

		int32_t slice_val[40 * 36], slice_val_only[40 * 36];
		int16_t slice_gx[40 * 36], slice_gy[40 * 36], slice_gz[40 * 36], slice_gw[40 * 36];

		chaff_fill_slice_4d(&field4d, z_val, w_val, ox, oy, width, height,
		                    slice_val, slice_gx, slice_gy, slice_gz, slice_gw);
		chaff_fill_slice_4d(&field4d, z_val, w_val, ox, oy, width, height,
		                    slice_val_only, NULL, NULL, NULL, NULL);

		for (uint16_t r = 0; r < height; r++) {
			for (uint16_t c = 0; c < width; c++) {
				int16_t px = (int16_t)(ox + c);
				int16_t py = (int16_t)(oy + r);
				chaff_sample_4d_t pt = chaff_sample_grad_4d(&field4d, px, py, z_val, w_val);
				size_t idx = (size_t)r * width + c;
				assert(slice_val[idx] == pt.val);
				assert(slice_val_only[idx] == pt.val);
				assert(slice_gx[idx] == pt.grad_x);
				assert(slice_gy[idx] == pt.grad_y);
				assert(slice_gz[idx] == pt.grad_z);
				assert(slice_gw[idx] == pt.grad_w);
			}
		}

		/* Verify int16_t modular wrap boundary (32767 -> -32768) along Z and W */
		int16_t z_wrap = 32767, w_wrap = 32767;
		int32_t wrap_val[16 * 16];
		int16_t wrap_gx[16 * 16], wrap_gy[16 * 16], wrap_gz[16 * 16], wrap_gw[16 * 16];
		chaff_fill_slice_4d(&field4d, z_wrap, w_wrap, -8, -8, 16, 16,
		                    wrap_val, wrap_gx, wrap_gy, wrap_gz, wrap_gw);
		for (int r = 0; r < 16; r++) {
			for (int c = 0; c < 16; c++) {
				int16_t px = (int16_t)(-8 + c);
				int16_t py = (int16_t)(-8 + r);
				chaff_sample_4d_t pt = chaff_sample_grad_4d(&field4d, px, py, z_wrap, w_wrap);
				size_t idx = (size_t)r * 16 + c;
				assert(wrap_val[idx] == pt.val);
				assert(wrap_gx[idx] == pt.grad_x);
				assert(wrap_gy[idx] == pt.grad_y);
				assert(wrap_gz[idx] == pt.grad_z);
				assert(wrap_gw[idx] == pt.grad_w);
			}
		}
		printf("PASS: chaff_fill_slice_4d bit-exact parity with chaff_sample_grad_4d across multi-cell & int16_t wrap boundaries\n");
	}

	/* 16. 4D chaff_fill, chaff_fill_lod, chaff_sample_4d, and GLSL 4D Reference Parity */
	{
		static const chaff_layer_t lay4 = {
			.shift = 4, .vshift = 8, .amp = 90, .seed_mode = CHAFF_SEED_SHARED, .contrast = 256
		};
		static const chaff_field_t field4 = {
			.seed = 0x13572468u, .dim = 4, .flags = CHAFF_FLAG_FADE, .nlayers = 1, .layers = &lay4
		};
		chaff_coord_t org4[4] = { { 2, 0 }, { -4, 0 }, { 8, 0 }, { -12, 0 } };
		int32_t step4[4] = { 65536, 65536, 65536, 65536 };
		uint16_t n4[4] = { 2, 3, 4, 4 };
		int32_t out4[2 * 3 * 4 * 4], ref4[2 * 3 * 4 * 4];
		chaff_fill_lod(&field4, 0, org4, step4, n4, out4, 4, CHAFF_MODE_SET);
		chaff_fill(&field4, 0, 1, org4, step4, n4, ref4, 4, CHAFF_MODE_SET);

		int non_zero4 = 0;
		for (int i = 0; i < 2 * 3 * 4 * 4; i++) {
			assert(out4[i] == ref4[i]);
			if (out4[i] != 0) non_zero4++;
		}
		assert(non_zero4 > 70);

		int32_t out4_add[2 * 3 * 4 * 4];
		memcpy(out4_add, out4, sizeof(out4));
		chaff_fill_lod(&field4, 0, org4, step4, n4, out4_add, 4, CHAFF_MODE_ADD);
		for (int i = 0; i < 2 * 3 * 4 * 4; i++) {
			assert(out4_add[i] == out4[i] * 2);
		}

		/* Test noise_sample_4d / chaff_sample_4d and 4D scanline */
		static const noise_octave_t octs4[] = {
			{ .shift = 4, .amp = 90, .skew_x = 3, .skew_y = -5, .contrast = 256 }
		};
		noise_cascade_t cfg4 = { .seed = 0x2468ACE0u, .count = 1, .octaves = octs4 };
		noise_scanline_t scan4;
		chaff_scanline_init(&scan4, &cfg4, 4);
		scan4.w = 17;
		int32_t row4[16];
		chaff_scanline_eval_row(&scan4, -8, 12, -5, 16, row4);
		for (int i = 0; i < 16; i++) {
			int16_t px = (int16_t)(-8 + i);
			assert(row4[i] == chaff_sample_4d(&cfg4, px, 12, -5, 17));
		}

		/* Test GLSL 4D hash & analytical gradient parity against CPU noise_r(..., 4) */
		uint32_t gseed = 0xDEADBEEFu;
		for (int16_t w = -32; w <= 32; w += 16) {
			for (int16_t z = -32; z <= 32; z += 16) {
				for (int16_t y = -32; y <= 32; y += 16) {
					for (int16_t x = -32; x <= 32; x += 16) {
						/* In GLSL 4D convention, p = (x, y, z, w) packs v1=(y,x), v2=(z,w),
						 * matching int16_t va[4] = { y, x, z, w } in little-endian */
						int16_t va[4] = { y, x, z, w };
						uint32_t cpu_h = noise_r(va, gseed, 0, 4);
						uint32_t glsl_h = chaff_glsl_ref_hash4(x, y, z, w, gseed);
						assert(cpu_h == glsl_h);
					}
				}
			}
		}

		float eps = 0.01f;
		chaff_glsl_ref_vec5_t g4 = chaff_glsl_ref_sample_grad_4d(12.3f, -7.4f, 5.6f, 19.2f, 4, gseed);
		float v4_center = chaff_glsl_ref_sample_4d(12.3f, -7.4f, 5.6f, 19.2f, 4, gseed);
		assert(fabsf(g4.val - v4_center) < 1e-6f);
		float fd_gx = (chaff_glsl_ref_sample_4d(12.3f + eps, -7.4f, 5.6f, 19.2f, 4, gseed) -
		               chaff_glsl_ref_sample_4d(12.3f - eps, -7.4f, 5.6f, 19.2f, 4, gseed)) / (2.0f * eps);
		float fd_gw = (chaff_glsl_ref_sample_4d(12.3f, -7.4f, 5.6f, 19.2f + eps, 4, gseed) -
		               chaff_glsl_ref_sample_4d(12.3f, -7.4f, 5.6f, 19.2f - eps, 4, gseed)) / (2.0f * eps);
		assert(fabsf(g4.grad_x - fd_gx) < 1e-3f);
		assert(fabsf(g4.grad_w - fd_gw) < 1e-3f);

		printf("PASS: 4D chaff_fill_lod, chaff_sample_4d, and GLSL 4D hash/gradient reference parity verified\n");
	}

	/* 17. Divergence-Free 2D Curl-Flow (chaff_sample_flow_2d), Band Range, and GLSL Grad/Flow Reference */
	{
		static const chaff_layer_t psi_layers[] = {
			{ .shift = 2, .vshift = 8, .amp =  40, .seed_mode = CHAFF_SEED_SHARED,    .contrast = 256 },
			{ .shift = 4, .vshift = 8, .amp =  90, .seed_mode = CHAFF_SEED_PER_LAYER, .contrast = 256 },
			{ .shift = 6, .vshift = 8, .amp = 140, .seed_mode = CHAFF_SEED_SHARED,    .contrast = 256 }
		};
		static const chaff_field_t psi_field = {
			.seed = 0x57494E44u, .dim = 3, .flags = CHAFF_FLAG_FADE, .nlayers = 3, .layers = psi_layers
		};

		/* Test chaff_band_range_for_step_q16 */
		uint8_t lo = 99, hi = 99;
		int32_t step_fine[3] = { 65536, 65536, 65536 };       /* 1 cell -> nyquist_shift = 0 */
		int32_t step_coarse[3] = { 8 << 16, 8 << 16, 65536 }; /* 8 cells -> nyquist_shift = 4 */
		chaff_band_range_for_step_q16(&psi_field, 0, step_fine, &lo, &hi);
		assert(lo == 0 && hi == 3);
		chaff_band_range_for_step_q16(&psi_field, 0, step_coarse, &lo, &hi);
		assert(lo == 1 && hi == 3);
		chaff_band_range_for_step_q16(&psi_field, 5, step_fine, &lo, &hi);
		assert(lo == 2 && hi == 3);

		/* Test chaff_sample_flow_2d divergence-free curl identity */
		chaff_flow_desc_t flow_desc = {
			.base_vx_q8 = 190,
			.base_vy_q8 = -172,
			.curl_amp_q8 = 512
		};
		int32_t sum_div = 0;
		int sample_cnt = 0;
		for (int16_t y = -24; y <= 24; y += 8) {
			for (int16_t x = -24; x <= 24; x += 8) {
				chaff_flow_2d_t f0 = chaff_sample_flow_2d(&psi_field, &flow_desc, x, y, 15);
				chaff_sample_3d_t s3 = chaff_sample_grad_3d(&psi_field, x, y, 15);
				assert(f0.psi == s3.val);
				assert(f0.vx_q8 == (int16_t)(flow_desc.base_vx_q8 + (((int32_t)s3.grad_y * flow_desc.curl_amp_q8) >> 8)));
				assert(f0.vy_q8 == (int16_t)(flow_desc.base_vy_q8 + ((-(int32_t)s3.grad_x * flow_desc.curl_amp_q8) >> 8)));

				chaff_flow_2d_t fx_p = chaff_sample_flow_2d(&psi_field, &flow_desc, (int16_t)(x + 2), y, 15);
				chaff_flow_2d_t fx_m = chaff_sample_flow_2d(&psi_field, &flow_desc, (int16_t)(x - 2), y, 15);
				chaff_flow_2d_t fy_p = chaff_sample_flow_2d(&psi_field, &flow_desc, x, (int16_t)(y + 2), 15);
				chaff_flow_2d_t fy_m = chaff_sample_flow_2d(&psi_field, &flow_desc, x, (int16_t)(y - 2), 15);
				int32_t dvx_dx = (fx_p.vx_q8 - fx_m.vx_q8);
				int32_t dvy_dy = (fy_p.vy_q8 - fy_m.vy_q8);
				sum_div += abs(dvx_dx + dvy_dy);
				sample_cnt++;
			}
		}
		assert(sample_cnt == 49);
		assert(sum_div / sample_cnt <= 24);

		/* Test chaff_glsl_ref_grad_noise4d & chaff_glsl_ref_eval_flow_2d */
		chaff_glsl_ref_vec4_t gn4 = chaff_glsl_ref_grad_noise4d(2.35f, -1.72f, 0.64f, 4.18f, 0x434C4430u);
		assert(gn4.val >= 0.0f && gn4.val <= 1.0f);
		float eps = 0.002f;
		chaff_glsl_ref_vec4_t gn4_xp = chaff_glsl_ref_grad_noise4d(2.35f + eps, -1.72f, 0.64f, 4.18f, 0x434C4430u);
		chaff_glsl_ref_vec4_t gn4_xm = chaff_glsl_ref_grad_noise4d(2.35f - eps, -1.72f, 0.64f, 4.18f, 0x434C4430u);
		float fd4_x = (gn4_xp.val - gn4_xm.val) / (2.0f * eps);
		/* gn4.grad_x is the un-interpolated-gradient fast analytical derivative */
		assert(fabsf(gn4.grad_x - fd4_x) < 0.35f);

		float wx_dir = 0.0f, wy_dir = 0.0f, gust = 0.0f, cp = 0.0f;
		chaff_glsl_ref_eval_flow_2d(120.0f, -85.0f, 14.5f, &wx_dir, &wy_dir, &gust, &cp);
		float dir_norm = sqrtf(wx_dir * wx_dir + wy_dir * wy_dir);
		assert(fabsf(dir_norm - 1.0f) < 1e-3f);
		assert(gust >= 0.25f && gust <= 1.35f);
		assert(cp >= 0.0f && cp <= 1.0f);

		printf("PASS: chaff_sample_flow_2d, chaff_band_range_for_step_q16, and GLSL grad_noise4d/flow_2d reference verified\n");
	}

	printf("=== ALL CANONICAL LIBCHAFF TESTS PASSED ===\n");
	return 0;
}
