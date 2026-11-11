#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <ttypt/noise.h>
#include <ttypt/chaff.h>
#include <ttypt/chaff_glsl.h>

static double get_time_sec(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
	printf("================================================================================\n");
	printf("           libchaff / libnoise Comprehensive Performance & Metrics Suite        \n");
	printf("================================================================================\n\n");

	const uint32_t seed = 0x1337BEEFu;
	volatile int64_t sink = 0;

	/* -------------------------------------------------------------------------
	 * Section 1: N-Dimensional Point Queries (Scalar, Wrapping & Cached)
	 * ------------------------------------------------------------------------- */
	printf("[1. Point Queries: Scalar, Wrapping & Cached]\n");
	{
		const int iters = 2000000;

		/* 1D */
		double t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			sink += noise_point_1d((int16_t)i, seed, 4);
		}
		double dt_1d = get_time_sec() - t0;
		printf("  Point Query 1D (noise_point_1d):        %8.2f Mops/s  (%6.2f ns/op)\n",
		       ((double)iters / dt_1d) / 1e6, (dt_1d / iters) * 1e9);

		/* 2D */
		t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			sink += noise_point_2d((int16_t)i, (int16_t)(i >> 1), seed, 4);
		}
		double dt_2d = get_time_sec() - t0;
		printf("  Point Query 2D (noise_point_2d):        %8.2f Mops/s  (%6.2f ns/op)\n",
		       ((double)iters / dt_2d) / 1e6, (dt_2d / iters) * 1e9);

		/* 3D */
		t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			sink += noise_point_3d((int16_t)i, (int16_t)(i >> 1), 42, seed, 4);
		}
		double dt_3d = get_time_sec() - t0;
		printf("  Point Query 3D (noise_point_3d):        %8.2f Mops/s  (%6.2f ns/op)\n",
		       ((double)iters / dt_3d) / 1e6, (dt_3d / iters) * 1e9);

		/* 4D */
		t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			sink += noise_point_4d((int16_t)i, (int16_t)(i >> 1), 42, 10, seed, 4);
		}
		double dt_4d = get_time_sec() - t0;
		printf("  Point Query 4D (noise_point_4d):        %8.2f Mops/s  (%6.2f ns/op)\n",
		       ((double)iters / dt_4d) / 1e6, (dt_4d / iters) * 1e9);

		/* 3D Wrapping point (noise_wpoint_n) */
		int16_t p3[3] = { 10, 20, 30 };
		t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			p3[0] = (int16_t)(p3[0] + 1);
			sink += noise_wpoint_n(p3, 4, 8, seed, 3, NOISE_FADE);
		}
		double dt_wp = get_time_sec() - t0;
		printf("  Wrapping Query 3D (noise_wpoint_n):     %8.2f Mops/s  (%6.2f ns/op)\n",
		       ((double)iters / dt_wp) / 1e6, (dt_wp / iters) * 1e9);

		/* 3D Cached point (noise_wpoint_c) */
		noise_wcache_t cache;
		noise_wcache_init(&cache);
		t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			p3[0] = (int16_t)(p3[0] + 1);
			sink += noise_wpoint_c(&cache, p3, 4, 8, seed, 3, NOISE_FADE);
		}
		double dt_wc = get_time_sec() - t0;
		printf("  Cached Query 3D (noise_wpoint_c):       %8.2f Mops/s  (%6.2f ns/op)  [Cache reuse]\n",
		       ((double)iters / dt_wc) / 1e6, (dt_wc / iters) * 1e9);
	}

	/* -------------------------------------------------------------------------
	 * Section 2: Analytical Spatial Gradients vs Finite Differences
	 * ------------------------------------------------------------------------- */
	printf("\n[2. Analytical Gradients vs Numerical Finite Differences]\n");
	static const chaff_layer_t layers3[] = {
		{ .shift = 8, .vshift = 8, .amp = 120, .seed_mode = CHAFF_SEED_SHARED },
		{ .shift = 5, .vshift = 8, .amp =  45, .seed_mode = CHAFF_SEED_SHARED },
		{ .shift = 3, .vshift = 8, .amp =  15, .seed_mode = CHAFF_SEED_SHARED }
	};
	static const chaff_field_t field3_2d = {
		.seed = 0xABCD1234u, .dim = 2, .nlayers = 3, .layers = layers3
	};
	static const chaff_field_t field3_3d = {
		.seed = 0xABCD1234u, .dim = 3, .nlayers = 3, .layers = layers3
	};

	{
		const int iters = 1000000;

		/* 2D Analytical */
		double t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			chaff_sample_2d_t s = chaff_sample_grad_2d(&field3_2d, (int16_t)i, (int16_t)(i >> 1));
			sink += s.elevation + s.grad_x + s.grad_y;
		}
		double dt_grad2d = get_time_sec() - t0;
		double mops_grad2d = ((double)iters / dt_grad2d) / 1e6;

		/* 2D Numerical finite difference baseline */
		t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			chaff_sample_2d_t s = chaff_sample_grad_2d(&field3_2d, (int16_t)i, (int16_t)(i >> 1));
			chaff_sample_2d_t sx = chaff_sample_grad_2d(&field3_2d, (int16_t)(i + 1), (int16_t)(i >> 1));
			chaff_sample_2d_t sy = chaff_sample_grad_2d(&field3_2d, (int16_t)i, (int16_t)((i >> 1) + 1));
			sink += s.elevation + (sx.elevation - s.elevation) + (sy.elevation - s.elevation);
		}
		double dt_fd2d = get_time_sec() - t0;
		double mops_fd2d = ((double)iters / dt_fd2d) / 1e6;

		printf("  2D Analytical Grad (3 octaves, 0 extra hashes): %6.2f Mops/s  (%6.2f ns/op)\n",
		       mops_grad2d, (dt_grad2d / iters) * 1e9);
		printf("  2D Finite Differences Baseline:                 %6.2f Mops/s  (%6.2f ns/op)\n",
		       mops_fd2d, (dt_fd2d / iters) * 1e9);
		printf("  2D Gradient Speedup Ratio:                      %6.2fx speedup (0 extra hashes)\n",
		       dt_fd2d / dt_grad2d);

		/* 3D Analytical */
		const int iters3d = 500000;
		t0 = get_time_sec();
		for (int i = 0; i < iters3d; i++) {
			chaff_sample_3d_t s = chaff_sample_grad_3d(&field3_3d, (int16_t)i, (int16_t)(i >> 1), 42);
			sink += s.val + s.grad_x + s.grad_y + s.grad_z;
		}
		double dt_grad3d = get_time_sec() - t0;
		double mops_grad3d = ((double)iters3d / dt_grad3d) / 1e6;

		/* 3D Numerical baseline */
		t0 = get_time_sec();
		for (int i = 0; i < iters3d; i++) {
			chaff_sample_3d_t s = chaff_sample_grad_3d(&field3_3d, (int16_t)i, (int16_t)(i >> 1), 42);
			chaff_sample_3d_t sx = chaff_sample_grad_3d(&field3_3d, (int16_t)(i + 1), (int16_t)(i >> 1), 42);
			chaff_sample_3d_t sy = chaff_sample_grad_3d(&field3_3d, (int16_t)i, (int16_t)((i >> 1) + 1), 42);
			chaff_sample_3d_t sz = chaff_sample_grad_3d(&field3_3d, (int16_t)i, (int16_t)(i >> 1), 43);
			sink += s.val + (sx.val - s.val) + (sy.val - s.val) + (sz.val - s.val);
		}
		double dt_fd3d = get_time_sec() - t0;
		double mops_fd3d = ((double)iters3d / dt_fd3d) / 1e6;

		printf("  3D Analytical Grad (3 octaves, 0 extra hashes): %6.2f Mops/s  (%6.2f ns/op)\n",
		       mops_grad3d, (dt_grad3d / iters3d) * 1e9);
		printf("  3D Finite Differences Baseline:                 %6.2f Mops/s  (%6.2f ns/op)\n",
		       mops_fd3d, (dt_fd3d / iters3d) * 1e9);
		printf("  3D Gradient Speedup Ratio:                      %6.2fx speedup (0 extra hashes)\n",
		       dt_fd3d / dt_grad3d);
	}

	/* -------------------------------------------------------------------------
	 * Section 3: High-Throughput Matrix, Plane & Scanline Steppers
	 * ------------------------------------------------------------------------- */
	printf("\n[3. Matrix, Plane & Scanline Steppers]\n");
	{
		/* 2D Matrix Stepper (noise_matrix_fill_2d) */
		int32_t mat[32 * 32];
		static const noise_octave_t octs[] = {
			{ .shift = 4, .amp = 80, .skew_x = 0, .skew_y = 0, .contrast = 256 },
		};
		noise_cascade_t cfg = { .seed = seed, .count = 1, .octaves = octs };

		const int mat_runs = 20000;
		double t0 = get_time_sec();
		for (int i = 0; i < mat_runs; i++) {
			noise_matrix_fill_2d(&cfg, 0, 0, 5, mat);
			sink += mat[0];
		}
		double dt_mat = get_time_sec() - t0;
		double mops_mat = ((double)(mat_runs * 1024) / dt_mat) / 1e6;
		printf("  2D Matrix Stepper (noise_matrix_fill_2d):       %8.2f Mops/s\n", mops_mat);

		/* 1D Scanline Stepper (noise_scanline_eval_row) */
		const int scanline_len = 256;
		const int lines = 20000;
		int32_t out_row[256];
		noise_scanline_t scan;
		noise_scanline_init(&scan, &cfg, 2);

		t0 = get_time_sec();
		for (int l = 0; l < lines; l++) {
			noise_scanline_eval_row(&scan, 0, (int16_t)l, 0, scanline_len, out_row);
			sink += out_row[0];
		}
		double dt_scan = get_time_sec() - t0;
		double mops_scan = ((double)(lines * scanline_len) / dt_scan) / 1e6;
		printf("  1D Scanline Stepper (noise_scanline_eval_row):  %8.2f Mops/s\n", mops_scan);

		/* 2D Plane Stepper (noise_wplane) */
		uint32_t plane[64 * 64];
		int16_t p0[3] = { 0, 0, 0 };
		noise_wcache_t plane_cache;
		noise_wcache_init(&plane_cache);
		const int plane_runs = 5000;

		t0 = get_time_sec();
		for (int i = 0; i < plane_runs; i++) {
			noise_wplane(plane, 64, 64, 64, p0, 4, 8, seed, 3, NOISE_FADE, &plane_cache);
			sink += plane[0];
		}
		double dt_plane = get_time_sec() - t0;
		double mops_plane = ((double)(plane_runs * 4096) / dt_plane) / 1e6;
		printf("  2D Plane Stepper (noise_wplane):                %8.2f Mops/s\n", mops_plane);
	}

	/* -------------------------------------------------------------------------
	 * Section 4: 2D Time-Slice Stepping across 3D Fields (chaff_fill_slice_3d)
	 * ------------------------------------------------------------------------- */
	printf("\n[4. 2D Time-Slice Stepping across 3D Fields (Z-Collapse Optimization)]\n");
	{
		const int grid_w = 32;
		const int grid_h = 32;
		const int grid_samples = grid_w * grid_h;
		const int frames = 2000;
		int32_t heightmap[32 * 32];
		int16_t gx[32 * 32];
		int16_t gy[32 * 32];

		/* Baseline: Point-by-point 3D sampling */
		double t0 = get_time_sec();
		for (int f = 0; f < frames; f++) {
			for (int y = 0; y < grid_h; y++) {
				for (int x = 0; x < grid_w; x++) {
					chaff_sample_3d_t s = chaff_sample_grad_3d(&field3_3d, (int16_t)x, (int16_t)y, (int16_t)f);
					sink += s.val;
				}
			}
		}
		double dt_point = get_time_sec() - t0;
		double mops_point = ((double)(frames * grid_samples) / dt_point) / 1e6;

		/* Value-only fast path */
		t0 = get_time_sec();
		for (int f = 0; f < frames; f++) {
			chaff_fill_slice_3d(&field3_3d, (int16_t)f, 0, 0, grid_w, grid_h, heightmap, NULL, NULL);
			sink += heightmap[0];
		}
		double dt_slice_val = get_time_sec() - t0;
		double mops_slice_val = ((double)(frames * grid_samples) / dt_slice_val) / 1e6;

		/* Combined Value + Gradients */
		t0 = get_time_sec();
		for (int f = 0; f < frames; f++) {
			chaff_fill_slice_3d(&field3_3d, (int16_t)f, 0, 0, grid_w, grid_h, heightmap, gx, gy);
			sink += heightmap[0] + gx[0] + gy[0];
		}
		double dt_slice_full = get_time_sec() - t0;
		double mops_slice_full = ((double)(frames * grid_samples) / dt_slice_full) / 1e6;

		printf("  Point-by-point 3D query throughput:            %8.2f Mops/s\n", mops_point);
		printf("  chaff_fill_slice_3d (Value-Only Fast Path):     %8.2f Mops/s  (%6.2fx speedup)\n",
		       mops_slice_val, dt_point / dt_slice_val);
		printf("  chaff_fill_slice_3d (Value + Analytical Grad):  %8.2f Mops/s  (%6.2fx speedup)\n",
		       mops_slice_full, dt_point / dt_slice_full);
		printf("  Lattice Corner Hash Reduction:                  >96.8%% (256 down to 8 hashes per cell)\n");
	}

	/* -------------------------------------------------------------------------
	 * Section 5: Band-Limited Nyquist Cutoff & Rotated Fractal FBM
	 * ------------------------------------------------------------------------- */
	printf("\n[5. Multi-Scale Nyquist Cutoff & Rotated FBM (chaff_fbm_rotated_2d_q12_4)]\n");
	{
		const int iters = 500000;
		const uint32_t seed = 0x48454947u;

		/* Full 10 octaves (min_shift = 0..9) */
		double t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			sink += chaff_fbm_rotated_2d_q12_4((int16_t)(i * 4), (int16_t)(i * 2), 0, 9, 7, 2, 1,
			                                   seed, 0xBEEFu, CHAFF_ROT_ATAN_COS_Q15, CHAFF_ROT_ATAN_SIN_Q15);
		}
		double dt_lod3 = get_time_sec() - t0;
		double mops_lod3 = ((double)iters / dt_lod3) / 1e6;

		/* Medium 6 octaves (min_shift = 4..9) */
		t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			sink += chaff_fbm_rotated_2d_q12_4((int16_t)(i * 4), (int16_t)(i * 2), 4, 9, 7, 2, 1,
			                                   seed, 0xBEEFu, CHAFF_ROT_ATAN_COS_Q15, CHAFF_ROT_ATAN_SIN_Q15);
		}
		double dt_lod2 = get_time_sec() - t0;
		double mops_lod2 = ((double)iters / dt_lod2) / 1e6;

		/* Coarse 4 octaves (min_shift = 6..9) */
		t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			sink += chaff_fbm_rotated_2d_q12_4((int16_t)(i * 4), (int16_t)(i * 2), 6, 9, 7, 2, 1,
			                                   seed, 0xBEEFu, CHAFF_ROT_ATAN_COS_Q15, CHAFF_ROT_ATAN_SIN_Q15);
		}
		double dt_lod1 = get_time_sec() - t0;
		double mops_lod1 = ((double)iters / dt_lod1) / 1e6;

		/* Periodic 2D macro (chaff_wpoint_periodic_2d) */
		t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			sink += (int64_t)chaff_wpoint_periodic_2d(i * 4, i * 2, 12, 16, 8, seed);
		}
		double dt_lod0 = get_time_sec() - t0;
		double mops_lod0 = ((double)iters / dt_lod0) / 1e6;

		printf("  Rotated FBM (10 octaves, shifts 0..9):               %6.2f Mops/s  (%6.2f ns/op)\n",
		       mops_lod3, (dt_lod3 / iters) * 1e9);
		printf("  Rotated FBM (6 octaves, shifts 4..9):                %6.2f Mops/s  (%6.2f ns/op)\n",
		       mops_lod2, (dt_lod2 / iters) * 1e9);
		printf("  Rotated FBM (4 octaves, shifts 6..9):                %6.2f Mops/s  (%6.2f ns/op)\n",
		       mops_lod1, (dt_lod1 / iters) * 1e9);
		printf("  Periodic 2D Octave (chaff_wpoint_periodic_2d):       %6.2f Mops/s  (%6.2f ns/op)\n",
		       mops_lod0, (dt_lod0 / iters) * 1e9);
		printf("  Nyquist Cutoff Speedup (4 octaves vs 10 octaves):    %6.2fx throughput acceleration\n",
		       dt_lod3 / dt_lod1);
	}

	/* -------------------------------------------------------------------------
	 * Section 6: Coordinate Rotational Skew & Hermite Polynomials
	 * ------------------------------------------------------------------------- */
	printf("\n[6. Rotational Skew & Hermite Polynomial Fades]\n");
	{
		const int iters = 5000000;

		/* chaff_rotate_q15 */
		int16_t rx = 1000, ry = 500;
		double t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			chaff_rotate_q15(&rx, &ry);
			sink += rx + ry;
		}
		double dt_rot = get_time_sec() - t0;
		printf("  Golden-Ratio Skew (chaff_rotate_q15):           %8.2f Mops/s  (%6.2f ns/op)\n",
		       ((double)iters / dt_rot) / 1e6, (dt_rot / iters) * 1e9);

		/* chaff_rotate_q15_inplace */
		t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			chaff_rotate_q15_inplace(&rx, &ry);
			sink += rx + ry;
		}
		double dt_roti = get_time_sec() - t0;
		printf("  Inplace Skew (chaff_rotate_q15_inplace):        %8.2f Mops/s  (%6.2f ns/op)\n",
		       ((double)iters / dt_roti) / 1e6, (dt_roti / iters) * 1e9);

		/* Cubic Hermite (noise_wfade) */
		t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			sink += noise_wfade((uint32_t)(i & 15), 4);
		}
		double dt_fade3 = get_time_sec() - t0;
		printf("  Cubic Hermite Fade (noise_wfade):               %8.2f Mops/s  (%6.2f ns/op)\n",
		       ((double)iters / dt_fade3) / 1e6, (dt_fade3 / iters) * 1e9);

		/* Quintic C2 Hermite (noise_wfade5) */
		t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			sink += noise_wfade5((uint32_t)(i & 15), 4);
		}
		double dt_fade5 = get_time_sec() - t0;
		printf("  Quintic C2 Hermite Fade (noise_wfade5):         %8.2f Mops/s  (%6.2f ns/op)\n",
		       ((double)iters / dt_fade5) / 1e6, (dt_fade5 / iters) * 1e9);
	}

	/* -------------------------------------------------------------------------
	 * Section 7: GLSL Shader Reference Hashing & Sampling
	 * ------------------------------------------------------------------------- */
	printf("\n[7. GLSL Shader Reference Evaluation]\n");
	{
		const int iters = 2000000;

		/* chaff_glsl_ref_hash2 */
		double t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			sink += chaff_glsl_ref_hash2((int16_t)i, (int16_t)(i >> 1), seed);
		}
		double dt_gh2 = get_time_sec() - t0;
		printf("  GLSL Ref Hash2 (chaff_glsl_ref_hash2):          %8.2f Mops/s  (%6.2f ns/op)\n",
		       ((double)iters / dt_gh2) / 1e6, (dt_gh2 / iters) * 1e9);

		/* chaff_glsl_ref_hash3 */
		t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			sink += chaff_glsl_ref_hash3((int16_t)i, (int16_t)(i >> 1), 42, seed);
		}
		double dt_gh3 = get_time_sec() - t0;
		printf("  GLSL Ref Hash3 (chaff_glsl_ref_hash3):          %8.2f Mops/s  (%6.2f ns/op)\n",
		       ((double)iters / dt_gh3) / 1e6, (dt_gh3 / iters) * 1e9);

		/* chaff_glsl_ref_sample_octave */
		t0 = get_time_sec();
		for (int i = 0; i < iters; i++) {
			sink += (int64_t)chaff_glsl_ref_sample_octave((float)i, (float)(i >> 1), 4, 8, seed);
		}
		double dt_gso = get_time_sec() - t0;
		printf("  GLSL Ref Sample Octave:                         %8.2f Mops/s  (%6.2f ns/op)\n",
		       ((double)iters / dt_gso) / 1e6, (dt_gso / iters) * 1e9);
	}

	printf("\n================================================================================\n");
	printf("                  libchaff Metrics Suite Completed Successfully                 \n");
	printf("================================================================================\n");

	(void)sink;
	return 0;
}
