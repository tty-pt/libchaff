#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ttypt/noise.h>
#include <ttypt/chaff.h>

/* Forward declare symbols exported from src/libchaff.c */
int ttypt_noise_point_1d(int16_t x, uint32_t seed, uint8_t shift);
int ttypt_noise_point_2d(int16_t x, int16_t y, uint32_t seed, uint8_t shift);
int ttypt_noise_point_3d(int16_t x, int16_t y, int16_t z, uint32_t seed, uint8_t shift);
int ttypt_noise_point_4d(int16_t x, int16_t y, int16_t z, int16_t w, uint32_t seed, uint8_t shift);
int ttypt_noise_point_n(const int16_t *p, uint32_t seed, uint8_t shift, uint8_t dim);
void ttypt_noise_oct(uint32_t *m, int16_t *s, size_t oct_n, octave_t *oct, unsigned seed, unsigned cy, uint8_t dim);
uint32_t ttypt_noise_wpoint_n(const int16_t *p, unsigned x, unsigned w, unsigned seed, uint8_t dim, unsigned flags);
uint32_t ttypt_noise_wpoint_c(noise_wcache_t *c, const int16_t *p, unsigned x, unsigned w, unsigned seed, uint8_t dim, unsigned flags);
void ttypt_noise_wrow(uint32_t *out, unsigned n, const int16_t *p0, int16_t dstep, unsigned x, unsigned w, unsigned seed, uint8_t dim, unsigned flags, noise_wcache_t *c);
void ttypt_noise_wplane(uint32_t *out, unsigned n, unsigned rows, size_t stride, const int16_t *p0, unsigned x, unsigned w, unsigned seed, uint8_t dim, unsigned flags, noise_wcache_t *c);
void ttypt_noise_matrix_fill_2d(const noise_cascade_t *cfg, int16_t origin_x, int16_t origin_y, uint8_t size_shift, int32_t *out_matrix);
void ttypt_noise_scanline_init(noise_scanline_t *scan, const noise_cascade_t *cfg, uint8_t dim);
void ttypt_noise_scanline_eval_row(noise_scanline_t *scan, int16_t start_x, int16_t y, int16_t z, size_t count, int32_t *out_row);
int32_t ttypt_noise_sample_2d(const noise_cascade_t *cfg, int16_t x, int16_t y);
int32_t ttypt_noise_sample_3d(const noise_cascade_t *cfg, int16_t x, int16_t y, int16_t z);

int chaff_point_1d_sym(int16_t x, uint32_t seed, uint8_t shift);
int chaff_point_2d_sym(int16_t x, int16_t y, uint32_t seed, uint8_t shift);
int chaff_point_3d_sym(int16_t x, int16_t y, int16_t z, uint32_t seed, uint8_t shift);
int chaff_point_4d_sym(int16_t x, int16_t y, int16_t z, int16_t w, uint32_t seed, uint8_t shift);
int chaff_point_n_sym(const int16_t *p, uint32_t seed, uint8_t shift, uint8_t dim);
void chaff_matrix_fill_2d_sym(const noise_cascade_t *cfg, int16_t origin_x, int16_t origin_y, uint8_t size_shift, int32_t *out_matrix);
void chaff_scanline_init_sym(noise_scanline_t *scan, const noise_cascade_t *cfg, uint8_t dim);
void chaff_scanline_eval_row_sym(noise_scanline_t *scan, int16_t start_x, int16_t y, int16_t z, size_t count, int32_t *out_row);
int32_t chaff_sample_2d_sym(const noise_cascade_t *cfg, int16_t x, int16_t y);
int32_t chaff_sample_3d_sym(const noise_cascade_t *cfg, int16_t x, int16_t y, int16_t z);

void ttypt_chaff_fill(const chaff_field_t *field, uint8_t layer_lo, uint8_t layer_hi,
                      const chaff_coord_t origin[], const int32_t step_q16[],
                      const uint16_t n[], int32_t *out, size_t stride, unsigned mode);
void chaff_fill_sym(const chaff_field_t *field, uint8_t layer_lo, uint8_t layer_hi,
                    const chaff_coord_t origin[], const int32_t step_q16[],
                    const uint16_t n[], int32_t *out, size_t stride, unsigned mode);
chaff_sample_3d_t chaff_sample_grad_3d_sym(const chaff_field_t *field, int16_t x, int16_t y, int16_t z);
void chaff_fill_slice_3d_sym(const chaff_field_t *field, int16_t z,
                            int16_t origin_x, int16_t origin_y,
                            uint16_t width, uint16_t height,
                            int32_t *out_val, int16_t *out_grad_x, int16_t *out_grad_y);

int main(void)
{
	printf("=== Testing libnoise / libchaff Shared Library Exported ABI Symbols ===\n");

	/* 1. ttypt_noise_point_* and chaff_point_*_sym exports */
	{
		uint32_t seed = 0x55AA1122u;
		uint8_t shift = 4;
		for (int16_t v = -50; v <= 50; v += 10) {
			assert(ttypt_noise_point_1d(v, seed, shift) == noise_point_1d(v, seed, shift));
			assert(chaff_point_1d_sym(v, seed, shift) == noise_point_1d(v, seed, shift));

			assert(ttypt_noise_point_2d(v, v, seed, shift) == noise_point_2d(v, v, seed, shift));
			assert(chaff_point_2d_sym(v, v, seed, shift) == noise_point_2d(v, v, seed, shift));

			assert(ttypt_noise_point_3d(v, v, v, seed, shift) == noise_point_3d(v, v, v, seed, shift));
			assert(chaff_point_3d_sym(v, v, v, seed, shift) == noise_point_3d(v, v, v, seed, shift));

			assert(ttypt_noise_point_4d(v, v, v, v, seed, shift) == noise_point_4d(v, v, v, v, seed, shift));
			assert(chaff_point_4d_sym(v, v, v, v, seed, shift) == noise_point_4d(v, v, v, v, seed, shift));

			int16_t p[3] = { v, (int16_t)(v * 2), (int16_t)(v * 3) };
			assert(ttypt_noise_point_n(p, seed, shift, 3) == noise_point_n(p, seed, shift, 3));
			assert(chaff_point_n_sym(p, seed, shift, 3) == noise_point_n(p, seed, shift, 3));
		}
		printf("PASS: exported point sampling symbols match inline implementations\n");
	}

	/* 2. ttypt_noise_oct and matrix fills */
	{
		uint32_t m1[256], m2[256];
		int16_t s[2] = { 0, 0 };
		octave_t oct[] = { { 4, 1 } };
		ttypt_noise_oct(m1, s, 1, oct, 42, 4, 2);
		noise_oct(m2, s, 1, oct, 42, 4, 2);
		for (int i = 0; i < 256; i++) {
			assert(m1[i] == m2[i]);
		}

		static const noise_octave_t octs[] = {
			{ .shift = 4, .amp = 70, .skew_x = 0, .skew_y = 0, .contrast = 256 }
		};
		noise_cascade_t cfg = { .seed = 1234, .count = 1, .octaves = octs };
		int32_t mat_t[256], mat_c[256], mat_ref[256];
		ttypt_noise_matrix_fill_2d(&cfg, 0, 0, 4, mat_t);
		chaff_matrix_fill_2d_sym(&cfg, 0, 0, 4, mat_c);
		noise_matrix_fill_2d(&cfg, 0, 0, 4, mat_ref);
		for (int i = 0; i < 256; i++) {
			assert(mat_t[i] == mat_ref[i]);
			assert(mat_c[i] == mat_ref[i]);
		}
		printf("PASS: exported matrix fill symbols match inline implementations\n");
	}

	/* 3. Wrapping sampler symbols (wpoint, wrow, wplane) */
	{
		int16_t p[3] = { 10, 20, 30 };
		uint32_t wp1 = ttypt_noise_wpoint_n(p, 4, 8, 999, 3, NOISE_FADE);
		uint32_t wp2 = noise_wpoint_n(p, 4, 8, 999, 3, NOISE_FADE);
		assert(wp1 == wp2);

		noise_wcache_t c1, c2;
		noise_wcache_init(&c1);
		noise_wcache_init(&c2);
		assert(ttypt_noise_wpoint_c(&c1, p, 4, 8, 999, 3, NOISE_FADE) ==
		       noise_wpoint_c(&c2, p, 4, 8, 999, 3, NOISE_FADE));

		uint32_t r1[64], r2[64];
		ttypt_noise_wrow(r1, 64, p, 1, 4, 8, 999, 3, NOISE_FADE, &c1);
		noise_wrow(r2, 64, p, 1, 4, 8, 999, 3, NOISE_FADE, &c2);
		for (int i = 0; i < 64; i++) {
			assert(r1[i] == r2[i]);
		}

		uint32_t pl1[16 * 16], pl2[16 * 16];
		ttypt_noise_wplane(pl1, 16, 16, 16, p, 4, 8, 999, 3, NOISE_FADE, &c1);
		noise_wplane(pl2, 16, 16, 16, p, 4, 8, 999, 3, NOISE_FADE, &c2);
		for (int i = 0; i < 256; i++) {
			assert(pl1[i] == pl2[i]);
		}
		printf("PASS: exported wrapping sampler symbols (wpoint, wrow, wplane) verified\n");
	}

	/* 4. Scanline stepper and scalar sample symbols */
	{
		static const noise_octave_t octs[] = {
			{ .shift = 4, .amp = 50, .skew_x = 0, .skew_y = 0, .contrast = 256 }
		};
		noise_cascade_t cfg = { .seed = 777, .count = 1, .octaves = octs };
		noise_scanline_t s1, s2;
		ttypt_noise_scanline_init(&s1, &cfg, 2);
		chaff_scanline_init_sym(&s2, &cfg, 2);

		int32_t row1[32], row2[32];
		ttypt_noise_scanline_eval_row(&s1, 0, 10, 0, 32, row1);
		chaff_scanline_eval_row_sym(&s2, 0, 10, 0, 32, row2);
		for (int i = 0; i < 32; i++) {
			assert(row1[i] == row2[i]);
		}

		assert(ttypt_noise_sample_2d(&cfg, 15, 25) == noise_sample_2d(&cfg, 15, 25));
		assert(chaff_sample_2d_sym(&cfg, 15, 25) == noise_sample_2d(&cfg, 15, 25));

		assert(ttypt_noise_sample_3d(&cfg, 15, 25, 35) == noise_sample_3d(&cfg, 15, 25, 35));
		assert(chaff_sample_3d_sym(&cfg, 15, 25, 35) == noise_sample_3d(&cfg, 15, 25, 35));
		printf("PASS: exported scanline and sample symbols verified\n");
	}

	/* 5. Chaff fill, slice, and 3D gradient symbols */
	{
		static const chaff_layer_t lay = {
			.shift = 4, .vshift = 8, .amp = 90, .seed_mode = CHAFF_SEED_SHARED, .contrast = 256
		};
		static const chaff_field_t field = {
			.seed = 0x4321u, .dim = 3, .flags = CHAFF_FLAG_FADE, .nlayers = 1, .layers = &lay
		};

		chaff_coord_t org[3] = { { 0, 0 }, { 0, 0 }, { 0, 0 } };
		int32_t step[3] = { 65536, 65536, 65536 };
		uint16_t n[3] = { 2, 8, 8 };
		int32_t out1[128], out2[128], out_ref[128];

		ttypt_chaff_fill(&field, 0, 1, org, step, n, out1, 8, CHAFF_MODE_SET);
		chaff_fill_sym(&field, 0, 1, org, step, n, out2, 8, CHAFF_MODE_SET);
		chaff_fill(&field, 0, 1, org, step, n, out_ref, 8, CHAFF_MODE_SET);

		for (int i = 0; i < 128; i++) {
			assert(out1[i] == out_ref[i]);
			assert(out2[i] == out_ref[i]);
		}

		chaff_sample_3d_t g_sym = chaff_sample_grad_3d_sym(&field, 12, 14, 16);
		chaff_sample_3d_t g_ref = chaff_sample_grad_3d(&field, 12, 14, 16);
		assert(g_sym.val == g_ref.val);
		assert(g_sym.grad_x == g_ref.grad_x);
		assert(g_sym.grad_y == g_ref.grad_y);
		assert(g_sym.grad_z == g_ref.grad_z);

		int32_t val_sym[16], val_ref[16];
		chaff_fill_slice_3d_sym(&field, 10, 0, 0, 4, 4, val_sym, NULL, NULL);
		chaff_fill_slice_3d(&field, 10, 0, 0, 4, 4, val_ref, NULL, NULL);
		for (int i = 0; i < 16; i++) {
			assert(val_sym[i] == val_ref[i]);
		}

		printf("PASS: exported ttypt_chaff_fill, chaff_fill_sym, grad_3d_sym, and slice_3d_sym verified\n");
	}

	printf("=== ALL SHARED LIBRARY EXPORTED ABI SYMBOLS PASSED ===\n");
	return 0;
}
