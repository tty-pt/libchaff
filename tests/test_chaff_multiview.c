#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>

#include <ttypt/noise.h>
#include <ttypt/chaff.h>

typedef struct {
	const char *name;
	uint8_t     min_shift;
} test_view_t;

int main(void)
{
	printf("=== Testing libchaff Multi-View & Multi-Scale Nyquist Consistency ===\n");

	const test_view_t views[] = {
		{ "View0_Full",     0 },
		{ "View1_Fine",     4 },
		{ "View2_Medium",   6 },
		{ "View3_Coarse",  10 }
	};
	const size_t view_count = sizeof(views) / sizeof(views[0]);

	/* Test 1: Monotonic Fine-Octave Energy Attenuation Across Cutoff Shifts */
	{
		uint32_t seed = 0x48454947u;
		double energy[4] = { 0.0, 0.0, 0.0, 0.0 };
		size_t sample_count = 0;

		for (int32_t y = -25; y <= 25; y++) {
			for (int32_t x = -25; x <= 25; x++) {
				int16_t wx = (int16_t)(x * 32);
				int16_t wy = (int16_t)(y * 32);

				for (size_t v = 0; v < view_count; v++) {
					int32_t f = chaff_fbm_rotated_2d_q12_4(
						wx, wy, views[v].min_shift, 9, 7, 2, 1,
						seed, 0xBEEFu, CHAFF_ROT_ATAN_COS_Q15, CHAFF_ROT_ATAN_SIN_Q15);
					energy[v] += (double)f * (double)f;
				}
				sample_count++;
			}
		}

		double rms[4];
		for (size_t v = 0; v < view_count; v++) {
			rms[v] = sqrt(energy[v] / (double)sample_count);
		}

		assert(rms[0] >= rms[1]);
		assert(rms[1] >= rms[2]);
		assert(rms[2] >= rms[3]);
		assert(rms[3] == 0.0);

		printf("  [PASS] Monotonic octave attenuation verified (RMS: %.1f >= %.1f >= %.1f >= %.1f)\n",
		       rms[0], rms[1], rms[2], rms[3]);
	}

	/* Test 2: Multi-View Reentrancy & Interleaved State Isolation */
	{
		uint32_t seed = 0xABCDEF01u;
		const int num_points = 200;
		int32_t serial_val[4][200];
		int32_t interleaved_val[4][200];

		for (size_t v = 0; v < view_count; v++) {
			for (int i = 0; i < num_points; i++) {
				int16_t wx = (int16_t)((i * 123) % 5000 - 2500);
				int16_t wy = (int16_t)((i * 79)  % 4000 - 2000);
				serial_val[v][i] = chaff_fbm_rotated_2d_q12_4(
					wx, wy, views[v].min_shift, 9, 7, 2, 1,
					seed, 0xBEEFu, CHAFF_ROT_ATAN_COS_Q15, CHAFF_ROT_ATAN_SIN_Q15);
			}
		}

		for (int i = 0; i < num_points; i++) {
			int16_t wx = (int16_t)((i * 123) % 5000 - 2500);
			int16_t wy = (int16_t)((i * 79)  % 4000 - 2000);
			for (size_t v = 0; v < view_count; v++) {
				interleaved_val[v][i] = chaff_fbm_rotated_2d_q12_4(
					wx, wy, views[v].min_shift, 9, 7, 2, 1,
					seed, 0xBEEFu, CHAFF_ROT_ATAN_COS_Q15, CHAFF_ROT_ATAN_SIN_Q15);
			}
		}

		for (size_t v = 0; v < view_count; v++) {
			for (int i = 0; i < num_points; i++) {
				assert(serial_val[v][i] == interleaved_val[v][i]);
			}
		}

		printf("  [PASS] Multi-view reentrancy verified: 100%% bit-identical between serial and interleaved sampling\n");
	}

	/* Test 3: Spatial Lipschitz Continuity Across All Cutoff Shifts */
	{
		uint32_t seed = 0x13579BDFu;
		for (int32_t step = 0; step < 500; step++) {
			int16_t x0 = (int16_t)(step * 10 - 2500);
			int16_t y0 = (int16_t)(step * 8  - 2000);
			int16_t x1 = (int16_t)(x0 + 10);
			int16_t y1 = (int16_t)(y0 + 8);

			for (size_t v = 0; v < view_count; v++) {
				int32_t e0 = chaff_fbm_rotated_2d_q12_4(
					x0, y0, views[v].min_shift, 9, 7, 2, 1,
					seed, 0xBEEFu, CHAFF_ROT_ATAN_COS_Q15, CHAFF_ROT_ATAN_SIN_Q15);
				int32_t e1 = chaff_fbm_rotated_2d_q12_4(
					x1, y1, views[v].min_shift, 9, 7, 2, 1,
					seed, 0xBEEFu, CHAFF_ROT_ATAN_COS_Q15, CHAFF_ROT_ATAN_SIN_Q15);
				int32_t delta = abs(e1 - e0);
				assert(delta < 2000);
			}
		}

		printf("  [PASS] Spatial Lipschitz continuity bounded across all view cutoff shifts\n");
	}

	printf("=== ALL LIBCHAFF MULTI-VIEW & MULTI-SCALE TESTS PASSED ===\n");
	return 0;
}
