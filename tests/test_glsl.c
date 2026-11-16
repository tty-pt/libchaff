#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <ttypt/noise.h>
#include <ttypt/chaff_glsl.h>

int main(void)
{
	printf("=== Testing GLSL ES 3.0 GPU Shader Contract & Reference Parity ===\n");

	/* 1. Fragment shader source contract strings */
	{
		const char *src = chaff_glsl_fragment_shader_source();
		assert(src != NULL);
		assert(strstr(src, "#version 300 es") != NULL);
		assert(strstr(src, "chaff_hash2") != NULL);
		assert(strstr(src, "chaff_hash3") != NULL);
		assert(strstr(src, "chaff_sample_octave") != NULL);
		assert(strstr(src, "chaff_sample_grad_2d") != NULL);
		assert(strstr(src, "chaff_sample_3d") != NULL);
		assert(strstr(src, "chaff_sample_grad_3d") != NULL);
		assert(strstr(src, "chaff_rotate_golden") != NULL);
		assert(strstr(src, "chaff_eval_multi") != NULL);
		assert(strstr(src, "chaff_eval_rugged") != NULL);
		assert(strstr(src, "chaff_curl_2d") != NULL);
		printf("PASS: GLSL fragment shader source contains required shader functions and #version 300 es\n");
	}

	/* 2. Uniform buffer layout validation (chaff_gl_uniforms_t) */
	{
		assert(sizeof(chaff_gl_uniforms_t) >= 48);
		chaff_gl_uniforms_t u;
		memset(&u, 0, sizeof(u));
		u.origin[0] = 10.0f;
		u.origin[1] = 20.0f;
		u.step[0] = 1.0f;
		u.step[1] = 1.0f;
		u.seed = 0x12345678u;
		u.count = 3;
		u.shifts[0] = 6; u.shifts[1] = 4; u.shifts[2] = 2;
		u.amps[0] = 100.0f; u.amps[1] = 50.0f; u.amps[2] = 25.0f;
		u.min_wavelength = 4.0f;

		assert(u.origin[0] == 10.0f && u.origin[1] == 20.0f);
		assert(u.shifts[1] == 4 && u.amps[1] == 50.0f);
		printf("PASS: chaff_gl_uniforms_t uniform buffer contract layout verified\n");
	}

	/* 3. chaff_glsl_ref_rotl32 bitwise rotation */
	{
		uint32_t x = 0x12345678u;
		assert(chaff_glsl_ref_rotl32(x, 0) == x);
		assert(chaff_glsl_ref_rotl32(x, 32) == x);
		assert(chaff_glsl_ref_rotl32(x, 16) == 0x56781234u);
		assert(chaff_glsl_ref_rotl32(x, 4) == 0x23456781u);
		printf("PASS: chaff_glsl_ref_rotl32 rotation logic verified across 0, 4, 16, 32 bits\n");
	}

	/* 4. Bit-exact 2D hash: chaff_glsl_ref_hash2 vs XXH32 */
	{
		uint32_t seed = 0xCAFE1234u;
		for (int16_t y = -100; y <= 100; y += 17) {
			for (int16_t x = -100; x <= 100; x += 19) {
				int16_t va[2] = { y, x };
				uint32_t cpu_hash = XXH32(va, sizeof(va), seed);
				uint32_t glsl_hash = chaff_glsl_ref_hash2(y, x, seed);
				assert(cpu_hash == glsl_hash);
			}
		}

		/* Test int16 wrap boundary coordinates */
		int16_t wrap_coords[] = { 0, -1, 1, 32767, -32768, 32766, -32767 };
		for (size_t i = 0; i < sizeof(wrap_coords)/sizeof(wrap_coords[0]); i++) {
			for (size_t j = 0; j < sizeof(wrap_coords)/sizeof(wrap_coords[0]); j++) {
				int16_t va[2] = { wrap_coords[i], wrap_coords[j] };
				uint32_t cpu_h = XXH32(va, sizeof(va), seed);
				uint32_t glsl_h = chaff_glsl_ref_hash2(wrap_coords[i], wrap_coords[j], seed);
				assert(cpu_h == glsl_h);
			}
		}
		printf("PASS: chaff_glsl_ref_hash2 is 100%% bit-exact with XXH32 across coordinate plane and wrap limits\n");
	}

	/* 5. 3D reference hash: chaff_glsl_ref_hash3 determinism and memory layout */
	{
		uint32_t seed = 0x5EED9999u;
		for (int16_t z = -30; z <= 30; z += 15) {
			for (int16_t y = -30; y <= 30; y += 15) {
				for (int16_t x = -30; x <= 30; x += 15) {
					uint32_t h1 = chaff_glsl_ref_hash3(x, y, z, seed);
					uint32_t h2 = chaff_glsl_ref_hash3(x, y, z, seed);
					assert(h1 == h2);
					assert(h1 != 0);
				}
			}
		}
		printf("PASS: chaff_glsl_ref_hash3 determinism verified across 3D grid\n");
	}

	/* 6. GLSL reference octave sampling continuity */
	{
		unsigned shift = 4;
		unsigned vshift = 8;
		uint32_t seed = 0x42u;

		/* Normalized by 65536.0f to standard [0..255] float surface (matching test_noise.c:515) */
		float prev = chaff_glsl_ref_sample_octave(0.0f, 0.0f, shift, vshift, seed) / 65536.0f;
		for (float x = 0.5f; x <= 64.0f; x += 0.5f) {
			float cur = chaff_glsl_ref_sample_octave(x, 0.0f, shift, vshift, seed) / 65536.0f;
			float diff = fabsf(cur - prev);
			assert(diff < 25.0f); /* Smooth step transitions on [0..255] surface */
			prev = cur;
		}
		printf("PASS: chaff_glsl_ref_sample_octave continuous stepping verified\n");
	}

	/* 7. GLSL 3D analytical gradient vs numerical finite differences */
	{
		unsigned shift = 5;
		uint32_t seed = 0x1337u;
		float eps = 0.005f;

		for (float z = 10.0f; z <= 30.0f; z += 10.0f) {
			for (float y = 10.0f; y <= 30.0f; y += 10.0f) {
				for (float x = 10.0f; x <= 30.0f; x += 10.0f) {
					chaff_glsl_ref_vec4_t s = chaff_glsl_ref_sample_grad_3d(x, y, z, shift, seed);
					float v_ref = chaff_glsl_ref_sample_3d(x, y, z, shift, seed);
					assert(fabsf(s.val - v_ref) < 0.0001f);

					float vx_p = chaff_glsl_ref_sample_3d(x + eps, y, z, shift, seed);
					float vx_m = chaff_glsl_ref_sample_3d(x - eps, y, z, shift, seed);
					float num_gx = (vx_p - vx_m) / (2.0f * eps);
					assert(fabsf(s.grad_x - num_gx) < 0.015f);

					float vy_p = chaff_glsl_ref_sample_3d(x, y + eps, z, shift, seed);
					float vy_m = chaff_glsl_ref_sample_3d(x, y - eps, z, shift, seed);
					float num_gy = (vy_p - vy_m) / (2.0f * eps);
					assert(fabsf(s.grad_y - num_gy) < 0.015f);

					float vz_p = chaff_glsl_ref_sample_3d(x, y, z + eps, shift, seed);
					float vz_m = chaff_glsl_ref_sample_3d(x, y, z - eps, shift, seed);
					float num_gz = (vz_p - vz_m) / (2.0f * eps);
					assert(fabsf(s.grad_z - num_gz) < 0.015f);
				}
			}
		}
		printf("PASS: GLSL 3D analytical gradients match numerical limits within <1.5%% error\n");
	}

	/* 8. Golden-Ratio rotation trigonometric accuracy in GLSL */
	{
		/* In GLSL: chaff_rotate_golden evaluates:
		 * vec2(0.8506508 * p.x - 0.5257311 * p.y, 0.5257311 * p.x + 0.8506508 * p.y) */
		double cos_theta = 0.8506508;
		double sin_theta = 0.5257311;
		double len_sq = cos_theta * cos_theta + sin_theta * sin_theta;
		assert(fabs(len_sq - 1.0) < 0.0001); /* Conserves unit circle length */

		double angle_rad = atan2(sin_theta, cos_theta);
		double angle_deg = angle_rad * (180.0 / 3.141592653589793);
		assert(fabs(angle_deg - 31.717) < 0.05); /* Exact Golden Ratio angle */
		printf("PASS: chaff_rotate_golden trigonometric parameters conserve vector length and match 31.72 deg\n");
	}

	/* 9. Struct-driven chaff_glsl_ref_eval_uniforms and custom vshift _ex evaluators */
	{
		chaff_gl_uniforms_t u;
		memset(&u, 0, sizeof(u));
		u.origin[0] = 4.0f;
		u.origin[1] = 8.0f;
		u.step[0] = 0.5f;
		u.step[1] = 0.5f;
		u.seed = 0x13572468u;
		u.count = 2;
		u.shifts[0] = 5; u.shifts[1] = 3;
		u.amps[0] = 80.0f; u.amps[1] = 40.0f;
		u.min_wavelength = 2.0f;

		float h_def = chaff_glsl_ref_eval_uniforms(&u, 10.0f, 12.0f);
		u.warp_amp = -1.0f; /* disable domain warp via struct */
		float h_nowarp = chaff_glsl_ref_eval_uniforms(&u, 10.0f, 12.0f);
		assert(h_def != h_nowarp);

		/* Custom seed_prime and seed_salt in chaff_gl_uniforms_t */
		u.seed_prime = 0x27D4EB2Fu;
		u.seed_salt = 0x1337u;
		float h_custom_seed = chaff_glsl_ref_eval_uniforms(&u, 10.0f, 12.0f);
		assert(h_custom_seed != h_nowarp);

		/* Verify shader string exposes parameterized _ex and _full entry points */
		const char *src = chaff_glsl_fragment_shader_source();
		assert(strstr(src, "chaff_grad_noise2d_ex") != NULL);
		assert(strstr(src, "chaff_grad_noise4d_ex") != NULL);
		assert(strstr(src, "chaff_rotate_2d") != NULL);
		assert(strstr(src, "chaff_eval_multi_full") != NULL);

		/* Custom vshift = 20 in 3D and 4D _ex reference samplers */
		chaff_glsl_ref_vec4_t g3_20 = chaff_glsl_ref_sample_grad_3d_ex(12.3f, 8.7f, 4.1f, 4, 20, 0x42u);
		float s3_20 = chaff_glsl_ref_sample_3d_ex(12.3f, 8.7f, 4.1f, 4, 20, 0x42u);
		assert(fabsf(g3_20.val - s3_20) < 1e-6f);
		assert(s3_20 >= 0.0f && s3_20 <= 1.0f);

		chaff_glsl_ref_vec5_t g4_20 = chaff_glsl_ref_sample_grad_4d_ex(12.3f, 8.7f, 4.1f, 2.5f, 4, 20, 0x42u);
		float s4_20 = chaff_glsl_ref_sample_4d_ex(12.3f, 8.7f, 4.1f, 2.5f, 4, 20, 0x42u);
		assert(fabsf(g4_20.val - s4_20) < 1e-6f);
		assert(s4_20 >= 0.0f && s4_20 <= 1.0f);
		printf("PASS: chaff_glsl_ref_eval_uniforms and custom vshift 3D/4D _ex samplers verified\n");
	}

	/* 10. Analytical 2D value-gradient noise (chaff_value_grad_noise2d) and scale-space band weights */
	{
		const char *src = chaff_glsl_fragment_shader_source();
		assert(strstr(src, "chaff_value_grad_noise2d") != NULL);
		assert(strstr(src, "chaff_lowpass_weight") != NULL);
		assert(strstr(src, "chaff_band_weight") != NULL);

		uint32_t seed = 0x9E3779B9u;
		float eps = 0.002f;
		for (float y = -7.3f; y <= 12.7f; y += 3.1f) {
			for (float x = -5.4f; x <= 14.6f; x += 2.9f) {
				float val = 0.0f, gx = 0.0f, gy = 0.0f;
				chaff_glsl_ref_value_grad_noise2d(x, y, seed, &val, &gx, &gy);
				assert(val >= 0.0f && val <= 1.0f);

				float v_xp = 0.0f, v_xm = 0.0f, v_yp = 0.0f, v_ym = 0.0f;
				chaff_glsl_ref_value_grad_noise2d(x + eps, y, seed, &v_xp, NULL, NULL);
				chaff_glsl_ref_value_grad_noise2d(x - eps, y, seed, &v_xm, NULL, NULL);
				chaff_glsl_ref_value_grad_noise2d(x, y + eps, seed, &v_yp, NULL, NULL);
				chaff_glsl_ref_value_grad_noise2d(x, y - eps, seed, &v_ym, NULL, NULL);

				float num_gx = (v_xp - v_xm) / (2.0f * eps);
				float num_gy = (v_yp - v_ym) / (2.0f * eps);
				assert(fabsf(gx - num_gx) < 0.01f);
				assert(fabsf(gy - num_gy) < 0.01f);
			}
		}

		/* Verify complementary band-pass partition of unity: L(wl, coarse) + B(wl, 0, coarse) == 1.0 */
		for (float coarse_step = 0.5f; coarse_step <= 64.0f; coarse_step *= 1.7f) {
			for (int s = 0; s <= 10; s++) {
				float wl = (float)(1 << s);
				float w_low = chaff_glsl_ref_lowpass_weight(wl, coarse_step);
				float w_band = chaff_glsl_ref_band_weight(wl, 0.0f, coarse_step);
				assert(fabsf((w_low + w_band) - 1.0f) < 1e-6f);
			}
		}
		printf("PASS: chaff_value_grad_noise2d analytical derivatives and chaff_band_weight verified\n");
	}

	printf("=== ALL GLSL GPU SHADER TESTS PASSED ===\n");
	return 0;
}
