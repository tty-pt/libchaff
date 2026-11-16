#ifndef TTYPT_CHAFF_GLSL_H
#define TTYPT_CHAFF_GLSL_H

/*
 * ttypt/chaff_glsl.h — Canonical GPU Shader Contract for libchaff.
 *
 * Exposes GLSL ES 3.0 / WebGL 2 shader source strings and uniform layouts
 * for hardware-accelerated procedural noise on clients.
 *
 * INVARIANT: Bit-exact corner hashing matching XXH32(p, 4, seed) on CPU.
 * Zero OpenGL/EGL runtime library links or compile-time dependencies on server.
 */

#include <stdint.h>
#include <math.h>

#define CHAFF_GLSL_VERSION \
	"#version 300 es\n" \
	"precision highp float;\n" \
	"precision highp int;\n"

/* Bit-exact XXH32 hash for 2D, 3D, and 4D lattice coordinates in GLSL ES 3.0.
 * Memory layout matches int16_t va[2] = { y, x }, va[3] = { y, x, z }, and va[4] = { y, x, z, w } in little-endian. */
#define CHAFF_GLSL_HASH_FN \
	"uint chaff_rotl32(uint x, int r) {\n" \
	"    return (x << uint(r)) | (x >> uint(32 - r));\n" \
	"}\n" \
	"uint chaff_hash2(ivec2 p, uint seed) {\n" \
	"    uint val = (uint(p.y) & 0xFFFFu) | ((uint(p.x) & 0xFFFFu) << 16u);\n" \
	"    uint h = seed + 0x165667B1u + 4u;\n" \
	"    h += val * 0xC2B2AE3Du;\n" \
	"    h = chaff_rotl32(h, 17) * 0x27D4EB2Fu;\n" \
	"    h ^= h >> 15u;\n" \
	"    h *= 0x85EBCA77u;\n" \
	"    h ^= h >> 13u;\n" \
	"    h *= 0xC2B2AE3Du;\n" \
	"    h ^= h >> 16u;\n" \
	"    return h;\n" \
	"}\n" \
	"uint chaff_hash3(ivec3 p, uint seed) {\n" \
	"    uint v1 = (uint(p.y) & 0xFFFFu) | ((uint(p.x) & 0xFFFFu) << 16u);\n" \
	"    uint v2 = (uint(p.z) & 0xFFFFu);\n" \
	"    uint h = seed + 0x165667B1u + 6u;\n" \
	"    h += v1 * 0xC2B2AE3Du;\n" \
	"    h = chaff_rotl32(h, 17) * 0x27D4EB2Fu;\n" \
	"    h += v2 * 0x85EBCA77u;\n" \
	"    h = chaff_rotl32(h, 13) * 0xC2B2AE3Du;\n" \
	"    h ^= h >> 15u;\n" \
	"    h *= 0x85EBCA77u;\n" \
	"    h ^= h >> 16u;\n" \
	"    return h;\n" \
	"}\n" \
	"uint chaff_hash4(ivec4 p, uint seed) {\n" \
	"    uint v1 = (uint(p.y) & 0xFFFFu) | ((uint(p.x) & 0xFFFFu) << 16u);\n" \
	"    uint v2 = (uint(p.z) & 0xFFFFu) | ((uint(p.w) & 0xFFFFu) << 16u);\n" \
	"    uint h = seed + 0x165667B1u + 8u;\n" \
	"    h += v1 * 0xC2B2AE3Du;\n" \
	"    h = chaff_rotl32(h, 17) * 0x27D4EB2Fu;\n" \
	"    h += v2 * 0xC2B2AE3Du;\n" \
	"    h = chaff_rotl32(h, 17) * 0x27D4EB2Fu;\n" \
	"    h ^= h >> 15u;\n" \
	"    h *= 0x85EBCA77u;\n" \
	"    h ^= h >> 13u;\n" \
	"    h *= 0xC2B2AE3Du;\n" \
	"    h ^= h >> 16u;\n" \
	"    return h;\n" \
	"}\n"

#define CHAFF_GLSL_GRAD2D_NORM 0.70710678f
#define CHAFF_GLSL_GRAD4D_NORM 0.72f
#define CHAFF_GLSL_DEFAULT_RIDGE_OCTAVE_STEP 0.25f

/* Continuous C2-quintic 2D and 4D gradient noise with analytical spatial derivatives.
 * Loop-free explicit corner evaluations for consumer GPU shader compiler friendliness. */
#define CHAFF_GLSL_GRAD_NOISE_FN \
	"vec2 chaff_grad_dir(int cx, int cy, uint seed) {\n" \
	"    uint h = chaff_hash3(ivec3(cx, cy, 0), seed);\n" \
	"    float gx = float(int(h & 0xFFFFu) - 32768) * (1.0 / 32768.0);\n" \
	"    float gy = float(int(h >> 16u) - 32768) * (1.0 / 32768.0);\n" \
	"    vec2 g = vec2(gx, gy);\n" \
	"    return g * inversesqrt(dot(g, g) + 1e-6);\n" \
	"}\n" \
	"vec3 chaff_grad_noise2d_ex(vec2 p, uint seed, float grad_norm) {\n" \
	"    float gn = (grad_norm > 0.0) ? grad_norm : 0.70710678;\n" \
	"    vec2 ip = floor(p);\n" \
	"    vec2 fp = p - ip;\n" \
	"    vec2 u = fp * fp * fp * (fp * (fp * 6.0 - 15.0) + 10.0);\n" \
	"    vec2 du = 30.0 * fp * fp * (fp * (fp - 2.0) + 1.0);\n" \
	"    ivec2 c = ivec2(ip);\n" \
	"    vec2 g00 = chaff_grad_dir(c.x,     c.y,     seed);\n" \
	"    vec2 g10 = chaff_grad_dir(c.x + 1, c.y,     seed);\n" \
	"    vec2 g01 = chaff_grad_dir(c.x,     c.y + 1, seed);\n" \
	"    vec2 g11 = chaff_grad_dir(c.x + 1, c.y + 1, seed);\n" \
	"    float v00 = dot(g00, fp);\n" \
	"    float v10 = dot(g10, fp - vec2(1.0, 0.0));\n" \
	"    float v01 = dot(g01, fp - vec2(0.0, 1.0));\n" \
	"    float v11 = dot(g11, fp - vec2(1.0, 1.0));\n" \
	"    float k0 = v10 - v00;\n" \
	"    float k1 = v01 - v00;\n" \
	"    float k2 = v11 - v01 - k0;\n" \
	"    float val = 0.5 + gn * (v00 + k0 * u.x + k1 * u.y + k2 * u.x * u.y);\n" \
	"    vec2 gInterp = g00 + (g10 - g00) * u.x + (g01 - g00) * u.y + (g11 - g01 - g10 + g00) * (u.x * u.y);\n" \
	"    vec2 grad = gn * (gInterp + du * vec2(k0 + k2 * u.y, k1 + k2 * u.x));\n" \
	"    return vec3(clamp(val, 0.0, 1.0), grad.x, grad.y);\n" \
	"}\n" \
	"vec3 chaff_grad_noise2d(vec2 p, uint seed) {\n" \
	"    return chaff_grad_noise2d_ex(p, seed, 0.70710678);\n" \
	"}\n" \
	"vec4 chaff_grad4(ivec4 c, uint seed) {\n" \
	"    uint h = chaff_hash4(c, seed);\n" \
	"    vec4 g = vec4(\n" \
	"        float(int( h        & 0xFFu) - 128),\n" \
	"        float(int((h >> 8u)  & 0xFFu) - 128),\n" \
	"        float(int((h >> 16u) & 0xFFu) - 128),\n" \
	"        float(int((h >> 24u) & 0xFFu) - 128)\n" \
	"    );\n" \
	"    return g * inversesqrt(dot(g, g) + 1e-6);\n" \
	"}\n" \
	"float chaff_grad4_dot(ivec4 c, vec4 d, uint seed) {\n" \
	"    return dot(chaff_grad4(c, seed), d);\n" \
	"}\n" \
	"vec4 chaff_grad_noise4d_ex(vec4 p, uint seed, float grad_norm) {\n" \
	"    float gn = (grad_norm > 0.0) ? grad_norm : 0.72;\n" \
	"    vec4 ip = floor(p);\n" \
	"    vec4 fp = p - ip;\n" \
	"    vec4 u = fp * fp * fp * (fp * (fp * 6.0 - 15.0) + 10.0);\n" \
	"    vec4 du = 30.0 * fp * fp * (fp * (fp - 2.0) + 1.0);\n" \
	"    ivec4 c = ivec4(ip);\n" \
	"    float d0 = mix(chaff_grad4_dot(c + ivec4(0,0,0,0), fp - vec4(0.0,0.0,0.0,0.0), seed), chaff_grad4_dot(c + ivec4(0,0,0,1), fp - vec4(0.0,0.0,0.0,1.0), seed), u.w);\n" \
	"    float d1 = mix(chaff_grad4_dot(c + ivec4(1,0,0,0), fp - vec4(1.0,0.0,0.0,0.0), seed), chaff_grad4_dot(c + ivec4(1,0,0,1), fp - vec4(1.0,0.0,0.0,1.0), seed), u.w);\n" \
	"    float d2 = mix(chaff_grad4_dot(c + ivec4(0,1,0,0), fp - vec4(0.0,1.0,0.0,0.0), seed), chaff_grad4_dot(c + ivec4(0,1,0,1), fp - vec4(0.0,1.0,0.0,1.0), seed), u.w);\n" \
	"    float d3 = mix(chaff_grad4_dot(c + ivec4(1,1,0,0), fp - vec4(1.0,1.0,0.0,0.0), seed), chaff_grad4_dot(c + ivec4(1,1,0,1), fp - vec4(1.0,1.0,0.0,1.0), seed), u.w);\n" \
	"    float d4 = mix(chaff_grad4_dot(c + ivec4(0,0,1,0), fp - vec4(0.0,0.0,1.0,0.0), seed), chaff_grad4_dot(c + ivec4(0,0,1,1), fp - vec4(0.0,0.0,1.0,1.0), seed), u.w);\n" \
	"    float d5 = mix(chaff_grad4_dot(c + ivec4(1,0,1,0), fp - vec4(1.0,0.0,1.0,0.0), seed), chaff_grad4_dot(c + ivec4(1,0,1,1), fp - vec4(1.0,0.0,1.0,1.0), seed), u.w);\n" \
	"    float d6 = mix(chaff_grad4_dot(c + ivec4(0,1,1,0), fp - vec4(0.0,1.0,1.0,0.0), seed), chaff_grad4_dot(c + ivec4(0,1,1,1), fp - vec4(0.0,1.0,1.0,1.0), seed), u.w);\n" \
	"    float d7 = mix(chaff_grad4_dot(c + ivec4(1,1,1,0), fp - vec4(1.0,1.0,1.0,0.0), seed), chaff_grad4_dot(c + ivec4(1,1,1,1), fp - vec4(1.0,1.0,1.0,1.0), seed), u.w);\n" \
	"    float c0 = mix(d0, d4, u.z);\n" \
	"    float c1 = mix(d1, d5, u.z);\n" \
	"    float c2 = mix(d2, d6, u.z);\n" \
	"    float c3 = mix(d3, d7, u.z);\n" \
	"    float b0 = mix(c0, c2, u.y);\n" \
	"    float b1 = mix(c1, c3, u.y);\n" \
	"    float val = clamp(0.5 + gn * mix(b0, b1, u.x), 0.0, 1.0);\n" \
	"    float dN_dx = (b1 - b0) * du.x * gn;\n" \
	"    float dN_dy = mix(c2 - c0, c3 - c1, u.x) * du.y * gn;\n" \
	"    float dz0 = mix(d4 - d0, d6 - d2, u.y);\n" \
	"    float dz1 = mix(d5 - d1, d7 - d3, u.y);\n" \
	"    float dN_dz = mix(dz0, dz1, u.x) * du.z * gn;\n" \
	"    return vec4(val, dN_dx, dN_dy, dN_dz);\n" \
	"}\n" \
	"vec4 chaff_grad_noise4d(vec4 p, uint seed) {\n" \
	"    return chaff_grad_noise4d_ex(p, seed, 0.72);\n" \
	"}\n" \
	"vec2 chaff_curl_2d(vec2 p, uint seed) {\n" \
	"    vec3 n = chaff_grad_noise2d(p, seed);\n" \
	"    return vec2(-n.z, n.y);\n" \
	"}\n" \
	"vec3 chaff_value_grad_noise2d(vec2 p, uint seed) {\n" \
	"    vec2 ip = floor(p);\n" \
	"    vec2 fp = p - ip;\n" \
	"    vec2 u = fp * fp * fp * (fp * (fp * 6.0 - 15.0) + 10.0);\n" \
	"    vec2 du = 30.0 * fp * fp * (fp * (fp - 2.0) + 1.0);\n" \
	"    ivec2 c = ivec2(ip);\n" \
	"    float v00 = float(chaff_hash2(c, seed) >> 8u) * (1.0 / 16777215.0);\n" \
	"    float v10 = float(chaff_hash2(c + ivec2(1, 0), seed) >> 8u) * (1.0 / 16777215.0);\n" \
	"    float v01 = float(chaff_hash2(c + ivec2(0, 1), seed) >> 8u) * (1.0 / 16777215.0);\n" \
	"    float v11 = float(chaff_hash2(c + ivec2(1, 1), seed) >> 8u) * (1.0 / 16777215.0);\n" \
	"    float k0 = v10 - v00;\n" \
	"    float k1 = v01 - v00;\n" \
	"    float k2 = v11 - v01 - k0;\n" \
	"    float val = v00 + k0 * u.x + k1 * u.y + k2 * u.x * u.y;\n" \
	"    vec2 grad = du * vec2(k0 + k2 * u.y, k1 + k2 * u.x);\n" \
	"    return vec3(val, grad.x, grad.y);\n" \
	"}\n"

/* Continuous scale-space C2 Nyquist low-pass and complementary band-pass weights */
#define CHAFF_GLSL_BAND_FN \
	"float chaff_lowpass_weight(float wavelength, float step_wu) {\n" \
	"    if (step_wu <= 0.0) return 1.0;\n" \
	"    float t = clamp((wavelength - 2.0 * step_wu) / (2.0 * step_wu), 0.0, 1.0);\n" \
	"    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);\n" \
	"}\n" \
	"float chaff_band_weight(float wavelength, float fine_step_wu, float coarse_step_wu) {\n" \
	"    float w_fine = chaff_lowpass_weight(wavelength, fine_step_wu);\n" \
	"    if (w_fine <= 0.0) return 0.0;\n" \
	"    float w_coarse = chaff_lowpass_weight(wavelength, coarse_step_wu);\n" \
	"    return w_fine * (1.0 - w_coarse);\n" \
	"}\n"

/* Continuous cubic Hermite interpolation matching noise_wcorners and noise_wstep.
 * Evaluates corner vertices at cell = floor(world_pos / 2^shift) * 2^shift. */
#define CHAFF_GLSL_EVAL_FN \
	CHAFF_GLSL_BAND_FN \
	"float chaff_sample_octave(vec2 world_pos, uint shift, uint vshift, uint seed) {\n" \
	"    float cell_size = float(1u << shift);\n" \
	"    vec2 p = world_pos / cell_size;\n" \
	"    ivec2 i0 = ivec2(floor(p));\n" \
	"    vec2 f = fract(p);\n" \
	"    vec2 u = f * f * (3.0 - 2.0 * f);\n" \
	"    int step_px = int(1u << shift);\n" \
	"    ivec2 cell = i0 * step_px;\n" \
	"    uint v00 = chaff_hash2(cell, seed) >> vshift;\n" \
	"    uint v10 = chaff_hash2(cell + ivec2(step_px, 0), seed) >> vshift;\n" \
	"    uint v01 = chaff_hash2(cell + ivec2(0, step_px), seed) >> vshift;\n" \
	"    uint v11 = chaff_hash2(cell + ivec2(step_px, step_px), seed) >> vshift;\n" \
	"    float b0 = mix(float(v00), float(v10), u.x);\n" \
	"    float b1 = mix(float(v01), float(v11), u.x);\n" \
	"    return mix(b0, b1, u.y);\n" \
	"}\n" \
	"vec3 chaff_sample_grad_2d(vec2 world_pos, uint shift, uint vshift, uint seed) {\n" \
	"    float cell_size = float(1u << shift);\n" \
	"    vec2 p = world_pos / cell_size;\n" \
	"    ivec2 i0 = ivec2(floor(p));\n" \
	"    vec2 f = fract(p);\n" \
	"    vec2 u = f * f * (3.0 - 2.0 * f);\n" \
	"    vec2 du = 6.0 * f * (1.0 - f);\n" \
	"    int step_px = int(1u << shift);\n" \
	"    ivec2 cell = i0 * step_px;\n" \
	"    float v00 = float(chaff_hash2(cell, seed) >> vshift);\n" \
	"    float v10 = float(chaff_hash2(cell + ivec2(step_px, 0), seed) >> vshift);\n" \
	"    float v01 = float(chaff_hash2(cell + ivec2(0, step_px), seed) >> vshift);\n" \
	"    float v11 = float(chaff_hash2(cell + ivec2(step_px, step_px), seed) >> vshift);\n" \
	"    float norm = float((1u << (32u - vshift)) - 1u);\n" \
	"    float b0 = mix(v00, v10, u.x);\n" \
	"    float b1 = mix(v01, v11, u.x);\n" \
	"    float val = mix(b0, b1, u.y) / norm;\n" \
	"    float dH_dx = mix(v10 - v00, v11 - v01, u.y);\n" \
	"    float dH_dy = b1 - b0;\n" \
	"    vec2 grad = (vec2(dH_dx, dH_dy) * du) / (norm * cell_size);\n" \
	"    return vec3(val, grad.x, grad.y);\n" \
	"}\n" \
	"float chaff_sample_3d_ex(vec3 pos, uint shift, uint vshift, uint seed) {\n" \
	"    float cell_size = float(1u << shift);\n" \
	"    vec3 p = pos / cell_size;\n" \
	"    ivec3 i0 = ivec3(floor(p));\n" \
	"    vec3 f = fract(p);\n" \
	"    vec3 u = f * f * (3.0 - 2.0 * f);\n" \
	"    int step_px = int(1u << shift);\n" \
	"    ivec3 cell = i0 * step_px;\n" \
	"    uint v000 = chaff_hash3(cell + ivec3(0, 0, 0), seed) >> vshift;\n" \
	"    uint v100 = chaff_hash3(cell + ivec3(step_px, 0, 0), seed) >> vshift;\n" \
	"    uint v010 = chaff_hash3(cell + ivec3(0, step_px, 0), seed) >> vshift;\n" \
	"    uint v110 = chaff_hash3(cell + ivec3(step_px, step_px, 0), seed) >> vshift;\n" \
	"    uint v001 = chaff_hash3(cell + ivec3(0, 0, step_px), seed) >> vshift;\n" \
	"    uint v101 = chaff_hash3(cell + ivec3(step_px, 0, step_px), seed) >> vshift;\n" \
	"    uint v011 = chaff_hash3(cell + ivec3(0, step_px, step_px), seed) >> vshift;\n" \
	"    uint v111 = chaff_hash3(cell + ivec3(step_px, step_px, step_px), seed) >> vshift;\n" \
	"    float b00 = mix(float(v000), float(v100), u.x);\n" \
	"    float b10 = mix(float(v010), float(v110), u.x);\n" \
	"    float b01 = mix(float(v001), float(v101), u.x);\n" \
	"    float b11 = mix(float(v011), float(v111), u.x);\n" \
	"    float b0 = mix(b00, b10, u.y);\n" \
	"    float b1 = mix(b01, b11, u.y);\n" \
	"    float norm = float((1u << (32u - vshift)) - 1u);\n" \
	"    return mix(b0, b1, u.z) / norm;\n" \
	"}\n" \
	"float chaff_sample_3d(vec3 pos, uint shift, uint seed) {\n" \
	"    return chaff_sample_3d_ex(pos, shift, 24u, seed);\n" \
	"}\n" \
	"vec4 chaff_sample_grad_3d_ex(vec3 pos, uint shift, uint vshift, uint seed) {\n" \
	"    float cell_size = float(1u << shift);\n" \
	"    vec3 p = pos / cell_size;\n" \
	"    ivec3 i0 = ivec3(floor(p));\n" \
	"    vec3 f = fract(p);\n" \
	"    vec3 u = f * f * (3.0 - 2.0 * f);\n" \
	"    vec3 du = 6.0 * f * (1.0 - f);\n" \
	"    int step_px = int(1u << shift);\n" \
	"    ivec3 cell = i0 * step_px;\n" \
	"    float v000 = float(chaff_hash3(cell + ivec3(0, 0, 0), seed) >> vshift);\n" \
	"    float v100 = float(chaff_hash3(cell + ivec3(step_px, 0, 0), seed) >> vshift);\n" \
	"    float v010 = float(chaff_hash3(cell + ivec3(0, step_px, 0), seed) >> vshift);\n" \
	"    float v110 = float(chaff_hash3(cell + ivec3(step_px, step_px, 0), seed) >> vshift);\n" \
	"    float v001 = float(chaff_hash3(cell + ivec3(0, 0, step_px), seed) >> vshift);\n" \
	"    float v101 = float(chaff_hash3(cell + ivec3(step_px, 0, step_px), seed) >> vshift);\n" \
	"    float v011 = float(chaff_hash3(cell + ivec3(0, step_px, step_px), seed) >> vshift);\n" \
	"    float v111 = float(chaff_hash3(cell + ivec3(step_px, step_px, step_px), seed) >> vshift);\n" \
	"    float norm = float((1u << (32u - vshift)) - 1u);\n" \
	"    float b00 = mix(v000, v100, u.x);\n" \
	"    float b10 = mix(v010, v110, u.x);\n" \
	"    float b01 = mix(v001, v101, u.x);\n" \
	"    float b11 = mix(v011, v111, u.x);\n" \
	"    float b0 = mix(b00, b10, u.y);\n" \
	"    float b1 = mix(b01, b11, u.y);\n" \
	"    float val = mix(b0, b1, u.z) / norm;\n" \
	"    float dx00 = v100 - v000;\n" \
	"    float dx10 = v110 - v010;\n" \
	"    float dx01 = v101 - v001;\n" \
	"    float dx11 = v111 - v011;\n" \
	"    float dx0 = mix(dx00, dx10, u.y);\n" \
	"    float dx1 = mix(dx01, dx11, u.y);\n" \
	"    float dN_dx = mix(dx0, dx1, u.z);\n" \
	"    float dy0 = b10 - b00;\n" \
	"    float dy1 = b11 - b01;\n" \
	"    float dN_dy = mix(dy0, dy1, u.z);\n" \
	"    float dN_dz = b1 - b0;\n" \
	"    vec3 grad = (vec3(dN_dx, dN_dy, dN_dz) * du) / (norm * cell_size);\n" \
	"    return vec4(val, grad.x, grad.y, grad.z);\n" \
	"}\n" \
	"vec4 chaff_sample_grad_3d(vec3 pos, uint shift, uint seed) {\n" \
	"    return chaff_sample_grad_3d_ex(pos, shift, 24u, seed);\n" \
	"}\n" \
	"float chaff_sample_4d_ex(vec4 pos, uint shift, uint vshift, uint seed) {\n" \
	"    float cell_size = float(1u << shift);\n" \
	"    vec4 p = pos / cell_size;\n" \
	"    ivec4 i0 = ivec4(floor(p));\n" \
	"    vec4 f = fract(p);\n" \
	"    vec4 u = f * f * (3.0 - 2.0 * f);\n" \
	"    int step_px = int(1u << shift);\n" \
	"    ivec4 cell = i0 * step_px;\n" \
	"    float d[8];\n" \
	"    for (int i = 0; i < 8; i++) {\n" \
	"        int ox = (i & 1) != 0 ? step_px : 0;\n" \
	"        int oy = (i & 2) != 0 ? step_px : 0;\n" \
	"        int oz = (i & 4) != 0 ? step_px : 0;\n" \
	"        float v0 = float(chaff_hash4(cell + ivec4(ox, oy, oz, 0), seed) >> vshift);\n" \
	"        float v1 = float(chaff_hash4(cell + ivec4(ox, oy, oz, step_px), seed) >> vshift);\n" \
	"        d[i] = mix(v0, v1, u.w);\n" \
	"    }\n" \
	"    float c0 = mix(d[0], d[4], u.z);\n" \
	"    float c1 = mix(d[1], d[5], u.z);\n" \
	"    float c2 = mix(d[2], d[6], u.z);\n" \
	"    float c3 = mix(d[3], d[7], u.z);\n" \
	"    float b0 = mix(c0, c2, u.y);\n" \
	"    float b1 = mix(c1, c3, u.y);\n" \
	"    float norm = float((1u << (32u - vshift)) - 1u);\n" \
	"    return mix(b0, b1, u.x) / norm;\n" \
	"}\n" \
	"float chaff_sample_4d(vec4 pos, uint shift, uint seed) {\n" \
	"    return chaff_sample_4d_ex(pos, shift, 24u, seed);\n" \
	"}\n" \
	"vec4 chaff_sample_grad_4d_ex(vec4 pos, uint shift, uint vshift, uint seed, out float out_grad_w) {\n" \
	"    float cell_size = float(1u << shift);\n" \
	"    vec4 p = pos / cell_size;\n" \
	"    ivec4 i0 = ivec4(floor(p));\n" \
	"    vec4 f = fract(p);\n" \
	"    vec4 u = f * f * (3.0 - 2.0 * f);\n" \
	"    vec4 du = 6.0 * f * (1.0 - f);\n" \
	"    int step_px = int(1u << shift);\n" \
	"    ivec4 cell = i0 * step_px;\n" \
	"    float d[8];\n" \
	"    float dw_c[8];\n" \
	"    for (int i = 0; i < 8; i++) {\n" \
	"        int ox = (i & 1) != 0 ? step_px : 0;\n" \
	"        int oy = (i & 2) != 0 ? step_px : 0;\n" \
	"        int oz = (i & 4) != 0 ? step_px : 0;\n" \
	"        float v0 = float(chaff_hash4(cell + ivec4(ox, oy, oz, 0), seed) >> vshift);\n" \
	"        float v1 = float(chaff_hash4(cell + ivec4(ox, oy, oz, step_px), seed) >> vshift);\n" \
	"        dw_c[i] = v1 - v0;\n" \
	"        d[i] = mix(v0, v1, u.w);\n" \
	"    }\n" \
	"    float norm = float((1u << (32u - vshift)) - 1u);\n" \
	"    float c0 = mix(d[0], d[4], u.z);\n" \
	"    float c1 = mix(d[1], d[5], u.z);\n" \
	"    float c2 = mix(d[2], d[6], u.z);\n" \
	"    float c3 = mix(d[3], d[7], u.z);\n" \
	"    float b0 = mix(c0, c2, u.y);\n" \
	"    float b1 = mix(c1, c3, u.y);\n" \
	"    float val = mix(b0, b1, u.x) / norm;\n" \
	"    float dN_dx = b1 - b0;\n" \
	"    float dN_dy = mix(c2 - c0, c3 - c1, u.x);\n" \
	"    float dz0 = mix(d[4] - d[0], d[6] - d[2], u.y);\n" \
	"    float dz1 = mix(d[5] - d[1], d[7] - d[3], u.y);\n" \
	"    float dN_dz = mix(dz0, dz1, u.x);\n" \
	"    float dw0 = mix(dw_c[0], dw_c[4], u.z);\n" \
	"    float dw1 = mix(dw_c[1], dw_c[5], u.z);\n" \
	"    float dw2 = mix(dw_c[2], dw_c[6], u.z);\n" \
	"    float dw3 = mix(dw_c[3], dw_c[7], u.z);\n" \
	"    float dw_y0 = mix(dw0, dw2, u.y);\n" \
	"    float dw_y1 = mix(dw1, dw3, u.y);\n" \
	"    float dN_dw = mix(dw_y0, dw_y1, u.x);\n" \
	"    float scale = 1.0 / (norm * cell_size);\n" \
	"    out_grad_w = (dN_dw * du.w) * scale;\n" \
	"    return vec4(val, (dN_dx * du.x) * scale, (dN_dy * du.y) * scale, (dN_dz * du.z) * scale);\n" \
	"}\n" \
	"vec4 chaff_sample_grad_4d(vec4 pos, uint shift, uint seed, out float out_grad_w) {\n" \
	"    return chaff_sample_grad_4d_ex(pos, shift, 24u, seed, out_grad_w);\n" \
	"}\n"

#define CHAFF_GLSL_EVAL_MULTI_FN \
	"vec2 chaff_rotate_2d(vec2 p, vec2 rot_cs) {\n" \
	"    return vec2(rot_cs.x * p.x - rot_cs.y * p.y, rot_cs.y * p.x + rot_cs.x * p.y);\n" \
	"}\n" \
	"vec2 chaff_rotate_golden(vec2 p) {\n" \
	"    return chaff_rotate_2d(p, vec2(0.8506508, 0.5257311));\n" \
	"}\n" \
	"float chaff_eval_multi_full(vec2 world_pos, uint seed, uint count, uint shifts[8], float amps[8],\n" \
	"                            float min_wavelength, uint vshift, uint seed_prime, uint seed_salt,\n" \
	"                            float warp_freq, uint warp_shift, float warp_amp, uint warp_salt, vec2 rot_cs) {\n" \
	"    uint vs = (vshift > 0u && vshift < 32u) ? vshift : 8u;\n" \
	"    float half_range = max(float(1u << (31u - vs)) / 65536.0, 1.0);\n" \
	"    uint prime = (seed_prime != 0u) ? seed_prime : 0x85EBCA77u;\n" \
	"    uint salt = (seed_salt != 0u) ? seed_salt : 1u;\n" \
	"    vec2 rcs = (rot_cs.x == 0.0 && rot_cs.y == 0.0) ? vec2(0.8506508, 0.5257311) : rot_cs;\n" \
	"    float total_val = 0.0;\n" \
	"    vec2 cur_p = world_pos;\n" \
	"    if (warp_amp > 0.0) {\n" \
	"        float wfreq = (warp_freq > 0.0) ? warp_freq : 0.0625;\n" \
	"        uint wshift = (warp_shift > 0u) ? warp_shift : 5u;\n" \
	"        uint wsalt = (warp_salt != 0u) ? warp_salt : 0x9E3779B9u;\n" \
	"        float warp_raw = chaff_sample_octave(world_pos * wfreq, wshift, vs, seed ^ wsalt);\n" \
	"        float warp_val = (warp_raw / 65536.0) - half_range;\n" \
	"        float d = warp_val * (warp_amp / half_range);\n" \
	"        cur_p += vec2(d, -d);\n" \
	"    }\n" \
	"    for (uint i = 0u; i < count && i < 8u; i++) {\n" \
	"        float wavelength = float(1u << shifts[i]);\n" \
	"        if (wavelength < min_wavelength) continue;\n" \
	"        uint oct_seed = seed ^ (i * prime + salt);\n" \
	"        float raw = chaff_sample_octave(cur_p, shifts[i], vs, oct_seed);\n" \
	"        float val = (raw / 65536.0) - half_range;\n" \
	"        total_val += val * (amps[i] / half_range);\n" \
	"        cur_p = chaff_rotate_2d(cur_p, rcs);\n" \
	"    }\n" \
	"    return total_val;\n" \
	"}\n" \
	"float chaff_eval_multi_ex(vec2 world_pos, uint seed, uint count, uint shifts[8], float amps[8],\n" \
	"                          float min_wavelength, float warp_freq, uint warp_shift, float warp_amp, uint warp_salt) {\n" \
	"    return chaff_eval_multi_full(world_pos, seed, count, shifts, amps, min_wavelength,\n" \
	"                                 8u, 0x85EBCA77u, 1u, warp_freq, warp_shift, warp_amp, warp_salt,\n" \
	"                                 vec2(0.8506508, 0.5257311));\n" \
	"}\n" \
	"float chaff_eval_multi(vec2 world_pos, uint seed, uint count, uint shifts[8], float amps[8], float min_wavelength) {\n" \
	"    return chaff_eval_multi_ex(world_pos, seed, count, shifts, amps, min_wavelength, 0.0625, 5u, 6.0, 0x9E3779B9u);\n" \
	"}\n" \
	"float chaff_eval_rugged(vec2 world_pos, uint seed, uint count, uint shifts[8], float amps[8], float min_wavelength) {\n" \
	"    return chaff_eval_multi(world_pos, seed, count, shifts, amps, min_wavelength);\n" \
	"}\n"

#define CHAFF_GLSL_FULLSCREEN_FRAG \
	CHAFF_GLSL_VERSION \
	"in vec2 v_uv;\n" \
	"out vec4 fragColor;\n" \
	"uniform vec2 u_origin;\n" \
	"uniform vec2 u_step;\n" \
	"uniform uint u_seed;\n" \
	"uniform uint u_count;\n" \
	"uniform uint u_shifts[8];\n" \
	"uniform float u_amps[8];\n" \
	"uniform float u_min_wavelength;\n" \
	"uniform uint u_vshift;\n" \
	"uniform uint u_seed_prime;\n" \
	"uniform uint u_seed_salt;\n" \
	"uniform uint u_warp_shift;\n" \
	"uniform uint u_warp_salt;\n" \
	"uniform float u_warp_freq;\n" \
	"uniform float u_warp_amp;\n" \
	"uniform vec2 u_rot_cs;\n" \
	CHAFF_GLSL_HASH_FN \
	CHAFF_GLSL_GRAD_NOISE_FN \
	CHAFF_GLSL_EVAL_FN \
	CHAFF_GLSL_EVAL_MULTI_FN \
	"void main() {\n" \
	"    vec2 world_pos = u_origin + gl_FragCoord.xy * u_step;\n" \
	"    float wamp = (u_warp_amp == 0.0) ? 6.0 : (u_warp_amp > 0.0 ? u_warp_amp : 0.0);\n" \
	"    float h = chaff_eval_multi_full(world_pos, u_seed, u_count, u_shifts, u_amps, u_min_wavelength,\n" \
	"                                    u_vshift, u_seed_prime, u_seed_salt,\n" \
	"                                    u_warp_freq, u_warp_shift, wamp, u_warp_salt, u_rot_cs);\n" \
	"    fragColor = vec4(h, h, h, 1.0);\n" \
	"}\n"

/* Declarative uniform buffer contract for client GPU pipelines */
typedef struct {
	float    origin[2];         /* Continuous world coordinate origin (wx, wy) */
	float    step[2];           /* Step size per pixel / texel */
	uint32_t seed;              /* Primary seed */
	uint32_t count;             /* Active octave count (up to 8) */
	uint32_t shifts[8];         /* Lattice size 2^shift per octave */
	float    amps[8];           /* Amplitude weight per octave */
	float    min_wavelength;    /* Nyquist cutoff filter: drop octaves with (1<<shift) < min_wavelength */
	uint32_t vshift;            /* Optional bitshift divider (0 = default 8) */
	uint32_t warp_shift;        /* Optional domain warp octave shift (0 = default 5) */
	uint32_t warp_salt;         /* Optional domain warp seed XOR salt (0 = default 0x9E3779B9u) */
	float    warp_freq;         /* Optional domain warp frequency scale (0.0f = default 0.0625f) */
	float    warp_amp;          /* Optional domain warp displacement amplitude (0.0f = default 6.0f, <0 = disable) */
	float    rot_cos;           /* Optional inter-octave rotation cos(theta) (0.0f = default golden 0.8506508f) */
	float    rot_sin;           /* Optional inter-octave rotation sin(theta) (0.0f = default golden 0.5257311f) */
	uint32_t seed_prime;        /* Optional per-octave seed prime (0 = default 0x85EBCA77u) */
	uint32_t seed_salt;         /* Optional per-octave seed salt (0 = default 1u) */
} chaff_gl_uniforms_t;

static inline const char *chaff_glsl_fragment_shader_source(void)
{
	return CHAFF_GLSL_FULLSCREEN_FRAG;
}

/* Reference C implementation of the exact GLSL shader logic for CPU-GPU parity verification */
static inline uint32_t chaff_glsl_ref_rotl32(uint32_t x, int r) {
	return (x << (uint32_t)r) | (x >> (uint32_t)(32 - r));
}

static inline uint32_t chaff_glsl_ref_hash2(int16_t y, int16_t x, uint32_t seed) {
	uint32_t val = (uint32_t)(uint16_t)y | ((uint32_t)(uint16_t)x << 16);
	uint32_t h = seed + 0x165667B1u + 4u;
	h += val * 0xC2B2AE3Du;
	h = chaff_glsl_ref_rotl32(h, 17) * 0x27D4EB2Fu;
	h ^= h >> 15;
	h *= 0x85EBCA77u;
	h ^= h >> 13;
	h *= 0xC2B2AE3Du;
	h ^= h >> 16;
	return h;
}

static inline float chaff_glsl_ref_sample_octave(float wx, float wy, unsigned shift, unsigned vshift, uint32_t seed) {
	float cell_size = (float)(1u << shift);
	float px = wx / cell_size;
	float py = wy / cell_size;
	int ix = (int)px - (px < 0 && px != (int)px ? 1 : 0);
	int iy = (int)py - (py < 0 && py != (int)py ? 1 : 0);
	float fx = px - (float)ix;
	float fy = py - (float)iy;
	float ux = fx * fx * (3.0f - 2.0f * fx);
	float uy = fy * fy * (3.0f - 2.0f * fy);
	int step_px = 1 << shift;
	int cx = ix * step_px;
	int cy = iy * step_px;
	uint32_t v00 = chaff_glsl_ref_hash2((int16_t)cy, (int16_t)cx, seed) >> vshift;
	uint32_t v10 = chaff_glsl_ref_hash2((int16_t)cy, (int16_t)(cx + step_px), seed) >> vshift;
	uint32_t v01 = chaff_glsl_ref_hash2((int16_t)(cy + step_px), (int16_t)cx, seed) >> vshift;
	uint32_t v11 = chaff_glsl_ref_hash2((int16_t)(cy + step_px), (int16_t)(cx + step_px), seed) >> vshift;
	float b0 = (float)v00 + ((float)v10 - (float)v00) * ux;
	float b1 = (float)v01 + ((float)v11 - (float)v01) * ux;
	return b0 + (b1 - b0) * uy;
}

static inline uint32_t chaff_glsl_ref_hash3(int16_t x, int16_t y, int16_t z, uint32_t seed) {
	uint32_t v1 = (uint32_t)(uint16_t)y | ((uint32_t)(uint16_t)x << 16);
	uint32_t v2 = (uint32_t)(uint16_t)z;
	uint32_t h = seed + 0x165667B1u + 6u;
	h += v1 * 0xC2B2AE3Du;
	h = chaff_glsl_ref_rotl32(h, 17) * 0x27D4EB2Fu;
	h += v2 * 0x85EBCA77u;
	h = chaff_glsl_ref_rotl32(h, 13) * 0xC2B2AE3Du;
	h ^= h >> 15;
	h *= 0x85EBCA77u;
	h ^= h >> 16;
	return h;
}

static inline float chaff_glsl_ref_sample_3d_ex(float wx, float wy, float wz, unsigned shift, unsigned vshift, uint32_t seed) {
	unsigned vs = (vshift > 0u && vshift < 32u) ? vshift : 24u;
	float norm = (float)((1ULL << (32u - vs)) - 1ULL);
	float cell_size = (float)(1u << shift);
	float px = wx / cell_size;
	float py = wy / cell_size;
	float pz = wz / cell_size;
	int ix = (int)px - (px < 0 && px != (int)px ? 1 : 0);
	int iy = (int)py - (py < 0 && py != (int)py ? 1 : 0);
	int iz = (int)pz - (pz < 0 && pz != (int)pz ? 1 : 0);
	float fx = px - (float)ix;
	float fy = py - (float)iy;
	float fz = pz - (float)iz;
	float ux = fx * fx * (3.0f - 2.0f * fx);
	float uy = fy * fy * (3.0f - 2.0f * fy);
	float uz = fz * fz * (3.0f - 2.0f * fz);
	int step_px = 1 << shift;
	int cx = ix * step_px;
	int cy = iy * step_px;
	int cz = iz * step_px;

	uint32_t v000 = chaff_glsl_ref_hash3((int16_t)cx, (int16_t)cy, (int16_t)cz, seed) >> vs;
	uint32_t v100 = chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)cy, (int16_t)cz, seed) >> vs;
	uint32_t v010 = chaff_glsl_ref_hash3((int16_t)cx, (int16_t)(cy + step_px), (int16_t)cz, seed) >> vs;
	uint32_t v110 = chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)(cy + step_px), (int16_t)cz, seed) >> vs;
	uint32_t v001 = chaff_glsl_ref_hash3((int16_t)cx, (int16_t)cy, (int16_t)(cz + step_px), seed) >> vs;
	uint32_t v101 = chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)cy, (int16_t)(cz + step_px), seed) >> vs;
	uint32_t v011 = chaff_glsl_ref_hash3((int16_t)cx, (int16_t)(cy + step_px), (int16_t)(cz + step_px), seed) >> vs;
	uint32_t v111 = chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)(cy + step_px), (int16_t)(cz + step_px), seed) >> vs;

	float b00 = (float)v000 + ((float)v100 - (float)v000) * ux;
	float b10 = (float)v010 + ((float)v110 - (float)v010) * ux;
	float b01 = (float)v001 + ((float)v101 - (float)v001) * ux;
	float b11 = (float)v011 + ((float)v111 - (float)v011) * ux;

	float b0 = b00 + (b10 - b00) * uy;
	float b1 = b01 + (b11 - b01) * uy;

	return (b0 + (b1 - b0) * uz) / norm;
}

static inline float chaff_glsl_ref_sample_3d(float wx, float wy, float wz, unsigned shift, uint32_t seed) {
	return chaff_glsl_ref_sample_3d_ex(wx, wy, wz, shift, 24u, seed);
}

typedef struct {
	float val;
	float grad_x;
	float grad_y;
	float grad_z;
} chaff_glsl_ref_vec4_t;

static inline chaff_glsl_ref_vec4_t chaff_glsl_ref_sample_grad_3d_ex(float wx, float wy, float wz, unsigned shift, unsigned vshift, uint32_t seed) {
	unsigned vs = (vshift > 0u && vshift < 32u) ? vshift : 24u;
	float norm = (float)((1ULL << (32u - vs)) - 1ULL);
	float cell_size = (float)(1u << shift);
	float px = wx / cell_size;
	float py = wy / cell_size;
	float pz = wz / cell_size;
	int ix = (int)px - (px < 0 && px != (int)px ? 1 : 0);
	int iy = (int)py - (py < 0 && py != (int)py ? 1 : 0);
	int iz = (int)pz - (pz < 0 && pz != (int)pz ? 1 : 0);
	float fx = px - (float)ix;
	float fy = py - (float)iy;
	float fz = pz - (float)iz;
	float ux = fx * fx * (3.0f - 2.0f * fx);
	float uy = fy * fy * (3.0f - 2.0f * fy);
	float uz = fz * fz * (3.0f - 2.0f * fz);
	float du_x = 6.0f * fx * (1.0f - fx);
	float du_y = 6.0f * fy * (1.0f - fy);
	float du_z = 6.0f * fz * (1.0f - fz);
	int step_px = 1 << shift;
	int cx = ix * step_px;
	int cy = iy * step_px;
	int cz = iz * step_px;

	float v000 = (float)(chaff_glsl_ref_hash3((int16_t)cx, (int16_t)cy, (int16_t)cz, seed) >> vs);
	float v100 = (float)(chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)cy, (int16_t)cz, seed) >> vs);
	float v010 = (float)(chaff_glsl_ref_hash3((int16_t)cx, (int16_t)(cy + step_px), (int16_t)cz, seed) >> vs);
	float v110 = (float)(chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)(cy + step_px), (int16_t)cz, seed) >> vs);
	float v001 = (float)(chaff_glsl_ref_hash3((int16_t)cx, (int16_t)cy, (int16_t)(cz + step_px), seed) >> vs);
	float v101 = (float)(chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)cy, (int16_t)(cz + step_px), seed) >> vs);
	float v011 = (float)(chaff_glsl_ref_hash3((int16_t)cx, (int16_t)(cy + step_px), (int16_t)(cz + step_px), seed) >> vs);
	float v111 = (float)(chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)(cy + step_px), (int16_t)(cz + step_px), seed) >> vs);

	float b00 = v000 + (v100 - v000) * ux;
	float b10 = v010 + (v110 - v010) * ux;
	float b01 = v001 + (v101 - v001) * ux;
	float b11 = v011 + (v111 - v011) * ux;

	float b0 = b00 + (b10 - b00) * uy;
	float b1 = b01 + (b11 - b01) * uy;

	float val = (b0 + (b1 - b0) * uz) / norm;

	float dx00 = v100 - v000;
	float dx10 = v110 - v010;
	float dx01 = v101 - v001;
	float dx11 = v111 - v011;
	float dx0 = dx00 + (dx10 - dx00) * uy;
	float dx1 = dx01 + (dx11 - dx01) * uy;
	float dN_dx = dx0 + (dx1 - dx0) * uz;

	float dy0 = b10 - b00;
	float dy1 = b11 - b01;
	float dN_dy = dy0 + (dy1 - dy0) * uz;

	float dN_dz = b1 - b0;

	chaff_glsl_ref_vec4_t out;
	out.val = val;
	out.grad_x = (dN_dx * du_x) / (norm * cell_size);
	out.grad_y = (dN_dy * du_y) / (norm * cell_size);
	out.grad_z = (dN_dz * du_z) / (norm * cell_size);
	return out;
}

static inline chaff_glsl_ref_vec4_t chaff_glsl_ref_sample_grad_3d(float wx, float wy, float wz, unsigned shift, uint32_t seed) {
	return chaff_glsl_ref_sample_grad_3d_ex(wx, wy, wz, shift, 24u, seed);
}

static inline uint32_t chaff_glsl_ref_hash4(int16_t x, int16_t y, int16_t z, int16_t w, uint32_t seed) {
	uint32_t v1 = (uint32_t)(uint16_t)y | ((uint32_t)(uint16_t)x << 16);
	uint32_t v2 = (uint32_t)(uint16_t)z | ((uint32_t)(uint16_t)w << 16);
	uint32_t h = seed + 0x165667B1u + 8u;
	h += v1 * 0xC2B2AE3Du;
	h = chaff_glsl_ref_rotl32(h, 17) * 0x27D4EB2Fu;
	h += v2 * 0xC2B2AE3Du;
	h = chaff_glsl_ref_rotl32(h, 17) * 0x27D4EB2Fu;
	h ^= h >> 15;
	h *= 0x85EBCA77u;
	h ^= h >> 13;
	h *= 0xC2B2AE3Du;
	h ^= h >> 16;
	return h;
}

static inline float chaff_glsl_ref_sample_4d_ex(float wx, float wy, float wz, float ww, unsigned shift, unsigned vshift, uint32_t seed) {
	unsigned vs = (vshift > 0u && vshift < 32u) ? vshift : 24u;
	float norm = (float)((1ULL << (32u - vs)) - 1ULL);
	float cell_size = (float)(1u << shift);
	float px = wx / cell_size;
	float py = wy / cell_size;
	float pz = wz / cell_size;
	float pw = ww / cell_size;
	int ix = (int)px - (px < 0 && px != (int)px ? 1 : 0);
	int iy = (int)py - (py < 0 && py != (int)py ? 1 : 0);
	int iz = (int)pz - (pz < 0 && pz != (int)pz ? 1 : 0);
	int iw = (int)pw - (pw < 0 && pw != (int)pw ? 1 : 0);
	float fx = px - (float)ix;
	float fy = py - (float)iy;
	float fz = pz - (float)iz;
	float fw = pw - (float)iw;
	float ux = fx * fx * (3.0f - 2.0f * fx);
	float uy = fy * fy * (3.0f - 2.0f * fy);
	float uz = fz * fz * (3.0f - 2.0f * fz);
	float uw = fw * fw * (3.0f - 2.0f * fw);
	int step_px = 1 << shift;
	int cx = ix * step_px;
	int cy = iy * step_px;
	int cz = iz * step_px;
	int cw = iw * step_px;

	float d[8];
	for (int i = 0; i < 8; i++) {
		int ox = (i & 1) ? step_px : 0;
		int oy = (i & 2) ? step_px : 0;
		int oz = (i & 4) ? step_px : 0;
		float v0 = (float)(chaff_glsl_ref_hash4((int16_t)(cx + ox), (int16_t)(cy + oy), (int16_t)(cz + oz), (int16_t)cw, seed) >> vs);
		float v1 = (float)(chaff_glsl_ref_hash4((int16_t)(cx + ox), (int16_t)(cy + oy), (int16_t)(cz + oz), (int16_t)(cw + step_px), seed) >> vs);
		d[i] = v0 + (v1 - v0) * uw;
	}

	float c0 = d[0] + (d[4] - d[0]) * uz;
	float c1 = d[1] + (d[5] - d[1]) * uz;
	float c2 = d[2] + (d[6] - d[2]) * uz;
	float c3 = d[3] + (d[7] - d[3]) * uz;

	float b0 = c0 + (c2 - c0) * uy;
	float b1 = c1 + (c3 - c1) * uy;

	return (b0 + (b1 - b0) * ux) / norm;
}

static inline float chaff_glsl_ref_sample_4d(float wx, float wy, float wz, float ww, unsigned shift, uint32_t seed) {
	return chaff_glsl_ref_sample_4d_ex(wx, wy, wz, ww, shift, 24u, seed);
}

typedef struct {
	float val;
	float grad_x;
	float grad_y;
	float grad_z;
	float grad_w;
} chaff_glsl_ref_vec5_t;

static inline chaff_glsl_ref_vec5_t chaff_glsl_ref_sample_grad_4d_ex(float wx, float wy, float wz, float ww, unsigned shift, unsigned vshift, uint32_t seed) {
	unsigned vs = (vshift > 0u && vshift < 32u) ? vshift : 24u;
	float norm = (float)((1ULL << (32u - vs)) - 1ULL);
	float cell_size = (float)(1u << shift);
	float px = wx / cell_size;
	float py = wy / cell_size;
	float pz = wz / cell_size;
	float pw = ww / cell_size;
	int ix = (int)px - (px < 0 && px != (int)px ? 1 : 0);
	int iy = (int)py - (py < 0 && py != (int)py ? 1 : 0);
	int iz = (int)pz - (pz < 0 && pz != (int)pz ? 1 : 0);
	int iw = (int)pw - (pw < 0 && pw != (int)pw ? 1 : 0);
	float fx = px - (float)ix;
	float fy = py - (float)iy;
	float fz = pz - (float)iz;
	float fw = pw - (float)iw;
	float ux = fx * fx * (3.0f - 2.0f * fx);
	float uy = fy * fy * (3.0f - 2.0f * fy);
	float uz = fz * fz * (3.0f - 2.0f * fz);
	float uw = fw * fw * (3.0f - 2.0f * fw);
	float du_x = 6.0f * fx * (1.0f - fx);
	float du_y = 6.0f * fy * (1.0f - fy);
	float du_z = 6.0f * fz * (1.0f - fz);
	float du_w = 6.0f * fw * (1.0f - fw);
	int step_px = 1 << shift;
	int cx = ix * step_px;
	int cy = iy * step_px;
	int cz = iz * step_px;
	int cw = iw * step_px;

	float d[8], dw_c[8];
	for (int i = 0; i < 8; i++) {
		int ox = (i & 1) ? step_px : 0;
		int oy = (i & 2) ? step_px : 0;
		int oz = (i & 4) ? step_px : 0;
		float v0 = (float)(chaff_glsl_ref_hash4((int16_t)(cx + ox), (int16_t)(cy + oy), (int16_t)(cz + oz), (int16_t)cw, seed) >> vs);
		float v1 = (float)(chaff_glsl_ref_hash4((int16_t)(cx + ox), (int16_t)(cy + oy), (int16_t)(cz + oz), (int16_t)(cw + step_px), seed) >> vs);
		dw_c[i] = v1 - v0;
		d[i] = v0 + (v1 - v0) * uw;
	}

	float c0 = d[0] + (d[4] - d[0]) * uz;
	float c1 = d[1] + (d[5] - d[1]) * uz;
	float c2 = d[2] + (d[6] - d[2]) * uz;
	float c3 = d[3] + (d[7] - d[3]) * uz;

	float b0 = c0 + (c2 - c0) * uy;
	float b1 = c1 + (c3 - c1) * uy;
	float val = (b0 + (b1 - b0) * ux) / norm;

	float dN_dx = b1 - b0;
	float dN_dy = (c2 - c0) + ((c3 - c1) - (c2 - c0)) * ux;
	float dz0 = (d[4] - d[0]) + ((d[6] - d[2]) - (d[4] - d[0])) * uy;
	float dz1 = (d[5] - d[1]) + ((d[7] - d[3]) - (d[5] - d[1])) * uy;
	float dN_dz = dz0 + (dz1 - dz0) * ux;

	float dw0 = dw_c[0] + (dw_c[4] - dw_c[0]) * uz;
	float dw1 = dw_c[1] + (dw_c[5] - dw_c[1]) * uz;
	float dw2 = dw_c[2] + (dw_c[6] - dw_c[2]) * uz;
	float dw3 = dw_c[3] + (dw_c[7] - dw_c[3]) * uz;
	float dw_y0 = dw0 + (dw2 - dw0) * uy;
	float dw_y1 = dw1 + (dw3 - dw1) * uy;
	float dN_dw = dw_y0 + (dw_y1 - dw_y0) * ux;

	float scale = 1.0f / (norm * cell_size);
	chaff_glsl_ref_vec5_t out;
	out.val = val;
	out.grad_x = (dN_dx * du_x) * scale;
	out.grad_y = (dN_dy * du_y) * scale;
	out.grad_z = (dN_dz * du_z) * scale;
	out.grad_w = (dN_dw * du_w) * scale;
	return out;
}

static inline chaff_glsl_ref_vec5_t chaff_glsl_ref_sample_grad_4d(float wx, float wy, float wz, float ww, unsigned shift, uint32_t seed) {
	return chaff_glsl_ref_sample_grad_4d_ex(wx, wy, wz, ww, shift, 24u, seed);
}

/* Reference C mirrors for CHAFF_GLSL_GRAD_NOISE_FN */
static inline void chaff_glsl_ref_grad_dir(int32_t cx, int32_t cy, uint32_t seed, float *out_gx, float *out_gy) {
	uint32_t h = chaff_glsl_ref_hash3((int16_t)cx, (int16_t)cy, 0, seed);
	float gx = (float)((int32_t)(h & 0xFFFFu) - 32768) * (1.0f / 32768.0f);
	float gy = (float)((int32_t)(h >> 16u) - 32768) * (1.0f / 32768.0f);
	float inv = 1.0f / sqrtf(gx * gx + gy * gy + 1e-6f);
	*out_gx = gx * inv;
	*out_gy = gy * inv;
}

static inline void chaff_glsl_ref_grad_noise2d_ex(float px, float py, uint32_t seed, float grad_norm,
                                                   float *out_val, float *out_gx, float *out_gy) {
	float gn = (grad_norm > 0.0f) ? grad_norm : CHAFF_GLSL_GRAD2D_NORM;
	float ipx = floorf(px);
	float ipy = floorf(py);
	float fpx = px - ipx;
	float fpy = py - ipy;
	float ux = fpx * fpx * fpx * (fpx * (fpx * 6.0f - 15.0f) + 10.0f);
	float uy = fpy * fpy * fpy * (fpy * (fpy * 6.0f - 15.0f) + 10.0f);
	float dux = 30.0f * fpx * fpx * (fpx * (fpx - 2.0f) + 1.0f);
	float duy = 30.0f * fpy * fpy * (fpy * (fpy - 2.0f) + 1.0f);
	int32_t cx = (int32_t)ipx;
	int32_t cy = (int32_t)ipy;

	float g00x, g00y, g10x, g10y, g01x, g01y, g11x, g11y;
	chaff_glsl_ref_grad_dir(cx,     cy,     seed, &g00x, &g00y);
	chaff_glsl_ref_grad_dir(cx + 1, cy,     seed, &g10x, &g10y);
	chaff_glsl_ref_grad_dir(cx,     cy + 1, seed, &g01x, &g01y);
	chaff_glsl_ref_grad_dir(cx + 1, cy + 1, seed, &g11x, &g11y);

	float v00 = g00x * fpx + g00y * fpy;
	float v10 = g10x * (fpx - 1.0f) + g10y * fpy;
	float v01 = g01x * fpx + g01y * (fpy - 1.0f);
	float v11 = g11x * (fpx - 1.0f) + g11y * (fpy - 1.0f);

	float k0 = v10 - v00;
	float k1 = v01 - v00;
	float k2 = v11 - v01 - k0;
	float val = 0.5f + gn * (v00 + k0 * ux + k1 * uy + k2 * ux * uy);
	if (val < 0.0f) val = 0.0f;
	if (val > 1.0f) val = 1.0f;

	float gix = g00x + (g10x - g00x) * ux + (g01x - g00x) * uy + (g11x - g01x - g10x + g00x) * (ux * uy);
	float giy = g00y + (g10y - g00y) * ux + (g01y - g00y) * uy + (g11y - g01y - g10y + g00y) * (ux * uy);

	if (out_val) *out_val = val;
	if (out_gx)  *out_gx  = gn * (gix + dux * (k0 + k2 * uy));
	if (out_gy)  *out_gy  = gn * (giy + duy * (k1 + k2 * ux));
}

static inline void chaff_glsl_ref_grad_noise2d(float px, float py, uint32_t seed,
                                               float *out_val, float *out_gx, float *out_gy) {
	chaff_glsl_ref_grad_noise2d_ex(px, py, seed, CHAFF_GLSL_GRAD2D_NORM, out_val, out_gx, out_gy);
}

static inline void chaff_glsl_ref_value_grad_noise2d(float px, float py, uint32_t seed,
                                                     float *out_val, float *out_gx, float *out_gy) {
	float ipx = floorf(px);
	float ipy = floorf(py);
	float fpx = px - ipx;
	float fpy = py - ipy;
	float ux = fpx * fpx * fpx * (fpx * (fpx * 6.0f - 15.0f) + 10.0f);
	float uy = fpy * fpy * fpy * (fpy * (fpy * 6.0f - 15.0f) + 10.0f);
	float dux = 30.0f * fpx * fpx * (fpx * (fpx - 2.0f) + 1.0f);
	float duy = 30.0f * fpy * fpy * (fpy * (fpy - 2.0f) + 1.0f);
	int16_t cx = (int16_t)ipx;
	int16_t cy = (int16_t)ipy;

	const float inv_norm = 1.0f / 16777215.0f;
	float v00 = (float)(chaff_glsl_ref_hash2(cy,              cx,              seed) >> 8u) * inv_norm;
	float v10 = (float)(chaff_glsl_ref_hash2(cy,              (int16_t)(cx+1), seed) >> 8u) * inv_norm;
	float v01 = (float)(chaff_glsl_ref_hash2((int16_t)(cy+1), cx,              seed) >> 8u) * inv_norm;
	float v11 = (float)(chaff_glsl_ref_hash2((int16_t)(cy+1), (int16_t)(cx+1), seed) >> 8u) * inv_norm;

	float k0 = v10 - v00;
	float k1 = v01 - v00;
	float k2 = v11 - v01 - k0;

	if (out_val) *out_val = v00 + k0 * ux + k1 * uy + k2 * ux * uy;
	if (out_gx)  *out_gx  = dux * (k0 + k2 * uy);
	if (out_gy)  *out_gy  = duy * (k1 + k2 * ux);
}

static inline float chaff_glsl_ref_lowpass_weight(float wavelength, float step_wu) {
	if (step_wu <= 0.0f) return 1.0f;
	float t = (wavelength - 2.0f * step_wu) / (2.0f * step_wu);
	if (t <= 0.0f) return 0.0f;
	if (t >= 1.0f) return 1.0f;
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static inline float chaff_glsl_ref_band_weight(float wavelength, float fine_step_wu, float coarse_step_wu) {
	float w_fine = chaff_glsl_ref_lowpass_weight(wavelength, fine_step_wu);
	if (w_fine <= 0.0f) return 0.0f;
	float w_coarse = chaff_glsl_ref_lowpass_weight(wavelength, coarse_step_wu);
	return w_fine * (1.0f - w_coarse);
}

static inline void chaff_glsl_ref_grad4(int32_t cx, int32_t cy, int32_t cz, int32_t cw,
                                        uint32_t seed, float out_g[4]) {
	uint32_t h = chaff_glsl_ref_hash4((int16_t)cx, (int16_t)cy, (int16_t)cz, (int16_t)cw, seed);
	float gx = (float)((int32_t)( h        & 0xFFu) - 128);
	float gy = (float)((int32_t)((h >> 8u)  & 0xFFu) - 128);
	float gz = (float)((int32_t)((h >> 16u) & 0xFFu) - 128);
	float gw = (float)((int32_t)((h >> 24u) & 0xFFu) - 128);
	float inv = 1.0f / sqrtf(gx * gx + gy * gy + gz * gz + gw * gw + 1e-6f);
	out_g[0] = gx * inv;
	out_g[1] = gy * inv;
	out_g[2] = gz * inv;
	out_g[3] = gw * inv;
}

static inline chaff_glsl_ref_vec4_t chaff_glsl_ref_grad_noise4d_ex(float px, float py, float pz, float pw, uint32_t seed, float grad_norm) {
	float gn = (grad_norm > 0.0f) ? grad_norm : CHAFF_GLSL_GRAD4D_NORM;
	float ipx = floorf(px), ipy = floorf(py), ipz = floorf(pz), ipw = floorf(pw);
	float fpx = px - ipx, fpy = py - ipy, fpz = pz - ipz, fpw = pw - ipw;
	float ux = fpx * fpx * fpx * (fpx * (fpx * 6.0f - 15.0f) + 10.0f);
	float uy = fpy * fpy * fpy * (fpy * (fpy * 6.0f - 15.0f) + 10.0f);
	float uz = fpz * fpz * fpz * (fpz * (fpz * 6.0f - 15.0f) + 10.0f);
	float uw = fpw * fpw * fpw * (fpw * (fpw * 6.0f - 15.0f) + 10.0f);
	float dux = 30.0f * fpx * fpx * (fpx * (fpx - 2.0f) + 1.0f);
	float duy = 30.0f * fpy * fpy * (fpy * (fpy - 2.0f) + 1.0f);
	float duz = 30.0f * fpz * fpz * (fpz * (fpz - 2.0f) + 1.0f);
	int32_t cx = (int32_t)ipx, cy = (int32_t)ipy, cz = (int32_t)ipz, cw = (int32_t)ipw;

	float d[8];
	for (int i = 0; i < 8; i++) {
		int ox = i & 1;
		int oy = (i >> 1) & 1;
		int oz = (i >> 2) & 1;
		float g0[4], g1[4];
		chaff_glsl_ref_grad4(cx + ox, cy + oy, cz + oz, cw,     seed, g0);
		chaff_glsl_ref_grad4(cx + ox, cy + oy, cz + oz, cw + 1, seed, g1);
		float v0 = g0[0] * (fpx - (float)ox) + g0[1] * (fpy - (float)oy) + g0[2] * (fpz - (float)oz) + g0[3] * fpw;
		float v1 = g1[0] * (fpx - (float)ox) + g1[1] * (fpy - (float)oy) + g1[2] * (fpz - (float)oz) + g1[3] * (fpw - 1.0f);
		d[i] = v0 + (v1 - v0) * uw;
	}

	float c0 = d[0] + (d[4] - d[0]) * uz;
	float c1 = d[1] + (d[5] - d[1]) * uz;
	float c2 = d[2] + (d[6] - d[2]) * uz;
	float c3 = d[3] + (d[7] - d[3]) * uz;
	float b0 = c0 + (c2 - c0) * uy;
	float b1 = c1 + (c3 - c1) * uy;
	float val = 0.5f + gn * (b0 + (b1 - b0) * ux);
	if (val < 0.0f) val = 0.0f;
	if (val > 1.0f) val = 1.0f;

	float dN_dx = (b1 - b0) * dux * gn;
	float dN_dy = ((c2 - c0) + ((c3 - c1) - (c2 - c0)) * ux) * duy * gn;
	float dz0 = (d[4] - d[0]) + ((d[6] - d[2]) - (d[4] - d[0])) * uy;
	float dz1 = (d[5] - d[1]) + ((d[7] - d[3]) - (d[5] - d[1])) * uy;
	float dN_dz = (dz0 + (dz1 - dz0) * ux) * duz * gn;

	chaff_glsl_ref_vec4_t out;
	out.val = val;
	out.grad_x = dN_dx;
	out.grad_y = dN_dy;
	out.grad_z = dN_dz;
	return out;
}

static inline chaff_glsl_ref_vec4_t chaff_glsl_ref_grad_noise4d(float px, float py, float pz, float pw, uint32_t seed) {
	return chaff_glsl_ref_grad_noise4d_ex(px, py, pz, pw, seed, CHAFF_GLSL_GRAD4D_NORM);
}

/* ============================================================================
 * Configurable Parameter Descriptors & Parameterized C Reference Evaluators
 * ============================================================================ */

typedef struct {
	int      max_octaves;       /**< Maximum FBM octaves to evaluate (default 4, max 8) */
	float    lacunarity;        /**< Frequency multiplier per octave (default 2.03) */
	float    gain;              /**< Amplitude multiplier per octave (default 0.50) */
	float    warp_strength;     /**< Domain warp displacement strength (default 0.32) */
	float    erosion_strength;  /**< Slope-driven analytical damping (default 0.28) */
	float    ridge_sharpness;   /**< Ridged multifractal folding blend (default 0.45) */
	uint32_t seed_stride;       /**< Per-octave golden-ratio hash salt (default 0x9E3779B9u) */
	float    rot_cos;           /**< Inter-octave rotation cos(theta) (0.0f = default 0.8f) */
	float    rot_sin;           /**< Inter-octave rotation sin(theta) (0.0f = default 0.6f) */
	float    nyquist_high;      /**< Smoothstep upper filter width cutoff (0.0f = default 0.55f) */
	float    nyquist_low;       /**< Smoothstep lower filter width cutoff (0.0f = default 0.18f) */
	float    ridge_octave_step; /**< Per-octave ridge ramp factor (0.0f = default CHAFF_GLSL_DEFAULT_RIDGE_OCTAVE_STEP = 0.25f) */
	float    grad_norm;         /**< 2D gradient noise normalization factor (0.0f = default CHAFF_GLSL_GRAD2D_NORM) */
} chaff_multi_warp_params_t;

#define CHAFF_MULTI_WARP_PARAMS_DEFAULT { \
	4, 2.03f, 0.50f, 0.32f, 0.28f, 0.45f, 0x9E3779B9u, 0.8f, 0.6f, 0.55f, 0.18f, \
	CHAFF_GLSL_DEFAULT_RIDGE_OCTAVE_STEP, CHAFF_GLSL_GRAD2D_NORM \
}

static inline float chaff_glsl_ref_eval_uniforms(
	const chaff_gl_uniforms_t *u, float frag_x, float frag_y)
{
	if (!u || u->count == 0)
		return 0.0f;

	unsigned vshift = (u->vshift > 0u && u->vshift < 32u) ? u->vshift : 8u;
	float half_range = (float)(1ULL << (31u - vshift)) / 65536.0f;
	if (half_range < 1.0f) half_range = 1.0f;

	float wx = u->origin[0] + frag_x * u->step[0];
	float wy = u->origin[1] + frag_y * u->step[1];
	float cur_px = wx;
	float cur_py = wy;

	float warp_amp = (u->warp_amp == 0.0f) ? 6.0f : (u->warp_amp > 0.0f ? u->warp_amp : 0.0f);
	if (warp_amp > 0.0f) {
		float wfreq = (u->warp_freq > 0.0f) ? u->warp_freq : 0.0625f;
		unsigned wshift = (u->warp_shift > 0u) ? u->warp_shift : 5u;
		uint32_t wsalt = (u->warp_salt != 0u) ? u->warp_salt : 0x9E3779B9u;
		float warp_raw = chaff_glsl_ref_sample_octave(wx * wfreq, wy * wfreq, wshift, vshift, u->seed ^ wsalt);
		float warp_val = (warp_raw / 65536.0f) - half_range;
		float d = warp_val * (warp_amp / half_range);
		cur_px += d;
		cur_py -= d;
	}

	float rcos = (u->rot_cos == 0.0f && u->rot_sin == 0.0f) ? 0.8506508f : u->rot_cos;
	float rsin = (u->rot_cos == 0.0f && u->rot_sin == 0.0f) ? 0.5257311f : u->rot_sin;
	uint32_t prime = (u->seed_prime != 0u) ? u->seed_prime : 0x85EBCA77u;
	uint32_t salt = (u->seed_salt != 0u) ? u->seed_salt : 1u;

	float total_val = 0.0f;
	uint32_t n = u->count < 8u ? u->count : 8u;
	for (uint32_t i = 0; i < n; i++) {
		float wavelength = (float)(1u << u->shifts[i]);
		if (wavelength < u->min_wavelength)
			continue;
		uint32_t oct_seed = u->seed ^ (i * prime + salt);
		float raw = chaff_glsl_ref_sample_octave(cur_px, cur_py, u->shifts[i], vshift, oct_seed);
		float val = (raw / 65536.0f) - half_range;
		total_val += val * (u->amps[i] / half_range);
		float nx = rcos * cur_px - rsin * cur_py;
		float ny = rsin * cur_px + rcos * cur_py;
		cur_px = nx;
		cur_py = ny;
	}
	return total_val;
}

static inline float chaff_glsl_ref_smoothstep(float e0, float e1, float x) {
	float t = (x - e0) / (e1 - e0);
	if (t < 0.0f) t = 0.0f;
	if (t > 1.0f) t = 1.0f;
	return t * t * (3.0f - 2.0f * t);
}

static inline void chaff_glsl_ref_curl_2d(float px, float py, uint32_t seed,
                                          float *out_vx, float *out_vy)
{
	float nv = 0.5f, gx = 0.0f, gy = 0.0f;
	chaff_glsl_ref_grad_noise2d(px, py, seed, &nv, &gx, &gy);
	if (out_vx) *out_vx = -gy;
	if (out_vy) *out_vy =  gx;
}

static inline void chaff_glsl_ref_eval_multi_warped_ex(
	const chaff_multi_warp_params_t *params,
	float wx, float wy, float base_shift, uint32_t seed, float texel_span,
	float *out_val, float *out_gx, float *out_gy)
{
	const chaff_multi_warp_params_t def = CHAFF_MULTI_WARP_PARAMS_DEFAULT;
	const chaff_multi_warp_params_t *mp = params ? params : &def;

	float lacunarity = (mp->lacunarity > 0.0f) ? mp->lacunarity : 2.03f;
	float gain       = (mp->gain > 0.0f)       ? mp->gain       : 0.50f;
	float rcos       = (mp->rot_cos == 0.0f && mp->rot_sin == 0.0f) ? 0.8f : mp->rot_cos;
	float rsin       = (mp->rot_cos == 0.0f && mp->rot_sin == 0.0f) ? 0.6f : mp->rot_sin;
	float nyq_hi     = (mp->nyquist_high > 0.0f) ? mp->nyquist_high : 0.55f;
	float nyq_lo     = (mp->nyquist_low > 0.0f)  ? mp->nyquist_low  : 0.18f;
	uint32_t stride  = (mp->seed_stride != 0u)   ? mp->seed_stride  : 0x9E3779B9u;
	float ridge_step = (mp->ridge_octave_step > 0.0f) ? mp->ridge_octave_step : CHAFF_GLSL_DEFAULT_RIDGE_OCTAVE_STEP;
	float grad_norm  = (mp->grad_norm > 0.0f)         ? mp->grad_norm         : CHAFF_GLSL_GRAD2D_NORM;

	float scale = exp2f(-base_shift);
	float px = wx * scale;
	float py = wy * scale;
	float base_fw = (texel_span * scale > 1e-5f) ? (texel_span * scale) : 1e-5f;

	float acc = 0.0f, wsum = 0.0f;
	float total_gx = 0.0f, total_gy = 0.0f;
	float slope_x = 0.0f, slope_y = 0.0f;
	float amp = 1.0f, freq = 1.0f;
	uint32_t s = seed;
	int octs = (mp->max_octaves > 0 && mp->max_octaves <= 8) ? mp->max_octaves : 4;

	for (int i = 0; i < octs; i++) {
		float oct_fw = base_fw * freq;
		float nyq = chaff_glsl_ref_smoothstep(nyq_hi, nyq_lo, oct_fw);
		if (nyq > 0.001f) {
			float qx = px * freq + slope_x * mp->warp_strength;
			float qy = py * freq + slope_y * mp->warp_strength;
			float nv = 0.5f, ngx = 0.0f, ngy = 0.0f;
			chaff_glsl_ref_grad_noise2d_ex(qx, qy, s, grad_norm, &nv, &ngx, &ngy);

			for (int r = 0; r < i; r++) {
				float rx = rcos * ngx - rsin * ngy;
				float ry = rsin * ngx + rcos * ngy;
				ngx = rx;
				ngy = ry;
			}

			slope_x += ngx * amp;
			slope_y += ngy * amp;
			float erosion = 1.0f / (1.0f + (slope_x * slope_x + slope_y * slope_y) * mp->erosion_strength);
			float ridge = 1.0f - fabsf(nv * 2.0f - 1.0f);
			float rblend = mp->ridge_sharpness * (float)i * ridge_step;
			float blended = nv * (1.0f - rblend) + (ridge * ridge) * rblend;

			float eff_amp = amp * erosion * nyq;
			acc      += blended * eff_amp;
			total_gx += ngx * (eff_amp * freq);
			total_gy += ngy * (eff_amp * freq);
			wsum     += eff_amp;
		}
		float rpx =  rcos * px + rsin * py;
		float rpy = -rsin * px + rcos * py;
		px = rpx;
		py = rpy;
		freq *= lacunarity;
		amp  *= gain;
		s    += stride;
	}

	float inv_w = (wsum > 1e-5f) ? (1.0f / wsum) : 1.0f;
	if (out_val) *out_val = (wsum > 1e-5f) ? (acc * inv_w) : 0.5f;
	if (out_gx)  *out_gx  = total_gx * inv_w * scale;
	if (out_gy)  *out_gy  = total_gy * inv_w * scale;
}

#endif /* TTYPT_CHAFF_GLSL_H */
