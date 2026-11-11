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

#define CHAFF_GLSL_VERSION \
	"#version 300 es\n" \
	"precision highp float;\n" \
	"precision highp int;\n"

/* Bit-exact XXH32 hash for 2D and 3D lattice coordinates in GLSL ES 3.0.
 * Memory layout matches int16_t va[2] = { y, x } and va[3] = { y, x, z } in little-endian. */
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
	"}\n"

/* Continuous cubic Hermite interpolation matching noise_wcorners and noise_wstep.
 * Evaluates corner vertices at cell = floor(world_pos / 2^shift) * 2^shift. */
#define CHAFF_GLSL_EVAL_FN \
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
	"    float b0 = mix(v00, v10, u.x);\n" \
	"    float b1 = mix(v01, v11, u.x);\n" \
	"    float val = mix(b0, b1, u.y) / 255.0;\n" \
	"    float dH_dx = mix(v10 - v00, v11 - v01, u.y);\n" \
	"    float dH_dy = b1 - b0;\n" \
	"    vec2 grad = (vec2(dH_dx, dH_dy) * du) / (255.0 * cell_size);\n" \
	"    return vec3(val, grad.x, grad.y);\n" \
	"}\n" \
	"float chaff_sample_3d(vec3 pos, uint shift, uint seed) {\n" \
	"    float cell_size = float(1u << shift);\n" \
	"    vec3 p = pos / cell_size;\n" \
	"    ivec3 i0 = ivec3(floor(p));\n" \
	"    vec3 f = fract(p);\n" \
	"    vec3 u = f * f * (3.0 - 2.0 * f);\n" \
	"    int step_px = int(1u << shift);\n" \
	"    ivec3 cell = i0 * step_px;\n" \
	"    uint v000 = chaff_hash3(cell + ivec3(0, 0, 0), seed) >> 24u;\n" \
	"    uint v100 = chaff_hash3(cell + ivec3(step_px, 0, 0), seed) >> 24u;\n" \
	"    uint v010 = chaff_hash3(cell + ivec3(0, step_px, 0), seed) >> 24u;\n" \
	"    uint v110 = chaff_hash3(cell + ivec3(step_px, step_px, 0), seed) >> 24u;\n" \
	"    uint v001 = chaff_hash3(cell + ivec3(0, 0, step_px), seed) >> 24u;\n" \
	"    uint v101 = chaff_hash3(cell + ivec3(step_px, 0, step_px), seed) >> 24u;\n" \
	"    uint v011 = chaff_hash3(cell + ivec3(0, step_px, step_px), seed) >> 24u;\n" \
	"    uint v111 = chaff_hash3(cell + ivec3(step_px, step_px, step_px), seed) >> 24u;\n" \
	"    float b00 = mix(float(v000), float(v100), u.x);\n" \
	"    float b10 = mix(float(v010), float(v110), u.x);\n" \
	"    float b01 = mix(float(v001), float(v101), u.x);\n" \
	"    float b11 = mix(float(v011), float(v111), u.x);\n" \
	"    float b0 = mix(b00, b10, u.y);\n" \
	"    float b1 = mix(b01, b11, u.y);\n" \
	"    return mix(b0, b1, u.z) / 255.0;\n" \
	"}\n" \
	"vec4 chaff_sample_grad_3d(vec3 pos, uint shift, uint seed) {\n" \
	"    float cell_size = float(1u << shift);\n" \
	"    vec3 p = pos / cell_size;\n" \
	"    ivec3 i0 = ivec3(floor(p));\n" \
	"    vec3 f = fract(p);\n" \
	"    vec3 u = f * f * (3.0 - 2.0 * f);\n" \
	"    vec3 du = 6.0 * f * (1.0 - f);\n" \
	"    int step_px = int(1u << shift);\n" \
	"    ivec3 cell = i0 * step_px;\n" \
	"    float v000 = float(chaff_hash3(cell + ivec3(0, 0, 0), seed) >> 24u);\n" \
	"    float v100 = float(chaff_hash3(cell + ivec3(step_px, 0, 0), seed) >> 24u);\n" \
	"    float v010 = float(chaff_hash3(cell + ivec3(0, step_px, 0), seed) >> 24u);\n" \
	"    float v110 = float(chaff_hash3(cell + ivec3(step_px, step_px, 0), seed) >> 24u);\n" \
	"    float v001 = float(chaff_hash3(cell + ivec3(0, 0, step_px), seed) >> 24u);\n" \
	"    float v101 = float(chaff_hash3(cell + ivec3(step_px, 0, step_px), seed) >> 24u);\n" \
	"    float v011 = float(chaff_hash3(cell + ivec3(0, step_px, step_px), seed) >> 24u);\n" \
	"    float v111 = float(chaff_hash3(cell + ivec3(step_px, step_px, step_px), seed) >> 24u);\n" \
	"    float b00 = mix(v000, v100, u.x);\n" \
	"    float b10 = mix(v010, v110, u.x);\n" \
	"    float b01 = mix(v001, v101, u.x);\n" \
	"    float b11 = mix(v011, v111, u.x);\n" \
	"    float b0 = mix(b00, b10, u.y);\n" \
	"    float b1 = mix(b01, b11, u.y);\n" \
	"    float val = mix(b0, b1, u.z) / 255.0;\n" \
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
	"    vec3 grad = (vec3(dN_dx, dN_dy, dN_dz) * du) / (255.0 * cell_size);\n" \
	"    return vec4(val, grad.x, grad.y, grad.z);\n" \
	"}\n"

#define CHAFF_GLSL_EVAL_MULTI_FN \
	"vec2 chaff_rotate_golden(vec2 p) {\n" \
	"    return vec2(0.8506508 * p.x - 0.5257311 * p.y, 0.5257311 * p.x + 0.8506508 * p.y);\n" \
	"}\n" \
	"float chaff_eval_multi(vec2 world_pos, uint seed, uint count, uint shifts[8], float amps[8], float min_wavelength) {\n" \
	"    float total_elevation = 0.0;\n" \
	"    float warp_raw = chaff_sample_octave(world_pos * 0.0625, 5u, 8u, seed ^ 0x57415250u);\n" \
	"    float warp_val = (warp_raw / 65536.0) - 128.0;\n" \
	"    vec2 cur_p = world_pos + vec2(warp_val * (6.0 / 128.0), -warp_val * (6.0 / 128.0));\n" \
	"    for (uint i = 0u; i < count && i < 8u; i++) {\n" \
	"        float wavelength = float(1u << shifts[i]);\n" \
	"        if (wavelength < min_wavelength) continue;\n" \
	"        uint oct_seed = seed ^ (i * 0x85EBCA77u + 1u);\n" \
	"        float raw = chaff_sample_octave(cur_p, shifts[i], 8u, oct_seed);\n" \
	"        float val = (raw / 65536.0) - 128.0;\n" \
	"        total_elevation += val * (amps[i] / 128.0);\n" \
	"        cur_p = chaff_rotate_golden(cur_p);\n" \
	"    }\n" \
	"    return total_elevation;\n" \
	"}\n" \
	"float chaff_eval_rugged(vec2 world_pos, uint seed, uint count, uint shifts[8], float amps[8], float min_wavelength) {\n" \
	"    return chaff_eval_multi(world_pos, seed, count, shifts, amps, min_wavelength);\n" \
	"}\n"

#define CHAFF_GLSL_WATER_FN \
	"vec4 chaff_water_waves_glsl(vec2 world_pos, float anim_time) {\n" \
	"    vec2 wind = vec2(0.7071, -0.7071);\n" \
	"    vec2 pMacro = world_pos * 0.035 - wind * (anim_time * 0.8);\n" \
	"    vec2 pMid   = chaff_rotate_golden(world_pos * 0.12) + vec2(0.5, 0.866) * (anim_time * 1.6);\n" \
	"    vec2 pCap   = world_pos * 0.45 - wind * (anim_time * 3.5);\n" \
	"    vec4 sMacro = chaff_sample_grad_3d(vec3(pMacro * 8.0, anim_time * 1.2), 6u, 0x5357u);\n" \
	"    vec4 sMid   = chaff_sample_grad_3d(vec3(pMid * 8.0, anim_time * 2.2), 4u, 0xCAFEu);\n" \
	"    vec3 sCap   = chaff_sample_grad_2d(pCap, 2u, 24u, 0xC491u);\n" \
	"    vec2 waveGrad = sMacro.yz * 4.32 + sMid.yz * 4.48 + sCap.yz * 0.42;\n" \
	"    vec3 waveNormal = normalize(vec3(-waveGrad.x, -waveGrad.y, 1.0));\n" \
	"    float waveHeight = sMacro.x * 2.0 + sMid.x * 0.8 + sCap.x * 0.2;\n" \
	"    return vec4(waveHeight, waveNormal.x, waveNormal.y, waveNormal.z);\n" \
	"}\n" \
	"float chaff_caustics_glsl(vec2 world_pos, float anim_time) {\n" \
	"    float c1 = chaff_sample_octave(world_pos * 0.5 + vec2(anim_time * 0.2), 3u, 24u, 0xCA11u);\n" \
	"    float c2 = chaff_sample_octave(world_pos * 0.5 - vec2(anim_time * 0.15), 3u, 24u, 0xCA22u);\n" \
	"    return pow(min(c1, c2) / 255.0, 2.0) * 2.5;\n" \
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
	CHAFF_GLSL_HASH_FN \
	CHAFF_GLSL_EVAL_FN \
	CHAFF_GLSL_EVAL_MULTI_FN \
	CHAFF_GLSL_WATER_FN \
	"void main() {\n" \
	"    vec2 world_pos = u_origin + gl_FragCoord.xy * u_step;\n" \
	"    float h = chaff_eval_multi(world_pos, u_seed, u_count, u_shifts, u_amps, u_min_wavelength);\n" \
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

static inline float chaff_glsl_ref_sample_3d(float wx, float wy, float wz, unsigned shift, uint32_t seed) {
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

	uint32_t v000 = chaff_glsl_ref_hash3((int16_t)cx, (int16_t)cy, (int16_t)cz, seed) >> 24u;
	uint32_t v100 = chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)cy, (int16_t)cz, seed) >> 24u;
	uint32_t v010 = chaff_glsl_ref_hash3((int16_t)cx, (int16_t)(cy + step_px), (int16_t)cz, seed) >> 24u;
	uint32_t v110 = chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)(cy + step_px), (int16_t)cz, seed) >> 24u;
	uint32_t v001 = chaff_glsl_ref_hash3((int16_t)cx, (int16_t)cy, (int16_t)(cz + step_px), seed) >> 24u;
	uint32_t v101 = chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)cy, (int16_t)(cz + step_px), seed) >> 24u;
	uint32_t v011 = chaff_glsl_ref_hash3((int16_t)cx, (int16_t)(cy + step_px), (int16_t)(cz + step_px), seed) >> 24u;
	uint32_t v111 = chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)(cy + step_px), (int16_t)(cz + step_px), seed) >> 24u;

	float b00 = (float)v000 + ((float)v100 - (float)v000) * ux;
	float b10 = (float)v010 + ((float)v110 - (float)v010) * ux;
	float b01 = (float)v001 + ((float)v101 - (float)v001) * ux;
	float b11 = (float)v011 + ((float)v111 - (float)v011) * ux;

	float b0 = b00 + (b10 - b00) * uy;
	float b1 = b01 + (b11 - b01) * uy;

	return (b0 + (b1 - b0) * uz) / 255.0f;
}

typedef struct {
	float val;
	float grad_x;
	float grad_y;
	float grad_z;
} chaff_glsl_ref_vec4_t;

static inline chaff_glsl_ref_vec4_t chaff_glsl_ref_sample_grad_3d(float wx, float wy, float wz, unsigned shift, uint32_t seed) {
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

	float v000 = (float)(chaff_glsl_ref_hash3((int16_t)cx, (int16_t)cy, (int16_t)cz, seed) >> 24u);
	float v100 = (float)(chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)cy, (int16_t)cz, seed) >> 24u);
	float v010 = (float)(chaff_glsl_ref_hash3((int16_t)cx, (int16_t)(cy + step_px), (int16_t)cz, seed) >> 24u);
	float v110 = (float)(chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)(cy + step_px), (int16_t)cz, seed) >> 24u);
	float v001 = (float)(chaff_glsl_ref_hash3((int16_t)cx, (int16_t)cy, (int16_t)(cz + step_px), seed) >> 24u);
	float v101 = (float)(chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)cy, (int16_t)(cz + step_px), seed) >> 24u);
	float v011 = (float)(chaff_glsl_ref_hash3((int16_t)cx, (int16_t)(cy + step_px), (int16_t)(cz + step_px), seed) >> 24u);
	float v111 = (float)(chaff_glsl_ref_hash3((int16_t)(cx + step_px), (int16_t)(cy + step_px), (int16_t)(cz + step_px), seed) >> 24u);

	float b00 = v000 + (v100 - v000) * ux;
	float b10 = v010 + (v110 - v010) * ux;
	float b01 = v001 + (v101 - v001) * ux;
	float b11 = v011 + (v111 - v011) * ux;

	float b0 = b00 + (b10 - b00) * uy;
	float b1 = b01 + (b11 - b01) * uy;

	float val = (b0 + (b1 - b0) * uz) / 255.0f;

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
	out.grad_x = (dN_dx * du_x) / (255.0f * cell_size);
	out.grad_y = (dN_dy * du_y) / (255.0f * cell_size);
	out.grad_z = (dN_dz * du_z) / (255.0f * cell_size);
	return out;
}

#endif /* TTYPT_CHAFF_GLSL_H */
