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
	"vec3 chaff_grad_noise2d(vec2 p, uint seed) {\n" \
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
	"    float val = 0.5 + 0.70710678 * (v00 + k0 * u.x + k1 * u.y + k2 * u.x * u.y);\n" \
	"    vec2 gInterp = g00 + (g10 - g00) * u.x + (g01 - g00) * u.y + (g11 - g01 - g10 + g00) * (u.x * u.y);\n" \
	"    vec2 grad = 0.70710678 * (gInterp + du * vec2(k0 + k2 * u.y, k1 + k2 * u.x));\n" \
	"    return vec3(clamp(val, 0.0, 1.0), grad.x, grad.y);\n" \
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
	"vec4 chaff_grad_noise4d(vec4 p, uint seed) {\n" \
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
	"    float val = clamp(0.5 + 0.72 * mix(b0, b1, u.x), 0.0, 1.0);\n" \
	"    float dN_dx = (b1 - b0) * du.x * 0.72;\n" \
	"    float dN_dy = mix(c2 - c0, c3 - c1, u.x) * du.y * 0.72;\n" \
	"    float dz0 = mix(d4 - d0, d6 - d2, u.y);\n" \
	"    float dz1 = mix(d5 - d1, d7 - d3, u.y);\n" \
	"    float dN_dz = mix(dz0, dz1, u.x) * du.z * 0.72;\n" \
	"    return vec4(val, dN_dx, dN_dy, dN_dz);\n" \
	"}\n"

/* Canonical 3-octave divergence-free recursive curl-flow field in GLSL.
 * Evaluates synoptic flow -> meso gusts -> micro eddies with optional Nyquist/distance band weights. */
#define CHAFF_GLSL_FLOW_FN \
	"vec4 chaff_eval_flow_2d_banded(vec2 worldPos, float anim_time, vec3 band_weights) {\n" \
	"    const mat2 rot  = mat2(0.8, -0.6, 0.6, 0.8);\n" \
	"    const mat2 irot = mat2(0.8,  0.6, -0.6, 0.8);\n" \
	"    const vec2 dir0 = vec2(0.741, -0.671);\n" \
	"    const vec2 dir1 = vec2(0.822, -0.569);\n" \
	"    const vec2 dir2 = vec2(0.640, -0.768);\n" \
	"    float w0_nyq = band_weights.x;\n" \
	"    float w1_nyq = band_weights.y;\n" \
	"    float w2_nyq = band_weights.z;\n" \
	"    if (w0_nyq <= 0.001) return vec4(dir0.x, dir0.y, 0.65, 0.0);\n" \
	"    float t0 = mod(anim_time * 0.05, 1024.0);\n" \
	"    vec2 p0 = worldPos * 0.0055 - dir0 * t0;\n" \
	"    vec3 w0 = chaff_grad_noise2d(p0, 0x57494E44u);\n" \
	"    vec2 baseWind = normalize(dir0 + vec2(-w0.z, w0.y) * (0.28 * w0_nyq));\n" \
	"    vec3 w1 = vec3(0.5, 0.0, 0.0);\n" \
	"    vec2 localWind1 = baseWind;\n" \
	"    if (w1_nyq > 0.001) {\n" \
	"        float t1 = mod(anim_time * 0.12, 1024.0);\n" \
	"        vec2 p1 = rot * (worldPos * 0.028 + w0.yz * 0.65) - dir1 * t1;\n" \
	"        w1 = chaff_grad_noise2d(p1, 0x47555354u);\n" \
	"        vec2 mesoGrad = irot * w1.yz;\n" \
	"        localWind1 = normalize(baseWind + vec2(-mesoGrad.y, mesoGrad.x) * (0.22 * w1_nyq));\n" \
	"    }\n" \
	"    vec3 w2 = vec3(0.5, 0.0, 0.0);\n" \
	"    vec2 windDir = localWind1;\n" \
	"    if (w2_nyq > 0.001) {\n" \
	"        float t2 = mod(anim_time * 0.22, 1024.0);\n" \
	"        vec2 p2 = rot * (rot * (worldPos * 0.120) + w0.yz * 0.45 + w1.yz * 0.55) - dir2 * t2;\n" \
	"        w2 = chaff_grad_noise2d(p2, 0x45444459u);\n" \
	"        vec2 microGrad = (irot * irot) * w2.yz;\n" \
	"        windDir = normalize(localWind1 + vec2(-microGrad.y, microGrad.x) * (0.16 * w2_nyq));\n" \
	"    }\n" \
	"    float gustEnv = clamp(0.65 + 0.45 * (w0.x - 0.5) * w0_nyq + 0.55 * (w1.x - 0.38) * w1_nyq + 0.30 * (w2.x - 0.42) * w2_nyq, 0.25, 1.35);\n" \
	"    float catspaw = smoothstep(0.38, 0.76, w1.x * 0.62 + w2.x * 0.38) * w1_nyq;\n" \
	"    return vec4(windDir.x, windDir.y, gustEnv, catspaw);\n" \
	"}\n" \
	"vec4 chaff_eval_flow_2d(vec2 worldPos, float anim_time) {\n" \
	"    return chaff_eval_flow_2d_banded(worldPos, anim_time, vec3(1.0, 1.0, 1.0));\n" \
	"}\n" \
	"/* Synoptic & Regional Geostrophic Curl Flow: planetary Rossby wave streamfunction psi0 (lambda~2200km)\n" \
	" * and regional jet psi1 (lambda~180km) driving continental cyclone commas and frontal convergence ridges. */\n" \
	"vec4 chaff_eval_synoptic_flow_2d(vec2 worldPos, float anim_time, float fw) {\n" \
	"    const vec2 prevDir = vec2(0.741, -0.671);\n" \
	"    const vec2 prevPerp = vec2(0.671, 0.741);\n" \
	"    float tFlow = mod(anim_time * 0.005, 1024.0);\n" \
	"    vec3 psi0 = chaff_grad_noise2d(worldPos * 0.00000042 - prevDir * (tFlow * 0.40), 0x4359434Cu);\n" \
	"    vec3 psi1 = chaff_grad_noise2d(worldPos * 0.00000180 + vec2(-psi0.z, psi0.y) * 0.30 - prevDir * tFlow, 0x5354524Du);\n" \
	"    vec2 synWarp = vec2(-psi0.z, psi0.y) * 260000.0 + vec2(-psi1.z, psi1.y) * 68000.0;\n" \
	"    vec2 vSyn = normalize(prevDir + vec2(-psi0.z, psi0.y) * 0.48 + vec2(-psi1.z, psi1.y) * 0.28);\n" \
	"    float convEnv = clamp(0.50 + 0.40 * (psi0.x - 0.50) + 0.30 * (psi1.x - 0.50), 0.0, 1.0);\n" \
	"    return vec4(vSyn.x, vSyn.y, convEnv, synWarp.x);\n" \
	"}\n" \
	"/* 3D Atmospheric Wind & Ekman Spiral Shear: evaluates horizontal wind direction with altitude veering,\n" \
	" * vertical wind-shear displacement vector S_shear(z), gust envelope, and vertical updraft velocity. */\n" \
	"void chaff_eval_wind_3d(vec3 worldPos, float zBase, float zTop, float anim_time, float fw,\n" \
	"                        out vec2 outWindDir, out vec2 outShearDisp, out float outGust, out float outUpdraft) {\n" \
	"    float spanZ = max(16.0, zTop - zBase);\n" \
	"    float h = clamp((worldPos.z - zBase) / spanZ, 0.0, 1.0);\n" \
	"    vec4 synFlow = chaff_eval_synoptic_flow_2d(worldPos.xy, anim_time, fw);\n" \
	"    float w0_sfc = smoothstep(0.45, 0.12, 0.0055 * fw);\n" \
	"    float w1_sfc = smoothstep(0.45, 0.12, 0.0280 * fw);\n" \
	"    float w2_sfc = smoothstep(0.45, 0.12, 0.1200 * fw);\n" \
	"    vec4 sfcFlow = chaff_eval_flow_2d_banded(worldPos.xy, anim_time, vec3(w0_sfc, w1_sfc, w2_sfc));\n" \
	"    /* Ekman spiral: veers clockwise up to 28 degrees with altitude, speed increases with h^1.35 */\n" \
	"    float veerAngle = h * 0.488;\n" \
	"    float cosV = cos(veerAngle), sinV = sin(veerAngle);\n" \
	"    vec2 baseDir = normalize(mix(sfcFlow.xy, synFlow.xy, 0.35 + 0.45 * h));\n" \
	"    vec2 atmoDir = vec2(baseDir.x * cosV - baseDir.y * sinV, baseDir.x * sinV + baseDir.y * cosV);\n" \
	"    float atmoSpeed = 1.0 + 0.85 * pow(h, 1.35);\n" \
	"    outWindDir = atmoDir;\n" \
	"    outShearDisp = atmoDir * (pow(h, 1.25) * spanZ * 0.35) * atmoSpeed;\n" \
	"    outGust = clamp(mix(sfcFlow.z, 1.0 + 0.50 * (synFlow.z - 0.5), h), 0.25, 1.75);\n" \
	"    outUpdraft = clamp(0.45 * synFlow.z + 0.35 * sfcFlow.z + 0.20 * sfcFlow.w, 0.10, 1.50);\n" \
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
	"}\n" \
	"float chaff_sample_4d(vec4 pos, uint shift, uint seed) {\n" \
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
	"        float v0 = float(chaff_hash4(cell + ivec4(ox, oy, oz, 0), seed) >> 24u);\n" \
	"        float v1 = float(chaff_hash4(cell + ivec4(ox, oy, oz, step_px), seed) >> 24u);\n" \
	"        d[i] = mix(v0, v1, u.w);\n" \
	"    }\n" \
	"    float c0 = mix(d[0], d[4], u.z);\n" \
	"    float c1 = mix(d[1], d[5], u.z);\n" \
	"    float c2 = mix(d[2], d[6], u.z);\n" \
	"    float c3 = mix(d[3], d[7], u.z);\n" \
	"    float b0 = mix(c0, c2, u.y);\n" \
	"    float b1 = mix(c1, c3, u.y);\n" \
	"    return mix(b0, b1, u.x) / 255.0;\n" \
	"}\n" \
	"vec4 chaff_sample_grad_4d(vec4 pos, uint shift, uint seed, out float out_grad_w) {\n" \
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
	"        float v0 = float(chaff_hash4(cell + ivec4(ox, oy, oz, 0), seed) >> 24u);\n" \
	"        float v1 = float(chaff_hash4(cell + ivec4(ox, oy, oz, step_px), seed) >> 24u);\n" \
	"        dw_c[i] = v1 - v0;\n" \
	"        d[i] = mix(v0, v1, u.w);\n" \
	"    }\n" \
	"    float c0 = mix(d[0], d[4], u.z);\n" \
	"    float c1 = mix(d[1], d[5], u.z);\n" \
	"    float c2 = mix(d[2], d[6], u.z);\n" \
	"    float c3 = mix(d[3], d[7], u.z);\n" \
	"    float b0 = mix(c0, c2, u.y);\n" \
	"    float b1 = mix(c1, c3, u.y);\n" \
	"    float val = mix(b0, b1, u.x) / 255.0;\n" \
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
	"    float scale = 1.0 / (255.0 * cell_size);\n" \
	"    out_grad_w = (dN_dw * du.w) * scale;\n" \
	"    return vec4(val, (dN_dx * du.x) * scale, (dN_dy * du.y) * scale, (dN_dz * du.z) * scale);\n" \
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
	CHAFF_GLSL_GRAD_NOISE_FN \
	CHAFF_GLSL_FLOW_FN \
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

static inline float chaff_glsl_ref_sample_4d(float wx, float wy, float wz, float ww, unsigned shift, uint32_t seed) {
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
		float v0 = (float)(chaff_glsl_ref_hash4((int16_t)(cx + ox), (int16_t)(cy + oy), (int16_t)(cz + oz), (int16_t)cw, seed) >> 24u);
		float v1 = (float)(chaff_glsl_ref_hash4((int16_t)(cx + ox), (int16_t)(cy + oy), (int16_t)(cz + oz), (int16_t)(cw + step_px), seed) >> 24u);
		d[i] = v0 + (v1 - v0) * uw;
	}

	float c0 = d[0] + (d[4] - d[0]) * uz;
	float c1 = d[1] + (d[5] - d[1]) * uz;
	float c2 = d[2] + (d[6] - d[2]) * uz;
	float c3 = d[3] + (d[7] - d[3]) * uz;

	float b0 = c0 + (c2 - c0) * uy;
	float b1 = c1 + (c3 - c1) * uy;

	return (b0 + (b1 - b0) * ux) / 255.0f;
}

typedef struct {
	float val;
	float grad_x;
	float grad_y;
	float grad_z;
	float grad_w;
} chaff_glsl_ref_vec5_t;

static inline chaff_glsl_ref_vec5_t chaff_glsl_ref_sample_grad_4d(float wx, float wy, float wz, float ww, unsigned shift, uint32_t seed) {
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
		float v0 = (float)(chaff_glsl_ref_hash4((int16_t)(cx + ox), (int16_t)(cy + oy), (int16_t)(cz + oz), (int16_t)cw, seed) >> 24u);
		float v1 = (float)(chaff_glsl_ref_hash4((int16_t)(cx + ox), (int16_t)(cy + oy), (int16_t)(cz + oz), (int16_t)(cw + step_px), seed) >> 24u);
		dw_c[i] = v1 - v0;
		d[i] = v0 + (v1 - v0) * uw;
	}

	float c0 = d[0] + (d[4] - d[0]) * uz;
	float c1 = d[1] + (d[5] - d[1]) * uz;
	float c2 = d[2] + (d[6] - d[2]) * uz;
	float c3 = d[3] + (d[7] - d[3]) * uz;

	float b0 = c0 + (c2 - c0) * uy;
	float b1 = c1 + (c3 - c1) * uy;
	float val = (b0 + (b1 - b0) * ux) / 255.0f;

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

	float scale = 1.0f / (255.0f * cell_size);
	chaff_glsl_ref_vec5_t out;
	out.val = val;
	out.grad_x = (dN_dx * du_x) * scale;
	out.grad_y = (dN_dy * du_y) * scale;
	out.grad_z = (dN_dz * du_z) * scale;
	out.grad_w = (dN_dw * du_w) * scale;
	return out;
}

/* Reference C mirrors for CHAFF_GLSL_GRAD_NOISE_FN and CHAFF_GLSL_FLOW_FN */
static inline void chaff_glsl_ref_grad_dir(int32_t cx, int32_t cy, uint32_t seed, float *out_gx, float *out_gy) {
	uint32_t h = chaff_glsl_ref_hash3((int16_t)cx, (int16_t)cy, 0, seed);
	float gx = (float)((int32_t)(h & 0xFFFFu) - 32768) * (1.0f / 32768.0f);
	float gy = (float)((int32_t)(h >> 16u) - 32768) * (1.0f / 32768.0f);
	float inv = 1.0f / sqrtf(gx * gx + gy * gy + 1e-6f);
	*out_gx = gx * inv;
	*out_gy = gy * inv;
}

static inline void chaff_glsl_ref_grad_noise2d(float px, float py, uint32_t seed,
                                               float *out_val, float *out_gx, float *out_gy) {
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
	float val = 0.5f + 0.70710678f * (v00 + k0 * ux + k1 * uy + k2 * ux * uy);
	if (val < 0.0f) val = 0.0f;
	if (val > 1.0f) val = 1.0f;

	float gix = g00x + (g10x - g00x) * ux + (g01x - g00x) * uy + (g11x - g01x - g10x + g00x) * (ux * uy);
	float giy = g00y + (g10y - g00y) * ux + (g01y - g00y) * uy + (g11y - g01y - g10y + g00y) * (ux * uy);

	if (out_val) *out_val = val;
	if (out_gx)  *out_gx  = 0.70710678f * (gix + dux * (k0 + k2 * uy));
	if (out_gy)  *out_gy  = 0.70710678f * (giy + duy * (k1 + k2 * ux));
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

static inline chaff_glsl_ref_vec4_t chaff_glsl_ref_grad_noise4d(float px, float py, float pz, float pw, uint32_t seed) {
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
	float val = 0.5f + 0.72f * (b0 + (b1 - b0) * ux);
	if (val < 0.0f) val = 0.0f;
	if (val > 1.0f) val = 1.0f;

	float dN_dx = (b1 - b0) * dux * 0.72f;
	float dN_dy = ((c2 - c0) + ((c3 - c1) - (c2 - c0)) * ux) * duy * 0.72f;
	float dz0 = (d[4] - d[0]) + ((d[6] - d[2]) - (d[4] - d[0])) * uy;
	float dz1 = (d[5] - d[1]) + ((d[7] - d[3]) - (d[5] - d[1])) * uy;
	float dN_dz = (dz0 + (dz1 - dz0) * ux) * duz * 0.72f;

	chaff_glsl_ref_vec4_t out;
	out.val = val;
	out.grad_x = dN_dx;
	out.grad_y = dN_dy;
	out.grad_z = dN_dz;
	return out;
}

static inline void chaff_glsl_ref_eval_flow_2d(float wx, float wy, float anim_time,
                                               float *out_dir_x, float *out_dir_y,
                                               float *out_gust, float *out_catspaw) {
	float raw_t0 = anim_time * 0.05f;
	float t0 = raw_t0 - 1024.0f * floorf(raw_t0 / 1024.0f);
	float p0x = wx * 0.0055f - 0.741f * t0;
	float p0y = wy * 0.0055f + 0.671f * t0;
	float w0v = 0.0f, w0gx = 0.0f, w0gy = 0.0f;
	chaff_glsl_ref_grad_noise2d(p0x, p0y, 0x57494E44u, &w0v, &w0gx, &w0gy);

	float bwx = 0.741f - w0gy * 0.28f;
	float bwy = -0.671f + w0gx * 0.28f;
	float bw_inv = 1.0f / sqrtf(bwx * bwx + bwy * bwy + 1e-6f);
	bwx *= bw_inv;
	bwy *= bw_inv;

	float raw_t1 = anim_time * 0.12f;
	float t1 = raw_t1 - 1024.0f * floorf(raw_t1 / 1024.0f);
	float u1x = wx * 0.028f + w0gx * 0.65f;
	float u1y = wy * 0.028f + w0gy * 0.65f;
	float p1x = ( 0.8f * u1x + 0.6f * u1y) - 0.822f * t1;
	float p1y = (-0.6f * u1x + 0.8f * u1y) + 0.569f * t1;
	float w1v = 0.0f, w1gx = 0.0f, w1gy = 0.0f;
	chaff_glsl_ref_grad_noise2d(p1x, p1y, 0x47555354u, &w1v, &w1gx, &w1gy);
	float mgx =  0.8f * w1gx - 0.6f * w1gy;
	float mgy =  0.6f * w1gx + 0.8f * w1gy;

	float lwx = bwx - mgy * 0.22f;
	float lwy = bwy + mgx * 0.22f;
	float lw_inv = 1.0f / sqrtf(lwx * lwx + lwy * lwy + 1e-6f);
	lwx *= lw_inv;
	lwy *= lw_inv;

	float raw_t2 = anim_time * 0.22f;
	float t2 = raw_t2 - 1024.0f * floorf(raw_t2 / 1024.0f);
	float r1x =  0.8f * (wx * 0.120f) + 0.6f * (wy * 0.120f) + w0gx * 0.45f + w1gx * 0.55f;
	float r1y = -0.6f * (wx * 0.120f) + 0.8f * (wy * 0.120f) + w0gy * 0.45f + w1gy * 0.55f;
	float p2x = ( 0.8f * r1x + 0.6f * r1y) - 0.640f * t2;
	float p2y = (-0.6f * r1x + 0.8f * r1y) + 0.768f * t2;
	float w2v = 0.0f, w2gx = 0.0f, w2gy = 0.0f;
	chaff_glsl_ref_grad_noise2d(p2x, p2y, 0x45444459u, &w2v, &w2gx, &w2gy);
	/* irot * irot = mat2(0.28, 0.96, -0.96, 0.28) */
	float u2gx =  0.28f * w2gx - 0.96f * w2gy;
	float u2gy =  0.96f * w2gx + 0.28f * w2gy;

	float fwx = lwx - u2gy * 0.16f;
	float fwy = lwy + u2gx * 0.16f;
	float fw_inv = 1.0f / sqrtf(fwx * fwx + fwy * fwy + 1e-6f);
	fwx *= fw_inv;
	fwy *= fw_inv;

	float gust = 0.42f + 0.45f * w0v + 0.55f * (w1v - 0.38f) + 0.30f * (w2v - 0.42f);
	if (gust < 0.25f) gust = 0.25f;
	if (gust > 1.35f) gust = 1.35f;

	float cp_raw = (w1v * 0.62f + w2v * 0.38f - 0.38f) / (0.76f - 0.38f);
	if (cp_raw < 0.0f) cp_raw = 0.0f;
	if (cp_raw > 1.0f) cp_raw = 1.0f;
	float cp = cp_raw * cp_raw * (3.0f - 2.0f * cp_raw);

	if (out_dir_x)   *out_dir_x   = fwx;
	if (out_dir_y)   *out_dir_y   = fwy;
	if (out_gust)    *out_gust    = gust;
	if (out_catspaw) *out_catspaw = cp;
}

static inline void chaff_glsl_ref_eval_synoptic_flow_2d(float wx, float wy, float anim_time, float fw,
                                                        float *out_vx, float *out_vy, float *out_conv, float *out_warp_x) {
	(void)fw;
	const float prevDirX = 0.741f, prevDirY = -0.671f;
	float raw_tFlow = anim_time * 0.005f;
	float tFlow = raw_tFlow - 1024.0f * floorf(raw_tFlow / 1024.0f);
	float psi0_v = 0.5f, psi0_gx = 0.0f, psi0_gy = 0.0f;
	chaff_glsl_ref_grad_noise2d(wx * 0.00000042f - prevDirX * (tFlow * 0.40f),
	                            wy * 0.00000042f - prevDirY * (tFlow * 0.40f),
	                            0x4359434Cu, &psi0_v, &psi0_gx, &psi0_gy);
	float psi1_v = 0.5f, psi1_gx = 0.0f, psi1_gy = 0.0f;
	chaff_glsl_ref_grad_noise2d(wx * 0.00000180f + (-psi0_gy) * 0.30f - prevDirX * tFlow,
	                            wy * 0.00000180f + psi0_gx * 0.30f - prevDirY * tFlow,
	                            0x5354524Du, &psi1_v, &psi1_gx, &psi1_gy);
	float synWarpX = (-psi0_gy) * 260000.0f + (-psi1_gy) * 68000.0f;
	float synVx = prevDirX + (-psi0_gy) * 0.48f + (-psi1_gy) * 0.28f;
	float synVy = prevDirY + psi0_gx * 0.48f + psi1_gx * 0.28f;
	float synInv = 1.0f / sqrtf(synVx * synVx + synVy * synVy + 1e-6f);
	synVx *= synInv;
	synVy *= synInv;
	float conv = 0.50f + 0.40f * (psi0_v - 0.50f) + 0.30f * (psi1_v - 0.50f);
	if (conv < 0.0f) conv = 0.0f;
	if (conv > 1.0f) conv = 1.0f;
	if (out_vx) *out_vx = synVx;
	if (out_vy) *out_vy = synVy;
	if (out_conv) *out_conv = conv;
	if (out_warp_x) *out_warp_x = synWarpX;
}

static inline void chaff_glsl_ref_eval_flow_2d_banded(float wx, float wy, float anim_time,
                                                      float w0_nyq, float w1_nyq, float w2_nyq,
                                                      float *out_dir_x, float *out_dir_y,
                                                      float *out_gust, float *out_catspaw) {
	if (w0_nyq <= 0.001f) {
		if (out_dir_x)   *out_dir_x   = 0.741f;
		if (out_dir_y)   *out_dir_y   = -0.671f;
		if (out_gust)    *out_gust    = 0.65f;
		if (out_catspaw) *out_catspaw = 0.0f;
		return;
	}
	float raw_t0 = anim_time * 0.05f;
	float t0 = raw_t0 - 1024.0f * floorf(raw_t0 / 1024.0f);
	float p0x = wx * 0.0055f - 0.741f * t0;
	float p0y = wy * 0.0055f + 0.671f * t0;
	float w0v = 0.5f, w0gx = 0.0f, w0gy = 0.0f;
	chaff_glsl_ref_grad_noise2d(p0x, p0y, 0x57494E44u, &w0v, &w0gx, &w0gy);

	float bwx = 0.741f - w0gy * (0.28f * w0_nyq);
	float bwy = -0.671f + w0gx * (0.28f * w0_nyq);
	float bw_inv = 1.0f / sqrtf(bwx * bwx + bwy * bwy + 1e-6f);
	bwx *= bw_inv;
	bwy *= bw_inv;

	float w1v = 0.5f, w1gx = 0.0f, w1gy = 0.0f;
	float lwx = bwx, lwy = bwy;
	if (w1_nyq > 0.001f) {
		float raw_t1 = anim_time * 0.12f;
		float t1 = raw_t1 - 1024.0f * floorf(raw_t1 / 1024.0f);
		float u1x = wx * 0.028f + w0gx * 0.65f;
		float u1y = wy * 0.028f + w0gy * 0.65f;
		float p1x = ( 0.8f * u1x + 0.6f * u1y) - 0.822f * t1;
		float p1y = (-0.6f * u1x + 0.8f * u1y) + 0.569f * t1;
		chaff_glsl_ref_grad_noise2d(p1x, p1y, 0x47555354u, &w1v, &w1gx, &w1gy);
		float mgx =  0.8f * w1gx - 0.6f * w1gy;
		float mgy =  0.6f * w1gx + 0.8f * w1gy;
		lwx = bwx - mgy * (0.22f * w1_nyq);
		lwy = bwy + mgx * (0.22f * w1_nyq);
		float lw_inv = 1.0f / sqrtf(lwx * lwx + lwy * lwy + 1e-6f);
		lwx *= lw_inv;
		lwy *= lw_inv;
	}

	float w2v = 0.5f, w2gx = 0.0f, w2gy = 0.0f;
	float fwx = lwx, fwy = lwy;
	if (w2_nyq > 0.001f) {
		float raw_t2 = anim_time * 0.22f;
		float t2 = raw_t2 - 1024.0f * floorf(raw_t2 / 1024.0f);
		float r1x =  0.8f * (wx * 0.120f) + 0.6f * (wy * 0.120f) + w0gx * 0.45f + w1gx * 0.55f;
		float r1y = -0.6f * (wx * 0.120f) + 0.8f * (wy * 0.120f) + w0gy * 0.45f + w1gy * 0.55f;
		float p2x = ( 0.8f * r1x + 0.6f * r1y) - 0.640f * t2;
		float p2y = (-0.6f * r1x + 0.8f * r1y) + 0.768f * t2;
		chaff_glsl_ref_grad_noise2d(p2x, p2y, 0x45444459u, &w2v, &w2gx, &w2gy);
		float u2gx =  0.28f * w2gx - 0.96f * w2gy;
		float u2gy =  0.96f * w2gx + 0.28f * w2gy;
		fwx = lwx - u2gy * (0.16f * w2_nyq);
		fwy = lwy + u2gx * (0.16f * w2_nyq);
		float fw_inv = 1.0f / sqrtf(fwx * fwx + fwy * fwy + 1e-6f);
		fwx *= fw_inv;
		fwy *= fw_inv;
	}

	float gust = 0.65f + 0.45f * (w0v - 0.5f) * w0_nyq + 0.55f * (w1v - 0.38f) * w1_nyq + 0.30f * (w2v - 0.42f) * w2_nyq;
	if (gust < 0.25f) gust = 0.25f;
	if (gust > 1.35f) gust = 1.35f;

	float cp_raw = (w1v * 0.62f + w2v * 0.38f - 0.38f) / (0.76f - 0.38f);
	if (cp_raw < 0.0f) cp_raw = 0.0f;
	if (cp_raw > 1.0f) cp_raw = 1.0f;
	float cp = cp_raw * cp_raw * (3.0f - 2.0f * cp_raw) * w1_nyq;

	if (out_dir_x)   *out_dir_x   = fwx;
	if (out_dir_y)   *out_dir_y   = fwy;
	if (out_gust)    *out_gust    = gust;
	if (out_catspaw) *out_catspaw = cp;
}

static inline float chaff_glsl_ref_smoothstep(float e0, float e1, float x) {
	float t = (x - e0) / (e1 - e0);
	if (t < 0.0f) t = 0.0f;
	if (t > 1.0f) t = 1.0f;
	return t * t * (3.0f - 2.0f * t);
}

static inline void chaff_glsl_ref_eval_wind_3d(float wx, float wy, float wz,
                                               float zBase, float zTop, float anim_time, float fw,
                                               float *out_dir_x, float *out_dir_y,
                                               float *out_shear_x, float *out_shear_y,
                                               float *out_gust, float *out_updraft) {
	float spanZ = (zTop - zBase) > 16.0f ? (zTop - zBase) : 16.0f;
	float h = (wz - zBase) / spanZ;
	if (h < 0.0f) h = 0.0f;
	if (h > 1.0f) h = 1.0f;
	float synVx = 0.741f, synVy = -0.671f, synConv = 0.5f, synWarpX = 0.0f;
	chaff_glsl_ref_eval_synoptic_flow_2d(wx, wy, anim_time, fw, &synVx, &synVy, &synConv, &synWarpX);
	float w0_sfc = chaff_glsl_ref_smoothstep(0.45f, 0.12f, 0.0055f * fw);
	float w1_sfc = chaff_glsl_ref_smoothstep(0.45f, 0.12f, 0.0280f * fw);
	float w2_sfc = chaff_glsl_ref_smoothstep(0.45f, 0.12f, 0.1200f * fw);
	float sfcVx = 0.741f, sfcVy = -0.671f, sfcGust = 0.65f, sfcCp = 0.0f;
	chaff_glsl_ref_eval_flow_2d_banded(wx, wy, anim_time, w0_sfc, w1_sfc, w2_sfc, &sfcVx, &sfcVy, &sfcGust, &sfcCp);
	float veerAngle = h * 0.488f;
	float cosV = cosf(veerAngle), sinV = sinf(veerAngle);
	float bBlend = 0.35f + 0.45f * h;
	float bvx = (1.0f - bBlend) * sfcVx + bBlend * synVx;
	float bvy = (1.0f - bBlend) * sfcVy + bBlend * synVy;
	float bInv = 1.0f / sqrtf(bvx * bvx + bvy * bvy + 1e-6f);
	bvx *= bInv;
	bvy *= bInv;
	float atmoDirX = bvx * cosV - bvy * sinV;
	float atmoDirY = bvx * sinV + bvy * cosV;
	float atmoSpeed = 1.0f + 0.85f * powf(h, 1.35f);
	float shearMag = powf(h, 1.25f) * spanZ * 0.35f * atmoSpeed;
	float gust = (1.0f - h) * sfcGust + h * (1.0f + 0.50f * (synConv - 0.5f));
	if (gust < 0.25f) gust = 0.25f;
	if (gust > 1.75f) gust = 1.75f;
	float updraft = 0.45f * synConv + 0.35f * sfcGust + 0.20f * sfcCp;
	if (updraft < 0.10f) updraft = 0.10f;
	if (updraft > 1.50f) updraft = 1.50f;
	if (out_dir_x) *out_dir_x = atmoDirX;
	if (out_dir_y) *out_dir_y = atmoDirY;
	if (out_shear_x) *out_shear_x = atmoDirX * shearMag;
	if (out_shear_y) *out_shear_y = atmoDirY * shearMag;
	if (out_gust) *out_gust = gust;
	if (out_updraft) *out_updraft = updraft;
}

#endif /* TTYPT_CHAFF_GLSL_H */
