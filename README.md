<p align="center">
  <img src="assets/logo.png" alt="libchaff logo" width="220" />
</p>

# libchaff

[![npm](https://img.shields.io/badge/npm-@tty-pt/chaff-CB3837?logo=npm)](npm/README.md)
[![C99](https://img.shields.io/badge/C-C99-555?logo=c)](#)
[![CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue)](#)
[![Zero Floats](https://img.shields.io/badge/FPU-zero%20floats-16A34A)](#)
[![Deterministic](https://img.shields.io/badge/parity-CPU%20%2B%20GPU%20bit--exact-blueviolet)](#)
[![Zero OpenGL](https://img.shields.io/badge/server-zero%20OpenGL-brightgreen)](#)

> High-efficiency procedural noise engine, multi-scale recursive LOD, seamless 2's complement data type boundary wrapping, and $C^2$ anti-resonance multi-octave synthesis for glitch-free planetary terrain and bathymetry. Canonical headers are `<ttypt/chaff.h>` and `<ttypt/noise.h>`.

---

## 1. Quickstart

Evaluate continuous multi-octave procedural terrain with analytical surface normal slopes, or step a dense 2D time-slice in under 20 lines of C99:

```c
#include <stdio.h>
#include <ttypt/chaff.h>

int main(void) {
    /* 1. Declarative multi-octave noise field definition */
    static const chaff_layer_t layers[] = {
        { .shift = 8, .vshift = 8, .amp = 120, .seed_mode = CHAFF_SEED_SHARED }, /* Macro relief */
        { .shift = 5, .vshift = 8, .amp =  45, .seed_mode = CHAFF_SEED_SHARED }, /* Rolling hills */
        { .shift = 3, .vshift = 8, .amp =  15, .seed_mode = CHAFF_SEED_SHARED }  /* Micro crags */
    };
    chaff_field_t field = {
        .seed = 0x1337BEEFu, .dim = 2, .nlayers = 3, .layers = layers
    };

    /* 2. Sparse point query with analytical surface normal slopes (zero extra hashes) */
    chaff_sample_2d_t s = chaff_sample_grad_2d(&field, 1024, -2048);
    printf("Elevation: %d, Slope Normal: (dX=%d, dY=%d)\n",
           s.elevation, s.grad_x, s.grad_y);

    /* 3. Bulk 2D time-slice across 3D field (e.g. animating water waves at time Z=42) */
    int32_t heightmap[32 * 32];
    chaff_fill_slice_3d(&field, 42 /* z */, 0 /* x0 */, 0 /* y0 */,
                        32, 32, heightmap, NULL, NULL);
    printf("Bulk slice evaluated (origin sample: %d)\n", heightmap[0]);

    return 0;
}
```

### Build & Link
```sh
# Native Linux / OpenBSD compilation (links against libchaff / libnoise):
cc worldgen.c -Iinclude -Llib -lchaff -lxxhash -o worldgen
# Or link via the backwards-compatible library name:
cc worldgen.c -Iinclude -Llib -lnoise -lxxhash -o worldgen
```

### Dual-Target Compilation Pipeline
The library compiles cleanly into both a native server shared object and a freestanding WebAssembly kernel:

```sh
# 1. Native server build (-mgeneral-regs-only enforces zero float registers):
cc -Wall -Wextra -Werror -O2 -Iinclude -mgeneral-regs-only -fPIC -shared \
   -o lib/libnoise.so src/libchaff.c -lxxhash

# 2. Freestanding browser WebAssembly build:
clang --target=wasm32 -O3 -flto -nostdlib -Wl,--no-entry \
      -Wl,--export-all -Iinclude -DXXH_INLINE_ALL -o dist/noise.wasm src/libchaff.c
```

---

## 2. Core Architectural Principles

`libchaff` is built from the ground up for massive, continuous procedural worlds spanning from sub-unit ocean caustics and sand dunes all the way up to orbital planetary horizons. It addresses the fundamental failure modes of traditional floating-point noise libraries: floating-point drift, seam cracking at coordinate limits, high-frequency aliasing during zoom-out, and Cartesian lattice terracing.

```
+-------------------------------------------------------------------------------+
|                           LIBCHAFF ARCHITECTURE                               |
+-------------------------------------------------------------------------------+
|  1. Integer Optimization    Zero FPU/SSE/AVX; register forward-difference     |
|                             stepping; >96% hash reduction in 3D slices.       |
|  2. Boundary Wrapping       2's complement int16_t modular arithmetic; zero   |
|                             seams or step discontinuities at 32767 -> -32768. |
|  3. Recursive & Band-LOD    Band-limited Nyquist thresholding (chaff_fill_lod)|
|                             and Q16 coordinate transformers (coord_scale).    |
|  4. Anti-Resonance Skew     Irrational Golden-Ratio rotation (theta ~ 31.7 deg|
|                             chaff_rotate_q15) breaks Cartesian terracing.     |
|  5. C2 Hermite Smoothstep   Perlin quintic polynomial (NOISE_FADE5) eliminates|
|                             slope creases and Mach banding at cell borders.   |
|  6. Zero-OpenGL Server      Headless OpenBSD/Linux server purity with bit-    |
|                             exact GLSL ES 3.0 WebGL2 shader parity.           |
+-------------------------------------------------------------------------------+
```

---

## 3. Maximizing Efficiency with Integer Operations

Traditional procedural noise libraries rely on floating-point arithmetic (`sin`, `cos`, IEEE 754 floats) and point-by-point function evaluations. For real-time chunk generation and terrain rendering, this imposes severe overhead: millions of transcendental instructions, branch mispredictions, and redundant corner hashes.

`libchaff` maximizes algorithmic and hardware efficiency through pure integer operations:

### A. Pure C99 Integer Core (`-mgeneral-regs-only`)
* The core library is strictly integer arithmetic: addition, subtraction, bitwise shifts, modular masking, and integer multiplications.
* Compiling with `-mgeneral-regs-only` verifies that **zero floating-point, SSE, AVX, or NEON instructions** exist in the compiled binary (`objdump` audit confirms zero `xmm`/`fadd`/`fmul` instructions).
* Eliminates IEEE 754 non-determinism across compilers (`-O2` vs `-O3`), architectures (x86_64 vs ARM64), and execution targets (WASM32 vs native).

### B. Hypercube Forward-Difference Matrix Stepping (`noise_m`, `noise_quad`)
* Naive grid evaluation queries each point independently, repeating corner hash computations $2^{\text{dim}}$ times per point ($400,000+$ hashes for a standard chunk).
* `libchaff`'s matrix stepper (`noise_quad`) precomputes integer step deltas along hypercube axes via `calc_steps()`, stepping intermediate lattice points in CPU registers using integer addition:
  $$\text{st}[n] = ((v[n + vl] - v[n]) \gg z) \ll y$$
* This amortizes corner hashing to virtually **~0 hashes per cell**, accelerating dense chunk rasterization by **10x to 15x**.

### C. 2D Slice Stepping Across 3D Fields (`chaff_fill_slice_3d`)
* For animated surfaces (such as dynamic ocean waves where $Z = \text{time}$) or 2.5D diorama terrain slices:
* Rather than evaluating 8 lattice corners for every $(x, y)$ coordinate at time $Z$, `chaff_fill_slice_3d()` pre-collapses the invariant $Z$ axis **once per 3D lattice cell** into a 2D quad in CPU registers:
  $$c_0 = v_0 + (((v_4 - v_0) \cdot u_z) \gg 16), \quad c_1 = v_1 + (((v_5 - v_1) \cdot u_z) \gg 16)$$
  $$c_2 = v_2 + (((v_6 - v_2) \cdot u_z) \gg 16), \quad c_3 = v_3 + (((v_7 - v_3) \cdot u_z) \gg 16)$$
* On a $32 \times 32$ cell, this cuts corner hashes from 256 down to 8 — a **96.8% reduction in hash operations** while producing bit-exact identical values to point queries.

### D. Scanline Caching (`noise_scanline_t`, `noise_wcache_t`)
* When rasterizing rows of pixels, `noise_scanline_eval_row()` and `noise_wrow()` cache active lattice corners in registers.
* Lattice corners are fetched only when crossing a cell boundary (e.g. 17 corner fetches for 256 pixels at shift 4, rather than 256 independent fetches).

### E. Analytical Spatial Gradients in Registers (Zero Extra Hashes)
* Determining surface normal slopes ($\frac{\partial H}{\partial x}, \frac{\partial H}{\partial y}, \frac{\partial H}{\partial z}$) traditionally requires finite central differences:
  $$\frac{\partial H}{\partial x} \approx \frac{H(x + \epsilon, y) - H(x - \epsilon, y)}{2\epsilon}$$
  This requires 2 to 4 additional noise evaluations per pixel (16 to 24 extra corner hashes in 3D).
* `chaff_sample_grad_2d()` and `chaff_sample_grad_3d()` differentiate the cubic Hermite smoothstep analytically:
  $$u(t) = t^2(3 - 2t) \implies u'(t) = 6t(1 - t)$$
* Derivatives are computed in fixed-point integer registers directly alongside the elevation sample. Normal slopes are evaluated with **zero additional corner hashes**, delivering a **3x speedup** over numerical finite differences.

---

## 4. Seamless 2's Complement Boundary Wrapping

In open-world procedural engines, coordinate systems eventually hit data type limits. Naive floating-point or integer implementations suffer from:
1. **Coordinate Truncation Glitches**: Integer overflow causing abrupt clamping or inversion.
2. **Boundary Seams**: Step discontinuities between maximum positive values and negative values (e.g. $+32767 \to -32768$).
3. **Floating-Point Precision Loss**: Floating-point coords far from the origin ($> 10^5$) lose sub-pixel precision, causing geometry jitter.

```
Coordinate Axis:  ... 32765   32766   32767 | -32768  -32767  -32766 ...
Lattice Cell:     [--------- Cell K --------] [-------- Cell K+1 -------]
                                            ^
                         Two's complement wrap: (int16_t)(32767 + 1) == -32768
                         Upper lattice corner seamlessly rolls over to -32768.
                         Hash key and Hermite offset are 100% continuous.
```

### Mathematical Continuity Across `int16_t` Limits
`libchaff` uses two's complement `int16_t` coordinates spanning $[-32768 \dots 32767]$:
1. **Lattice Key Masking**: The lower corner of a cell is masked by bitshift:
   ```c
   int16_t cell = (int16_t)(p & COORDMASK_LSHIFT(x));
   ```
2. **Modular Corner Offset**: The far vertex of a quad is computed via 16-bit modular addition:
   ```c
   int16_t far_corner = (int16_t)(cell + (1 << x));
   ```
   When `cell == 32768 - (1 << x)`, adding $2^x$ wraps naturally to $-32768$:
   $$(32767 + 1) \equiv -32768 \pmod{65536}$$
3. **Identical Hash Digest**: Because the hash input is the 16-bit integer coordinate itself, the digest evaluated at $+32768$ matches $-32768$ bit-for-bit.
4. **Fractional Offset Continuity**: Intra-cell position $f \in [0, 2^x)$ is computed with bitwise masking:
   ```c
   uint32_t fix = (uint16_t)p & (uint16_t)~mask;
   ```
   As coordinates increment from $+32767$ to $-32768$, `fix` progresses smoothly from $(2^x - 1)$ to $0$, continuing the smoothstep without a single step discontinuity or missing pixel.

**Result**: Infinite coordinate traversal and continuously incrementing time counters (`(int16_t)anim_tick`) wrap around the data type boundary with **zero seams, zero tearing, and zero branch penalties**.

---

## 5. Recursive & Band-Limited Level of Detail (LOD)

Rendering a continuous planet from orbit down to sub-unit terrain requires smooth transitions across orders of magnitude of scale. Naive multi-octave noise causes two severe artifacts during zoom:
1. **High-Frequency Moiré Aliasing**: Sampling high-frequency octaves at low sampling rates (large step sizes) produces chaotic visual noise and strobing.
2. **Abrupt LOD Popping**: Dropping octaves abruptly causes visible snapping and geometric tears at chunk borders.

### A. Band-Limited Nyquist Filtering (`chaff_fill_lod`)
`chaff_fill_lod()` dynamically filters the octave cascade based on the Nyquist-Shannon sampling theorem. An octave with wavelength $\lambda = 2^{\text{shift}}$ cannot be accurately represented by a sampling step size $\Delta > \frac{\lambda}{2}$.

```c
/* Automatic Nyquist cutoff based on LOD resolution */
uint8_t min_shift = 0;
if      (lod == 0) min_shift = 6; /* Macro planetary rooms (step 64) */
else if (lod == 1) min_shift = 5; /* Distant terrain (step 16)       */
else if (lod == 2) min_shift = 3; /* Medium terrain (step 4)        */
/* LOD 3: all octaves active (step 1, full detail)                  */

uint8_t lo = 0;
while (lo < field->nlayers && field->layers[lo].shift < min_shift) {
    lo++; /* Prune sub-Nyquist high frequencies */
}
chaff_fill(field, lo, field->nlayers, origin, step_q16, n, out, stride, mode);
```
* **Performance Gain**: Distant LOD 1 and LOD 2 evaluations skip 3 to 5 fine octaves, accelerating rendering by **2x to 4x**.
* **Visual Stability**: Distant hills and coastlines retain stable silhouettes without pixel flickering or moiré patterns.

### B. Hierarchical Coordinate Scaling (`chaff_coord_scale`)
Fixed-point coordinates are scaled across recursive LOD layers without precision loss:
```c
typedef struct {
    int16_t  cell;  /* Lattice/room units */
    uint16_t frac;  /* 1/65536 sub-unit fractional precision */
} chaff_coord_t;
```
`chaff_coord_scale(coord, power_of_two)` scales the combined 32-bit fixed-point value by $2^k$, preserving sub-pixel fractional alignments across recursive quadtree subdivisions.

---

## 6. Smooth Multi-Octave Noise Without Distortions

Standard Value Noise often exhibits objectionable visual artifacts: grid-aligned ridges, diagonal bias, and flat terraces resembling layered steps. `libchaff` eliminates these artifacts through mathematical formulation:

### A. $C^2$ Quintic Hermite Smoothstep (`NOISE_FADE5`)
Standard cubic Hermite smoothing $u_3(t) = 3t^2 - 2t^3$ has a non-zero second derivative at boundaries ($u''_3(0) = 6, u''_3(1) = -6$). This discontinuity in surface curvature produces visible creases and Mach banding along cell borders.

`libchaff` implements Ken Perlin's quintic polynomial in integer arithmetic (`NOISE_FADE5`):
$$u_5(t) = 6t^5 - 15t^4 + 10t^3$$
Its first and second derivatives are:
$$u_5'(0) = u_5'(1) = 0, \quad u_5''(0) = u_5''(1) = 0$$
Both velocity and acceleration vanish at lattice cell edges. The resulting heightfield is **$C^2$ continuous**, eliminating all boundary creases and surface normal cracks.

### B. Anti-Resonance Rotational Skew (`chaff_rotate_q15`)
When accumulating $N$ octaves of fractal noise on aligned Cartesian grids, power-of-two lattice boundaries coincide ($x \pmod{16} == 0$, $x \pmod{32} == 0$). Their stationary points ($u'(t) = 0$) line up, creating severe grid terracing and square plateaus.

`libchaff` eliminates Cartesian lattice resonance by rotating consecutive octave coordinate systems by an **irrational Golden-Ratio angle** ($\theta \approx 31.7^\circ$):

```c
#define CHAFF_ROT_COS 27874 /* cos(31.7 deg) * 32768 */
#define CHAFF_ROT_SIN 17227 /* sin(31.7 deg) * 32768 */

static inline void chaff_rotate_q15(int16_t *x, int16_t *y) {
    int32_t rx = ((int32_t)(*x) * CHAFF_ROT_COS - (int32_t)(*y) * CHAFF_ROT_SIN) >> 15;
    int32_t ry = ((int32_t)(*x) * CHAFF_ROT_SIN + (int32_t)(*y) * CHAFF_ROT_COS) >> 15;
    *x = (int16_t)rx;
    *y = (int16_t)ry;
}
```
* Each successive octave samples terrain along an independently rotated orientation.
* Eliminates cardinal axis alignment, diagonal banding, and staircase terracing across infinite distances.

### C. Continuous Planetary 12-Octave Ladder (`chaff_elevation_q12_4`)
Procedural planetary elevation is evaluated via a 12-octave geometric progression:
* **Macro Shifts 14..9**: Continental landmasses and oceanic basins mapped across the $65,536 \times 65,536$ room planetary lattice.
* **Fine Shifts 8..3**: Sub-room topography, rolling hills, dune ripples, and coastal bathymetry in sub-pixel Q12.4 precision.
* **Calibrated Sea-Level Baseline**: Continental median elevation is calibrated to the canonical sea-level zero crossing, ensuring that ocean basins, coastal shallows, and mountain ranges align across all zoom levels.

---

## 7. Glitch-Free Planet Rendering & Zero-OpenGL Invariant

NeverDark and the pure-C web stack adhere to a strict architectural rule:

### The Server's Right Not to Sell GPU
* The server runtime (`nd.so`, `libaxil-nd.so`) runs strictly as pure software C99 integer math (`-mgeneral-regs-only`, zero floats, zero GPU libraries).
* **Zero OpenGL Linkage**: Zero `#include <GL/gl.h>`, `<EGL/egl.h>`, `-lGL`, or `-lgbm`. Headless Linux/OpenBSD server daemons run on cloud instances with zero graphics hardware.
* All collision detection, room exits, and player physics are computed deterministically via CPU integer math.

### Bit-Exact Client GPU Acceleration (`<ttypt/chaff_glsl.h>`)
Clients with GPU capabilities (browser WebGL2 via Web HAL or native devices via `liblamina`) evaluate noise shaders on GPU compute/fragment cores.

To prevent client-server desynchronization (e.g. a client rendering dry land where the server registers drowning water collision), `<ttypt/chaff_glsl.h>` exports a **bit-exact GLSL ES 3.0 integer hash**:

```glsl
/* Bit-exact XXH32 integer hash matching CPU libchaff in GLSL ES 3.0 */
uint chaff_rotl32(uint x, int r) {
    return (x << uint(r)) | (x >> uint(32 - r));
}

uint chaff_hash2(ivec2 p, uint seed) {
    uint val = (uint(p.y) & 0xFFFFu) | ((uint(p.x) & 0xFFFFu) << 16u);
    uint h = seed + 0x165667B1u + 4u;
    h += val * 0xC2B2AE3Du;
    h = chaff_rotl32(h, 17) * 0x27D4EB2Fu;
    h ^= h >> 15u;
    h *= 0x85EBCA77u;
    h ^= h >> 13u;
    h *= 0xC2B2AE3Du;
    h ^= h >> 16u;
    return h;
}
```
* **Level-Set Equivalence**: The GPU shader evaluates the exact same 32-bit corner hashes and Hermite smoothstep as the server CPU.
* The shoreline level-set $C_{\text{shore}} = \{ (wx, wy) \mid H(wx, wy) = W \}$ matches identically between WebGL2 fragment shaders and server collision logic.

---

## 8. Public C API Reference

The library provides two primary headers: `<ttypt/chaff.h>` (modern declarative cascades, analytical gradients, planetary elevation) and `<ttypt/noise.h>` (low-level matrix steppers, point samplers, and wrapping primitives).

### Data Structures

| Type | Header | Description |
|---|---|---|
| `chaff_coord_t` | `<ttypt/chaff.h>` | Fixed-point coordinate: `int16_t cell` + `uint16_t frac` (1/65536 unit). |
| `chaff_layer_t` | `<ttypt/chaff.h>` | Octave descriptor: `shift`, `vshift`, signed `amp`, `seed_mode`, `contrast`. |
| `chaff_field_t` | `<ttypt/chaff.h>` | Multi-layer declarative field descriptor with seed, dimensionality, and layer array. |
| `chaff_sample_2d_t` | `<ttypt/chaff.h>` | 2D query result: `int32_t elevation`, analytical slopes `int16_t grad_x`, `grad_y`. |
| `chaff_sample_3d_t` | `<ttypt/chaff.h>` | 3D query result: `int32_t val`, analytical gradients `grad_x`, `grad_y`, `grad_z`. |
| `noise_wcache_t` | `<ttypt/noise.h>` | Lattice corner vertex cache for scanline and row stepping. |
| `noise_scanline_t` | `<ttypt/noise.h>` | Declarative multi-octave scanline stepper state. |

### Core Evaluation Functions

#### Sparse Point Samplers
* `chaff_sample_grad_2d(field, wx, wy)`: Evaluates continuous 2D height and analytical surface slopes in a single pass with zero extra hashes.
* `chaff_sample_grad_3d(field, x, y, z)`: Evaluates 3D value and spatial gradient $(dN/dx, dN/dy, dN/dz)$ using analytical Hermite derivatives in 8 corners.
* `chaff_sample_rugged_2d(field, wx, wy)`: Point evaluation with irrational Golden-Ratio rotational skew applied across octaves to break lattice alignment.
* `chaff_elevation_q12_4(world, lod, wx, wy, &coarse, &fine)`: Authoritative 12-octave continuous planetary elevation engine returning Q12.4 fixed-point elevation with calibrated sea-level zero-crossing.
* `noise_point_n(p, seed, shift, dim)`: Generic $N$-dimensional zero-allocation point query ($N \in [1..4]$).
* `noise_wpoint_n(p, shift, w, seed, dim, flags)`: Two's complement wrapping point query.

#### High-Throughput Matrix & Region Steppers
* `chaff_fill(field, lo, hi, origin, step_q16, n, out, stride, mode)`: Primary fixed-point multi-layer region generator.
* `chaff_fill_lod(field, lod, origin, step_q16, n, out, stride, mode)`: Band-limited region generator pruning sub-Nyquist octaves.
* `chaff_fill_slice_3d(field, z, x0, y0, w, h, out_val, out_gx, out_gy)`: Bulk 2D slice stepper across 3D field; collapses $Z$ axis once per cell in registers (>96% hash reduction).
* `noise_matrix_fill_2d(cfg, ox, oy, size_shift, out)`: Bulk macroscopic quad fill using register additions.
* `noise_scanline_eval_row(scan, sx, y, z, count, out_row)`: Stepper fetching corners once per cell boundary.

#### Coordinate Transformers & Math
* `chaff_coord_scale(coord, power_of_two)`: Power-of-two zoom scaling for fixed-point coordinates.
* `chaff_rotate_q15(&x, &y)`: Fixed-point rotation by Golden-Ratio angle ($\theta \approx 31.7^\circ$).
* `noise_fdiv(a, b)`: Euclidean integer floor division preserving mathematical continuity across negative coordinates.
* `noise_wfade5(f, x)`: $C^2$ continuous quintic Hermite polynomial $6t^5 - 15t^4 + 10t^3$ mapping $[0..2^x)$ to $[0..2^x)$.

---

## 9. Performance Metrics & Benchmarks

Run the benchmark suite:
```sh
make -C external/libchaff bench
```

Empirical throughput measured on standard host environment:

| Benchmark Operation | Target Method | Throughput | Latency | Hash Reductions / Speedup |
|---|---|---|---|---|
| **Point Query 2D** | `noise_point_n` | **18.76 Mops/s** | 53.31 ns/op | Baseline 2D point lookup |
| **Point Query 3D** | `noise_point_n` | **7.70 Mops/s** | 129.91 ns/op | Baseline 3D point lookup |
| **Analytical 2D Gradient** | `chaff_sample_grad_2d` | **5.45 Mops/s** | 183.42 ns/op | 3 octaves, **0 additional hashes** |
| **Analytical 3D Gradient** | `chaff_sample_grad_3d` | **2.32 Mops/s** | 430.71 ns/op | 3 octaves, **0 additional hashes** |
| **2D Time-Slice Stepper** | `chaff_fill_slice_3d` | **88.44 Mops/s** | 11.31 ns/sample | **37.30x speedup** vs 3D point query (2.37 Mops/s); **>96% corner hash reduction** |
| **1D Scanline Stepper** | `noise_scanline_eval_row` | **120.23 Mops/s** | 8.32 ns/sample | 17 corner fetches per 256 pixels |
| **Planetary 12-Octave LOD 3** | `chaff_elevation_q12_4` | **1.33 Mops/s** | 754.63 ns/op | Full 12 octaves, sub-pixel relief |
| **Planetary 12-Octave LOD 0** | `chaff_elevation_q12_4` | **2.64 Mops/s** | 378.11 ns/op | Macro 6 octaves, **2.00x Nyquist speedup** |

---

## 10. Verification of Core Architectural Promises & Testing Ecosystem

Modeled after the testing and metrics architecture in `external/libcorm`, `external/libchaff` provides a modular 4-suite test harness and turnkey runner verifying numerical stability, boundary wrapping, and GPU shader parity:

```sh
# Run all 4 modular test suites:
make -C external/libchaff test

# Run the complete test suite and comprehensive operational metrics:
./external/libchaff/test.sh

# Run standalone metrics & throughput benchmarks:
make -C external/libchaff bench
```

### Modular Test Suite Layout
1. **Core Engine Suite (`bin/test_noise`)**: 35 automated assertions verifying 1D–4D point sampling, hypercube matrix stepping, scanline caches, plane steppers, and boundary continuity.
2. **Canonical libchaff Public API Suite (`bin/test_chaff`)**: 11 comprehensive sections verifying point wrappers (`chaff_point_1d`..`4d`, `chaff_point_n`), canonical matrix/sample adapters (`chaff_oct`, `chaff_matrix_fill_2d`), `chaff_coord_scale`, `chaff_fade_frac`, 1D/3D volume `chaff_fill` in SET and ADD modes, coordinate skewing, contrast, 2D/3D analytical gradients, 2D time-slice stepping (`chaff_fill_slice_3d`) across multi-cell and negative coordinates, rotational skew (`chaff_rotate_q15`, `chaff_rotate_q15_inplace`), planetary 12-octave elevation (`chaff_elevation_q12_4`), and defensive NULL-safety guards.
3. **Shared Library Exported ABI Suite (`bin/test_symbols`)**: 5 sections verifying that all exported symbols in `src/libchaff.c` (`ttypt_noise_*`, `chaff_*_sym`, `ttypt_chaff_fill`, `chaff_fill_sym`, `chaff_sample_grad_3d_sym`, `chaff_fill_slice_3d_sym`) resolve cleanly and match inline header functions bit-for-bit.
4. **GLSL ES 3.0 GPU Shader Contract Suite (`bin/test_glsl`)**: 8 sections verifying shader strings, uniform buffer structure layout (`chaff_gl_uniforms_t`), rotl32 bit operations, bit-exact 2D/3D integer hashes (`chaff_hash2`, `chaff_hash3`), continuous octave sampling, 3D analytical gradients, and Golden-Ratio rotation parameters.

### Verified Test Suite Assertions
1. **Generic Point Sampling**: 1D, 2D, 3D, and 4D sampling bounded within valid [0..255] bounds.
2. **Deterministic Parity**: Identical output across repeated calls with identical seeds.
3. **3D Point Continuity**: Value continuity verified along arbitrary 3D spatial trajectories.
4. **Generic Matrix Stepping**: Hypercube forward-difference stepping in 2D and 3D.
5. **Wrapping Sampler Equivalence**: Value-equal to `noise_m` across 232,368 sample grid.
6. **Data Type Boundary Wrapping**: 0 seams across $32767 \to -32768$ boundary on all axes.
7. **Animation Counter Continuity**: `int16_t` casting of continuously increasing counter runs seamlessly.
8. **Stepper & Point Equivalence**: Cached point and row stepper produce bit-exact values to point sampler.
9. **Corner Reuse Efficiency**: Row stepper fetches only 17 corners for 256 pixels at shift 4.
10. **Lattice Identity Mapping**: Fade maps $[0, 2^x)$ onto itself preserving boundary corners.
11. **Plane Stepper Efficiency**: Plane stepper fetches only 16 corners for $64 \times 64$ pixels at shift 4.
12. **Declarative Point Determinism**: 2D and 3D declarative field evaluation determinism.
13. **Bulk Matrix 2D Fill**: Non-zero surface coverage verified across coordinate plane.
14. **Scanline Cache Efficiency**: Row evaluation amortizes lattice corners with zero redundant fetches.
15. **Cross-Representation Equivalence**: `point == row[k] == matrix[r][k]` across 2D & 3D wrap boundaries.
16. **Chaff & Noise Octave Parity**: `chaff_fill` at fractional zero equals `noise_oct` bit-for-bit.
17. **Layer Nesting Linearity**: $\sum(0..1) + \sum(1..3) == \sum(0..3)$ verified with fractional coordinates.
18. **Fixed-Point Coordinate Scaling**: `chaff_coord_scale` verified across recursive power-of-two zoom levels.
19. **Band-Limited Nyquist Cuts**: `chaff_fill_lod` correctly suppresses sub-Nyquist high frequencies.
20. **Analytical Gradient Shader Contract**: Analytical Hermite gradients exported to GLSL interface.
21. **Bit-Exact GLSL Shader Hash**: `chaff_hash2` is 100% bit-exact with CPU XXH32 across coordinate grid.
22. **Continuous GLSL Level-Set Parity**: GLSL evaluation matches CPU `chaff_fill` within numerical tolerance.
23. **Monotonic Sub-Cell Ramps**: Sub-cell slope ramps are strictly monotonic without truncation cliffs.
24. **$C^2$ Quintic Hermite Polynomial**: `NOISE_FADE5` verified with zero 1st and 2nd derivatives at cell edges.
25. **Active Pipeline $C^2$ Integration**: `NOISE_FADE5` and `chaff_lerp` active in continuous sampling.
26. **3D GLSL Hash Parity**: `chaff_hash3` reference implementation verified across 3D grid.
27. **3D Analytical Gradient Parity**: Analytical Hermite gradients match numerical derivatives on CPU.
28. **3D Slice Bit-Exactness**: `chaff_fill_slice_3d` is 100% bit-exact to `chaff_sample_grad_3d` (1024/1024 samples).
29. **3D Slice Hash Amortization**: `chaff_fill_slice_3d` collapses $Z$ axis once per cell (>96% hash reduction).
30. **GLSL 3D Gradient Parity**: GLSL analytical gradient matches numerical limit within numerical tolerance.
31. **Promise 1 (Integer Operations)**: `noise_fdiv` satisfies Euclidean invariant ($0 \le r < |b|$) across all negative/positive coordinates.
32. **Promise 2 (Boundary Wrapping)**: 0 seams across `int16_t` wrap ($32767 \to -32768$) in multi-octave fields (step $\le$ normal intra-cell limit).
33. **Promise 3 (Anti-Resonance Skew)**: Golden-Ratio rotational skew (`chaff_rotate_q15`) breaks Cartesian lattice alignment.
34. **Promise 4 (Recursive LOD Conservation)**: Multi-scale recursive LOD downsampling preserves regional mean elevation.
35. **Promise 5 (Glitch-Free Planet)**: Continuous planetary terrain across room divisor boundaries ($wx = 79 \to 80$).

---

---

## 11. Spatial Index & Quadtree Integration (`libchunk`)

`libchaff` is designed to pair directly with `libchunk` (`<ttypt/chunk.h>`) for real-time 3D planetary rendering, camera frustum culling, and multi-scale LOD management.

### Canonical Sampler Binding
```c
#include <ttypt/chaff.h>
#include <ttypt/chunk.h>

void chaff_chunk_sampler(void *user_ctx, uint16_t world, uint8_t level,
                         int32_t wx, int32_t wy, chunk_sample_t *out)
{
    /* Select appropriate procedural LOD based on quadtree hierarchy level */
    uint8_t lod = (level >= 6) ? 0 : ((level >= 3) ? 1 : (uint8_t)(3 - level));
    int32_t coarse = 0, fine = 0;
    int32_t elev_q12_4 = chaff_elevation_q12_4(world, lod, wx, wy, &coarse, &fine);

    /* Pack into generic libchunk spatial sample */
    out->val = (int16_t)elev_q12_4;
    out->attr[0] = (elev_q12_4 > 100 * 16) ? 2 : 10; /* Biome / material */
    out->attr[1] = out->attr[0];
    out->attr[2] = 0;
    out->attr[3] = (elev_q12_4 < 0) ? (uint8_t)(-elev_q12_4 >> 4) : 0; /* Water depth */
    out->rgba = (out->attr[3] > 0) ? 0xFF003080u : 0xFF208020u;
    out->flags = (out->attr[3] > 0) ? 1 : 0;
}
```

The resulting chunk system can be queried with column-major $4 \times 4$ ModelView-Projection (MVP) matrices using Gribb-Hartmann 6-plane frustum extraction and analytic spherical curvature sag envelopes via `chunk_system_query_matrix()`.

---

## 12. Installation & Build

### Prebuilt Packages
Prebuilt packages for Linux (APT / Alpine / Arch / Fedora-RHEL), macOS (Homebrew), Windows (winget / MSYS2), and OpenBSD are distributed via [tty.pt packages](https://github.com/tty-pt/ci/blob/main/docs/install.md). Use **libchaff** as the package name.

### Build from Source
```sh
make              # builds lib/libchaff.so and lib/libnoise.so
make wasm         # builds npm/dist/chaff.wasm
make test         # runs the 35-assertion C & GLSL parity verification suite
make npm-test     # runs the Node.js & WebAssembly test suite
sudo make install # install headers and libraries to system
```

---

## 13. JavaScript & WebAssembly Bindings (`@tty-pt/chaff`)

For browser, Node.js, and WebGL2 shader pipelines, `libchaff` ships an isomorphic WebAssembly and GLSL package on npm:

```sh
npm install @tty-pt/chaff
```

See [npm/README.md](npm/README.md) for the full TypeScript, WebAssembly, and GLSL shader API reference.

---

## License

CC BY-NC-SA 4.0 — Author: Paulo Quirino <q@qnixsoft.com>
