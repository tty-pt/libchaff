# Changelog

All notable changes to `libchaff` (`@tty-pt/chaff`) are documented in this file.

## [Unreleased] (target: v1.1.1)

### Added
- **4-Wide SIMD 2D Noise, `fade5` LUT & VLA-Free 2D Fast Paths (`include/ttypt/chaff.h`, `include/ttypt/noise.h`, `npm/include/ttypt/*`, `tests/test_chaff.c`)**:
  - Added 4-wide SIMD (`__wasm_simd128__` with bit-exact scalar fallback) 2D evaluation APIs in `include/ttypt/chaff.h`: `chaff_rot2d_i32_q15_4x()`, `chaff_wpoint_periodic_2d_4x()`, `chaff_fbm_rotated_2d_4x()`, `chaff_fbm_rotated_2d_step_q12_4_4x()`, and `chaff_grad_noise2d_4x()`.
  - Added branchless 4-byte XXH32 helpers (`noise_pack2`, `noise_hash2_1`, `noise_hash2_v128`, `noise_hash2_4x`), 512-entry quintic fade LUT (`s_noise_fade5_q16_lut`) accelerating `noise_wfade5()`, and 4-wide SIMD/scalar primitives (`noise_wfade5_v128`, `noise_wfade5_4x`, `noise_wstep_v128`, `noise_wstep_v128_u24`, `noise_wstep_4x`, `noise_wpoint_2d_4x`) in `include/ttypt/noise.h`.
  - Added VLA-free `dim == 2` fast paths in `noise_get_v()` and `noise_wpoint_c()` (`include/ttypt/noise.h`) eliminating stack VLAs for 2D noise and FBM.
  - Added bit-exact scalar-vs-4x parity tests in `tests/test_chaff.c`.
- **Thread-Local 4-Corner Lattice Cell Caching (`include/ttypt/chaff.h`, `npm/include/ttypt/chaff.h`)**:
  - Added 4-corner cell caching to `chaff_wpoint_periodic_2d_ex()` / `chaff_wpoint_periodic_2d_4x()` and per-octave cell caching (`CHAFF_FBM_MAX_OCTAVES = 8`) to `chaff_fbm_rotated_2d()`, reusing evaluated quad corner hashes whenever successive queries land inside the same `(cell_x, cell_y, shift, period_shift, seed)` lattice cell.
  - Qualified noise cell caches with portable thread-local storage (`_CHAFF_TLS` / `__declspec(thread)` / `__thread` / `_Thread_local`) so concurrent worker threads (`stele_wq`) evaluate periodic and rotated FBM noise with zero data races.

## [1.1.0] - 2026-11-16

### Added
- **Periodic Tiling, Domain Skew & 4D Slice Stepping (`include/ttypt/chaff.h`, `include/ttypt/noise.h`, `include/ttypt/chaff_glsl.h`)**:
  - Wired `period_shift` and `skew[2]` across point samplers (`noise_wpoint_n_periodic`), corner caches (`noise_wcorners_periodic`), analytical 2D/3D/4D gradient samplers (`chaff_sample_grad_2d`, `chaff_sample_grad_3d`, `chaff_sample_grad_4d`), and GLSL shader contracts (`chaff_glsl.h`).
  - Added bulk 2D slice stepper across 4D fields (`chaff_fill_slice_4d`) pre-collapsing invariant $W$ and $Z$ axes once per 4D cell in registers.
  - Added struct-driven rotated FBM accumulator (`chaff_fbm_opts_t`, `chaff_fbm_rotated_2d`), periodic 2D sampler options (`chaff_periodic_opts_t`, `chaff_sample_periodic_2d_opts`), banded 2D curl flow (`chaff_eval_flow_2d_banded`), and 3D Ekman spiral atmospheric wind shear.
- **Universal 7-OS & NPM CI Pipeline (`.github/workflows/ci.yml`, `Makefile`)**:
  - Migrated `Makefile` to `include ../mk/include.mk` and added `.github/workflows/ci.yml` publishing `libchaff` across `deb,apk,rpm,pacman,brew,openbsd,winget` and `@tty-pt/chaff` to npm with `linux-x64`, `darwin-arm64`, `darwin-x64`, `win32-x64` N-API prebuilds and `chaff.wasm`.

### Changed
- Linked `chaff.wasm` / `noise.wasm` with shared memory and the `@tty-pt/stele` single-allocator substrate (`>=1.7.0` / `^1.7.0`).

## [1.0.0] - 2026-11-11

### Added
- Generic zero-magic-constant `<ttypt/chaff.h>`, `<ttypt/noise.h>`, and `<ttypt/chaff_glsl.h>` headers.
- Continuous Nyquist cutoff derivation from sampling stride (`chaff_nyquist_shift_q16`, `chaff_fill_lod`).
- Parameterized monotonic fractal amplitude scaling (`chaff_octave_amplitude_ex`), periodic 2D domain sampling (`chaff_wpoint_periodic_2d`), fixed-point 2D rotation (`chaff_rot2d_q15`, `chaff_rot2d_i32_q15`), rotated FBM accumulator (`chaff_fbm_rotated_2d_q12_4`), and declarative `CHAFF_LAYER_RIDGED` layer flag.
- Shared library (`libchaff.so`), WebAssembly module (`chaff.wasm`), and universal `@tty-pt/chaff` Node.js/Browser package under `BSD-2-Clause`.
