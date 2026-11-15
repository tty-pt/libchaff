# Changelog

All notable changes to `libchaff` (`@tty-pt/chaff`) are documented in this file.

## [Unreleased]

### Added
- **Periodic Tiling, Domain Skew & 4D Slice Stepping (`include/ttypt/chaff.h`, `include/ttypt/noise.h`, `include/ttypt/chaff_glsl.h`)**:
  - Wired `period_shift` and `skew[2]` across point samplers (`noise_wpoint_n_periodic`), corner caches (`noise_wcorners_periodic`), analytical 2D/3D/4D gradient samplers (`chaff_sample_grad_2d`, `chaff_sample_grad_3d`, `chaff_sample_grad_4d`), and GLSL shader contracts (`chaff_glsl.h`).
  - Added bulk 2D slice stepper across 4D fields (`chaff_fill_slice_4d`) pre-collapsing invariant $W$ and $Z$ axes once per 4D cell in registers.
  - Added struct-driven rotated FBM accumulator (`chaff_fbm_opts_t`, `chaff_fbm_rotated_2d`) and periodic 2D sampler options (`chaff_periodic_opts_t`, `chaff_sample_periodic_2d_opts`).

### Changed
- Linked `chaff.wasm` / `noise.wasm` with shared memory and the `@tty-pt/stele` single-allocator substrate (`>=1.7.0` / `^1.7.0`).

## [1.1.0] - 2026-11-13

### Added
- Banded 2D flow evaluation with smoothstep Nyquist transitions (`chaff_eval_flow_2d_banded`, `chaff_glsl_ref_eval_flow_2d_banded`).
- Calibrated 3D atmospheric Ekman spiral with veer angle up to 28 degrees.
- Upgraded substrate baseline to @tty-pt/stele@^1.6.0 (>=1.6.0).

## [1.0.0] - 2026-11-11

### Added
- Generic zero-magic-constant `<ttypt/chaff.h>`, `<ttypt/noise.h>`, and `<ttypt/chaff_glsl.h>` headers.
- Continuous Nyquist cutoff derivation from sampling stride (`chaff_nyquist_shift_q16`, `chaff_fill_lod`).
- Parameterized monotonic fractal amplitude scaling (`chaff_octave_amplitude_ex`), periodic 2D domain sampling (`chaff_wpoint_periodic_2d`), fixed-point 2D rotation (`chaff_rot2d_q15`, `chaff_rot2d_i32_q15`), rotated FBM accumulator (`chaff_fbm_rotated_2d_q12_4`), and declarative `CHAFF_LAYER_RIDGED` layer flag.
- Shared library (`libchaff.so`), WebAssembly module (`chaff.wasm`), and universal `@tty-pt/chaff` Node.js/Browser package under `BSD-2-Clause`.
