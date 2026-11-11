# Changelog

All notable changes to `libchaff` (`@tty-pt/chaff`) are documented in this file.

## [1.0.0] - 2026-11-11

### Added
- Generic zero-magic-constant `<ttypt/chaff.h>`, `<ttypt/noise.h>`, and `<ttypt/chaff_glsl.h>` headers.
- Continuous Nyquist cutoff derivation from sampling stride (`chaff_nyquist_shift_q16`, `chaff_fill_lod`).
- Parameterized monotonic fractal amplitude scaling (`chaff_octave_amplitude_ex`), periodic 2D domain sampling (`chaff_wpoint_periodic_2d`), fixed-point 2D rotation (`chaff_rot2d_q15`, `chaff_rot2d_i32_q15`), rotated FBM accumulator (`chaff_fbm_rotated_2d_q12_4`), and declarative `CHAFF_LAYER_RIDGED` layer flag.
- Shared library (`libchaff.so`), WebAssembly module (`chaff.wasm`), and universal `@tty-pt/chaff` Node.js/Browser package under `BSD-2-Clause`.
