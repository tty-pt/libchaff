# @tty-pt/chaff

Universal Node.js and Browser WebAssembly bindings for `libchaff` (`<ttypt/chaff.h>`, `<ttypt/noise.h>`), an N-dimensional procedural noise engine with Hermite smoothstep interpolation, periodic domain wrapping, analytical gradients, and hypercube matrix stepping.

## Features
- **Point & Analytical Gradient Queries ($O(1)$)**: Zero-allocation continuous noise and analytical gradient queries across 1D, 2D, 3D, and 4D (`chaff_sample_grad_2d/3d/4d`).
- **Matrix & 3D/4D Slice Stepping**: Bulk hypercube (`noiseOctaves`), 3D slice (`chaff_fill_slice_3d`), and 4D spatiotemporal slice (`chaff_fill_slice_4d`) generation without per-point function call overhead.
- **Periodic Tiling, Domain Skew & Rotated FBM**: Power-of-two toroidal wrapping (`period_shift`), anisotropic shearing (`skew[2]`), and domain-rotated 2D FBM (`chaff_fbm_rotated_2d`).
- **Dual Runtime**: Native C execution in Node.js via Node-API, and high-performance shared-memory WebAssembly in the browser via `@tty-pt/stele`.

## Usage
```javascript
import { noisePoint2D, noisePoint3D, noiseOctaves } from "@tty-pt/chaff";

// 2D continuous point query (returns 0..255)
const val2d = noisePoint2D(100, 200, 42 /* seed */, 4 /* shift */);

// 3D continuous point query
const val3d = noisePoint3D(10, 20, 30, 42, 4);

// Multi-octave 2D matrix (16x16 = 256 cells)
const grid = noiseOctaves(2, [0, 0], [{ x: 4, w: 1 }], 42, 4);
```

## License
BSD-2-Clause — Copyright (c) 2026, tty-pt. See `LICENSE`.
