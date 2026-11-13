import { loadWasmModule, loadNativeAddon, getSubstrate } from "@tty-pt/stele";

const nativeBinding = loadNativeAddon(import.meta.url, "chaff") || loadNativeAddon(import.meta.url, "noise");
let wasmInstance = null;

export function getChaffWasm() {
  if (wasmInstance) return wasmInstance;
  wasmInstance = loadWasmModule(new URL("./chaff.wasm", import.meta.url));
  return wasmInstance;
}

export const getNoiseWasm = getChaffWasm;

export function chaffPoint1D(x, seed, shift, options = {}) {
  if (!options.wasm && nativeBinding && nativeBinding.point1D) {
    return nativeBinding.point1D(x, seed, shift);
  }
  const wasm = getChaffWasm();
  return wasm.exports.noise_wasm_point_1d(x, seed, shift);
}

export function chaffPoint2D(x, y, seed, shift, options = {}) {
  if (!options.wasm && nativeBinding && nativeBinding.point2D) {
    return nativeBinding.point2D(x, y, seed, shift);
  }
  const wasm = getChaffWasm();
  return wasm.exports.noise_wasm_point_2d(x, y, seed, shift);
}

export function chaffPoint3D(x, y, z, seed, shift, options = {}) {
  if (!options.wasm && nativeBinding && nativeBinding.point3D) {
    return nativeBinding.point3D(x, y, z, seed, shift);
  }
  const wasm = getChaffWasm();
  return wasm.exports.noise_wasm_point_3d(x, y, z, seed, shift);
}

export function chaffPoint4D(x, y, z, w, seed, shift, options = {}) {
  if (!options.wasm && nativeBinding && nativeBinding.point4D) {
    return nativeBinding.point4D(x, y, z, w, seed, shift);
  }
  const wasm = getChaffWasm();
  return wasm.exports.noise_wasm_point_4d(x, y, z, w, seed, shift);
}

export function chaffPointN(coords, seed, shift, options = {}) {
  if (!options.wasm && nativeBinding && nativeBinding.pointN) {
    return nativeBinding.pointN(coords, seed, shift);
  }
  const wasm = getChaffWasm();
  const substrate = getSubstrate();
  const ptr = wasm.exports.malloc(coords.length * 2);
  const view = new Int16Array(substrate.memory.buffer, ptr, coords.length);
  for (let i = 0; i < coords.length; i++) view[i] = coords[i];
  const res = wasm.exports.noise_wasm_point_n(ptr, seed, shift, coords.length);
  wasm.exports.free(ptr);
  return res;
}

export function chaffOctaves(dim, origin, octaves, seed, cy, options = {}) {
  if (!options.wasm && nativeBinding && nativeBinding.octaves) {
    return nativeBinding.octaves(dim, origin, octaves, seed, cy);
  }
  const wasm = getChaffWasm();
  const substrate = getSubstrate();
  const total = 1 << (dim * cy);
  const octCount = Math.min(octaves.length, 16);
  const outPtr = wasm.exports.malloc(total * 4);
  const origPtr = wasm.exports.malloc(dim * 2);
  const octPtr = wasm.exports.malloc(octCount * 8);

  const origView = new Int16Array(substrate.memory.buffer, origPtr, dim);
  for (let i = 0; i < dim; i++) origView[i] = origin[i] || 0;

  const octView = new Uint32Array(substrate.memory.buffer, octPtr, octCount * 2);
  for (let i = 0; i < octCount; i++) {
    octView[i * 2] = octaves[i].x;
    octView[i * 2 + 1] = octaves[i].w;
  }

  wasm.exports.noise_wasm_octaves(outPtr, origPtr, octCount, octPtr, seed, cy, dim);
  const res = new Uint32Array(substrate.memory.buffer, outPtr, total).slice();

  wasm.exports.free(outPtr);
  wasm.exports.free(origPtr);
  wasm.exports.free(octPtr);

  return res;
}

export const noisePoint1D = chaffPoint1D;
export const noisePoint2D = chaffPoint2D;
export const noisePoint3D = chaffPoint3D;
export const noisePoint4D = chaffPoint4D;
export const noisePointN = chaffPointN;
export const noiseOctaves = chaffOctaves;
