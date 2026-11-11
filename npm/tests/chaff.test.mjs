import test from "node:test";
import assert from "node:assert/strict";
import {
  chaffPoint1D,
  chaffPoint2D,
  chaffPoint3D,
  chaffPoint4D,
  chaffPointN,
  chaffOctaves,
  getChaffWasm
} from "../dist/index.js";

test("@tty-pt/chaff: point evaluation across 1D, 2D, 3D, 4D within [0..255]", () => {
  const v1 = chaffPoint1D(10, 42, 4);
  const v2 = chaffPoint2D(10, 20, 42, 4);
  const v3 = chaffPoint3D(10, 20, 30, 42, 4);
  const v4 = chaffPoint4D(10, 20, 30, 40, 42, 4);
  const vn = chaffPointN([10, 20, 30], 42, 4);

  assert.ok(v1 >= 0 && v1 <= 255, `v1 out of range: ${v1}`);
  assert.ok(v2 >= 0 && v2 <= 255, `v2 out of range: ${v2}`);
  assert.ok(v3 >= 0 && v3 <= 255, `v3 out of range: ${v3}`);
  assert.ok(v4 >= 0 && v4 <= 255, `v4 out of range: ${v4}`);
  assert.ok(vn >= 0 && vn <= 255, `vn out of range: ${vn}`);
});

test("@tty-pt/chaff: determinism across repeated point queries", () => {
  for (let i = -10; i <= 10; i++) {
    const a = chaffPoint3D(i, i * 2, i * 3, 99, 3);
    const b = chaffPoint3D(i, i * 2, i * 3, 99, 3);
    assert.equal(a, b, `Deterministic query failed at step ${i}`);
  }
});

test("@tty-pt/chaff: 3D point continuity", () => {
  let prev = chaffPoint3D(0, 0, 0, 1234, 4);
  for (let x = 1; x <= 64; x++) {
    const cur = chaffPoint3D(x, 0, 0, 1234, 4);
    const diff = Math.abs(cur - prev);
    assert.ok(diff <= 40, `Jump too steep between x=${x-1} and x=${x}: diff=${diff}`);
    prev = cur;
  }
});

test("@tty-pt/chaff: multi-octave matrix generation", () => {
  const oct2d = [{ x: 4, w: 1 }];
  const m2d = chaffOctaves(2, [0, 0], oct2d, 55, 4);
  assert.equal(m2d.length, 256);
  assert.ok(m2d[0] !== 0 || m2d[1] !== 0, "2D matrix empty");

  const oct3d = [{ x: 3, w: 1 }];
  const m3d = chaffOctaves(3, [0, 0, 0], oct3d, 55, 3);
  assert.equal(m3d.length, 512);
  assert.ok(m3d[0] !== 0 || m3d[1] !== 0, "3D matrix empty");
});

test("@tty-pt/chaff: WebAssembly exports are available and functional", () => {
  const wasm = getChaffWasm();
  assert.ok(wasm, "WASM instance not loaded");
  assert.equal(typeof wasm.exports.noise_wasm_point_2d, "function");
  const wasmVal = wasm.exports.noise_wasm_point_2d(15, 35, 777, 4);
  assert.ok(wasmVal >= 0 && wasmVal <= 255);
});
