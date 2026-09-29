/* Run with: node learning/vm_playground/core.test.cjs
 * Or: node --test --test-isolation=none learning/vm_playground/core.test.cjs
 * Direct execution avoids restricted-shell child-process spawning on Windows.
 */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const C = require('./core.js');

function close(actual, expected, tolerance = 1e-12) {
  assert.ok(Number.isFinite(actual), `nonfinite value: ${actual}`);
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${actual} differs from ${expected} by more than ${tolerance}`);
}

test('polar and Cartesian responses agree over phases, angles and amplitudes', () => {
  const angles = C.grid(361);
  close(angles[0], -Math.PI);
  close(angles.at(-1), Math.PI);
  for (const kappa of [0, 1e-12, 0.001, 1, 5]) {
    for (const phase of [-4 * C.TAU, -Math.PI, -0.73, 0, Math.PI / 3, Math.PI, 3 * C.TAU]) {
      const { a, b } = C.coefficients(kappa, phase);
      close(Math.hypot(a, b), kappa, Math.max(kappa, 1e-12) * 1e-14);
      for (const theta of angles) {
        close(C.polar(theta, kappa, phase), C.cartesian(theta, a, b),
          Math.max(kappa, 1e-12) * 1e-13);
      }
    }
  }
  const worked = C.coefficients(2, Math.PI / 3);
  close(worked.a, 1);
  close(worked.b, -Math.sqrt(3));
});

test('preferred angle has the correct sign and is undefined only at zero amplitude', () => {
  assert.equal(C.preferred(0, 1.2), null);
  for (const phase of [-9.2, -Math.PI, -0.4, 0, Math.PI / 3, 8.4]) {
    for (const kappa of [1e-12, 0.001, 2]) {
      const preferred = C.preferred(kappa, phase);
      assert.ok(preferred >= 0 && preferred < C.TAU);
      close(C.polar(preferred, kappa, phase), kappa, kappa * 1e-13);
      close(C.polar(preferred + Math.PI, kappa, phase), -kappa, kappa * 1e-13);
    }
  }
  for (const angle of C.grid()) close(C.polar(angle, 0, 1.2), 0);
});

test('angle wrapping and core coordinate responses agree at and across the boundary', () => {
  for (const theta of [-Math.PI - 1e-9, -Math.PI, -1e-12, 0, Math.PI, C.TAU, 9.3]) {
    for (const turns of [-8, -1, 0, 1, 8]) {
      const shifted = theta + turns * C.TAU;
      const reduced = C.wrap(shifted);
      assert.ok(reduced >= 0 && reduced < C.TAU);
      close(Math.cos(reduced), Math.cos(theta));
      close(Math.sin(reduced), Math.sin(theta));
      close(C.polar(shifted, 2, 0.7), C.polar(theta, 2, 0.7));
      const { a, b } = C.coefficients(2, 0.7);
      close(C.cartesian(shifted, a, b), C.cartesian(theta, a, b));
    }
  }
});

test('integer harmonics preserve circular wrap and supply the repeated period', () => {
  for (const harmonic of [1, 2, 3, 4]) {
    for (const theta of C.grid(97)) {
      for (const kappa of [0, 0.001, 4, 12]) {
        for (const center of [-Math.PI, 0.27, Math.PI]) {
          for (const response of [C.sum, C.kernel, C.chordal]) {
            const baseline = response(theta, kappa, center, harmonic);
            close(response(theta + C.TAU, kappa, center, harmonic), baseline, 2e-12);
            close(response(theta, kappa, center + C.TAU, harmonic), baseline, 2e-12);
            close(response(theta + C.TAU / harmonic, kappa, center, harmonic), baseline, 2e-12);
          }
        }
      }
    }
    // A half-period shift distinguishes the positive-kappa peak from its trough.
    close(C.sum(Math.PI / harmonic, 4, 0, harmonic), -4);
    close(C.kernel(Math.PI / harmonic, 4, 0, harmonic), Math.exp(-8));
  }
  // A noninteger harmonic is deliberately outside the circular-wrap guarantee.
  assert.ok(Math.abs(C.sum(0, 2, 0, 1.5) - C.sum(C.TAU, 2, 0, 1.5)) > 3.9);
});

test('peak kernel equals the independent unit-circle Gaussian and exponential forms', () => {
  for (const theta of C.grid(121)) {
    for (const center of [-Math.PI, -0.61, 0, 2.4]) {
      for (const harmonic of [1, 2, 3, 4]) {
        for (const kappa of [0, 1e-12, 0.001, 2, 12]) {
          // Independent law-of-cosines distance, not core's sine-difference formula.
          const dot = Math.cos(harmonic * theta) * Math.cos(harmonic * center)
            + Math.sin(harmonic * theta) * Math.sin(harmonic * center);
          const squaredDistance = Math.max(0, 2 - 2 * dot);
          const gaussian = Math.exp(-kappa * squaredDistance / 2);
          const exponential = Math.exp(kappa * (Math.cos(harmonic * (theta - center)) - 1));
          const actual = C.kernel(theta, kappa, center, harmonic);
          close(actual, gaussian, 1e-13);
          close(actual, exponential, 1e-13);
          close(C.chordal(theta, kappa, center, harmonic), gaussian, 1e-13);
          assert.ok(actual > 0 && actual <= 1);
        }
      }
    }
  }
  close(C.kernel(Math.PI / 2, 2, 0), Math.exp(-2));
});

test('kappa scales only sum amplitude but narrows the peak-normalized exponential', () => {
  for (const harmonic of [1, 2, 3, 4]) {
    for (const theta of C.grid(101)) {
      for (const kappa of [0.001, 0.2, 1, 4, 12]) {
        close(C.sum(theta, kappa, 0.31, harmonic) / kappa, C.sum(theta, 1, 0.31, harmonic));
      }
    }
    for (const kappa of [0.001, 1, 4, 12]) {
      close(C.sum(Math.PI / (3 * harmonic), kappa, 0, harmonic) / kappa, 0.5);
      close(C.kernel(0, kappa, 0, harmonic), 1);
    }
    const offset = Math.PI / (3 * harmonic);
    assert.ok(C.kernel(offset, 1, 0, harmonic) > C.kernel(offset, 4, 0, harmonic));
    assert.ok(C.kernel(offset, 4, 0, harmonic) > C.kernel(offset, 12, 0, harmonic));
  }
  for (const theta of C.grid()) {
    close(C.sum(theta, 0, 0), 0);
    close(C.kernel(theta, 0, 0), 1);
  }
  // The constant kappa=0 kernel integrates to 2*pi, not the density integral 1.
  const points = C.grid(1001), step = points[1] - points[0];
  const integral = step * points.reduce((area, theta, i) =>
    area + C.kernel(theta, 0, 0) * (i === 0 || i === points.length - 1 ? 0.5 : 1), 0);
  close(integral, C.TAU, 1e-12);
});

test('each reset receives an independent default copy and cannot mutate the template', () => {
  assert.ok(Object.isFrozen(C.DEFAULTS));
  for (const lesson of ['coordinates', 'locality', 'history', 'selection']) {
    assert.ok(Object.isFrozen(C.DEFAULTS[lesson]));
    const first = C.defaults(lesson), original = { ...C.DEFAULTS[lesson] };
    const property = Object.keys(first)[0];
    first[property] = 'changed by learner';
    first.extra = true;
    const reset = C.defaults(lesson);
    assert.notEqual(reset, first);
    assert.deepEqual(reset, original);
    assert.deepEqual(C.DEFAULTS[lesson], original);
  }
});

const pool = C.selectionPool();
test('selection exercise uses a reproducible finite pool with nested development samples', () => {
  assert.deepEqual(C.selectionPool(), pool);
  assert.equal(pool.length, C.DEFAULTS.selection.repetitions);
  for (const row of pool) {
    for (const key of ['development16', 'development128', 'reserved']) {
      assert.equal(row[key].length, 64);
      assert.ok(row[key].every(value => Number.isFinite(value) && value > 0));
    }
    for (let candidate = 0; candidate < 64; candidate++) {
      // The first 16 squared residuals contribute to the 128-sample total.
      assert.ok(128 * row.development128[candidate] >= 16 * row.development16[candidate]);
    }
  }
  assert.notDeepEqual(pool[0].development16, pool[0].reserved);
});

test('selector respects candidate prefixes and uses first index for exact ties', () => {
  assert.equal(C.select([2, 1, 1, -999], 3), 1);
  assert.equal(C.select([2, 1, 1, -999], 1), 0);
  assert.equal(C.select([2, 1, 1, -999], 4), 3);
  assert.equal(C.select([0, 0, 0], 3), 0);
});

test('winner summaries equal independent argmins and nested minima never rise', () => {
  for (const development of [16, 128]) {
    const summary = C.selectionSummary(pool, development);
    let lastAverage = Infinity;
    for (const result of summary) {
      let devTotal = 0, heldTotal = 0;
      for (const [index, row] of pool.entries()) {
        const scores = row[`development${development}`].slice(0, result.count);
        const minimum = Math.min(...scores);
        const independentWinner = scores.indexOf(minimum);
        assert.equal(result.winners[index], independentWinner);
        devTotal += minimum;
        heldTotal += row.reserved[independentWinner];
        if (result.count > 1) {
          const previousCount = result.count / 4;
          assert.ok(minimum <= Math.min(...scores.slice(0, previousCount)));
        }
      }
      close(result.development, devTotal / pool.length);
      close(result.reserved, heldTotal / pool.length);
      assert.ok(result.development < lastAverage, 'authored pool means decrease strictly');
      lastAverage = result.development;
      // A fixed-seed engineering check, not a statistical significance claim.
      assert.ok(result.reserved > 0.9 && result.reserved < 1.1);
    }
  }
  const small = C.selectionSummary(pool, 16).at(-1).development;
  const large = C.selectionSummary(pool, 128).at(-1).development;
  assert.ok(large > small, 'more development observations reduce winner optimism in this pool');
});

test('reserved-score interventions cannot alter development selection or mutate the pool', () => {
  const before = JSON.stringify(pool);
  const intervention = pool.map(row => ({
    ...row, reserved: row.reserved.map((_, candidate) => 1000 + (63 - candidate) ** 2),
  }));
  for (const development of [16, 128]) {
    const original = C.selectionSummary(pool, development);
    const changed = C.selectionSummary(intervention, development);
    for (let i = 0; i < original.length; i++) {
      assert.deepEqual(original[i].winners, changed[i].winners);
      assert.equal(original[i].development, changed[i].development);
      assert.notEqual(original[i].reserved, changed[i].reserved);
    }
  }
  assert.equal(JSON.stringify(pool), before);
});

test('historical browser data is recursively frozen, independently of interactive defaults', () => {
  const context = vm.createContext({ window: {} });
  vm.runInContext(fs.readFileSync(path.join(__dirname, 'evidence.js'), 'utf8'), context);
  const evidence = context.window.VM_EVIDENCE;
  assert.equal(evidence.cells.length, 720);
  const inspect = object => {
    if (object && typeof object === 'object') {
      assert.ok(Object.isFrozen(object));
      Object.values(object).forEach(inspect);
    }
  };
  inspect(evidence);
  const original = evidence.cells[0].normalized_clean_mse;
  assert.throws(() => vm.runInContext(
    '"use strict"; window.VM_EVIDENCE.cells[0].normalized_clean_mse = 99;', context),
  /read only|Cannot assign/);
  assert.throws(() => vm.runInContext('"use strict"; window.VM_EVIDENCE = {};', context),
    /read only|Cannot assign/);
  assert.equal(evidence.cells[0].normalized_clean_mse, original);
  assert.equal(evidence.gate_passed, false);
});
