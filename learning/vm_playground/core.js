/* Pure teaching calculations. No research imports, training, storage, or network. */
(function (root) {
  'use strict';
  const TAU = 2 * Math.PI;
  const DEFAULTS = Object.freeze({
    coordinates: Object.freeze({ kappa: 2, phase: Math.PI / 3, theta: 0 }),
    locality: Object.freeze({ kappa: 4, center: 0, harmonic: 1, period: 24, offset: Math.PI / 2 }),
    history: Object.freeze({ view: 'balanced', family: 'smooth', labels: 128, noise: 0.05, seconds: 2 }),
    selection: Object.freeze({ candidates: 16, development: 16, seed: 20260922, repetitions: 200, reserved: 256 })
  });
  const defaults = (lesson) => ({ ...DEFAULTS[lesson] });
  const wrap = (angle) => ((angle % TAU) + TAU) % TAU;
  const coefficients = (kappa, phase) => ({ a: kappa * Math.cos(phase), b: -kappa * Math.sin(phase) });
  const polar = (theta, kappa, phase) => kappa * Math.cos(theta + phase);
  const cartesian = (theta, a, b) => a * Math.cos(theta) + b * Math.sin(theta);
  const preferred = (kappa, phase) => kappa === 0 ? null : wrap(-phase);
  const sum = (theta, kappa, center, harmonic = 1) => kappa * Math.cos(harmonic * (theta - center));
  // Stable equivalent of exp(kappa * (cos(h*(theta-center)) - 1)).
  const kernel = (theta, kappa, center, harmonic = 1) => Math.exp(-2 * kappa * Math.sin(harmonic * (theta - center) / 2) ** 2);
  function chordal(theta, kappa, center, harmonic = 1) {
    const x = harmonic * theta, c = harmonic * center;
    return Math.exp(-kappa / 2 * ((Math.cos(x) - Math.cos(c)) ** 2 + (Math.sin(x) - Math.sin(c)) ** 2));
  }
  function grid(count = 361) {
    return Array.from({ length: count }, (_, i) => -Math.PI + TAU * i / (count - 1));
  }
  // Mulberry32 + Box-Muller: fixed authored independent equal-risk residuals.
  function random(seed) {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6D2B79F5) >>> 0;
      let t = state;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function normal(rng) {
    return Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(TAU * rng());
  }
  // Selection receives development scores only. First index breaks exact ties.
  function select(scores, count) {
    let best = 0;
    for (let i = 1; i < count; i++) if (scores[i] < scores[best]) best = i;
    return best;
  }
  function selectionPool() {
    const cfg = DEFAULTS.selection;
    const dev = random(cfg.seed), reserved = random(cfg.seed ^ 0x9e3779b9);
    const trials = [];
    for (let r = 0; r < cfg.repetitions; r++) {
      const row = { development16: [], development128: [], reserved: [] };
      for (let c = 0; c < 64; c++) {
        let small = 0, total = 0, held = 0;
        for (let j = 0; j < 128; j++) {
          const squared = normal(dev) ** 2;
          total += squared;
          if (j < 16) small += squared;
        }
        for (let j = 0; j < cfg.reserved; j++) held += normal(reserved) ** 2;
        row.development16.push(small / 16);
        row.development128.push(total / 128);
        row.reserved.push(held / cfg.reserved);
      }
      trials.push(row);
    }
    return trials;
  }
  function selectionSummary(pool, development = 16) {
    return [1, 4, 16, 64].map((count) => {
      let dev = 0, held = 0;
      const winners = [];
      for (const row of pool) {
        const scores = row['development' + development];
        const winner = select(scores, count);
        winners.push(winner);
        dev += scores[winner];
        held += row.reserved[winner];
      }
      return { count, development: dev / pool.length, reserved: held / pool.length, winners };
    });
  }
  const api = { TAU, DEFAULTS, defaults, wrap, coefficients, polar, cartesian, preferred, sum, kernel, chordal, grid, select, selectionPool, selectionSummary };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.VMCore = Object.freeze(api);
})(typeof globalThis !== 'undefined' ? globalThis : this);
