/* DOM wiring and SVG views for the isolated VM-N03 teaching page. */
(() => {
  'use strict';
  const C = window.VMCore, E = window.VM_EVIDENCE;
  const $ = (id) => document.getElementById(id);
  const numeric = (id) => Number($(id).value);
  const f = (x, digits = 4) => (Math.abs(x) < 0.5 * 10 ** -digits ? 0 : x).toFixed(digits);
  const escape = (x) => String(x).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const colors = { teal: '#007f7b', amber: '#ab5816', gray: '#819397', purple: '#745099' };
  let syntheticPool;
  function metric(value, label) { return `<div class="metric"><strong>${value}</strong><span>${label}</span></div>`; }
  function source(ref, label, field = '') {
    const item = E.sources[ref.id];
    const location = ref.line ? `CSV line ${ref.line}; record ${ref.json_pointer}` : (ref.pointer || '') + field;
    return `<a href="${item.href}" title="${escape(item.path + ' · ' + location)}">${label}</a>`;
  }
  function linePlot(target, lines, { title, ymin, ymax, ylabel = 'Response (arbitrary units)', probe = null, xvalues = C.grid(), ticks = null, xlabel = 'Input angle θ (radians)' }) {
    const compact = target === 'sumPlot' || target === 'kernelPlot';
    const width = compact ? 420 : 760, height = 310, left = compact ? 52 : 65, right = compact ? 15 : 25, top = 26, bottom = 60;
    const xmin = xvalues[0], xmax = xvalues[xvalues.length - 1];
    const px = (x) => left + (x - xmin) / (xmax - xmin) * (width - left - right);
    const py = (y) => top + (ymax - y) / (ymax - ymin) * (height - top - bottom);
    const xticks = ticks || [[-Math.PI, '−π'], [-Math.PI / 2, '−π/2'], [0, '0'], [Math.PI / 2, 'π/2'], [Math.PI, 'π']];
    const yticks = [ymin, (ymin + ymax) / 2, ymax];
    let svg = `<svg class="plot" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${target}Title ${target}Desc"><title id="${target}Title">${escape(title)}</title><desc id="${target}Desc">${escape(ylabel)} versus ${escape(xlabel)}. Numeric readings and explanation follow the plot.</desc>`;
    for (const y of yticks) svg += `<line class="axis" x1="${left}" x2="${width - right}" y1="${py(y)}" y2="${py(y)}"/><text x="${left - 10}" y="${py(y) + 5}" text-anchor="end">${f(y, 2)}</text>`;
    for (const [x, label] of xticks) svg += `<text x="${px(x)}" y="${height - bottom + 26}" text-anchor="middle">${escape(label)}</text>`;
    svg += `<text x="${left}" y="16">${escape(ylabel)}</text><text x="${width / 2}" y="${height - 10}" text-anchor="middle">${escape(xlabel)}</text>`;
    for (const line of lines) {
      const values = xvalues.map((x, i) => line.values ? line.values[i] : line.fn(x));
      const path = values.map((y, i) => `${i ? 'L' : 'M'}${px(xvalues[i]).toFixed(2)},${py(y).toFixed(2)}`).join(' ');
      svg += `<path d="${path}" fill="none" stroke="${colors[line.color]}" stroke-width="${line.dash ? 2 : 3}"${line.dash ? ' stroke-dasharray="7 6"' : ''}/>`;
      if (xvalues.length < 10) values.forEach((y, i) => { svg += `<circle cx="${px(xvalues[i])}" cy="${py(y)}" r="4" fill="${colors[line.color]}"/>`; });
    }
    if (probe !== null) svg += `<line x1="${px(probe)}" x2="${px(probe)}" y1="${top}" y2="${height - bottom}" stroke="#5b737a" stroke-dasharray="3 5"/>`;
    $(target).innerHTML = svg + '</svg>';
  }
  function coordinates() {
    const k = numeric('amplitude'), mu = numeric('phase'), theta = numeric('probe');
    const { a, b } = C.coefficients(k, mu), peak = C.preferred(k, mu);
    $('amplitudeValue').textContent = f(k, 3);
    $('phaseValue').textContent = f(mu) + ' rad';
    $('probeValue').textContent = f(theta) + ' rad';
    $('coordinateMetrics').innerHTML = metric(f(a), 'a · cosine coefficient') + metric(f(b), 'b · sine coefficient') + metric(peak === null ? 'Undefined' : f(peak) + ' rad', 'Preferred angle in [0, 2π)');
    linePlot('coordinatePlot', [{ fn: (x) => C.polar(x, k, mu), color: 'teal' }, { fn: (x) => C.cartesian(x, a, b), color: 'amber', dash: true }], { title: 'Polar and Cartesian functions overlap', ymin: -5, ymax: 5, probe: theta });
    const error = Math.max(...C.grid().map((x) => Math.abs(C.polar(x, k, mu) - C.cartesian(x, a, b))));
    const value = C.polar(theta, k, mu), wrapped = C.polar(theta + C.TAU, k, mu);
    $('coordinateReadout').textContent = `At θ: ${f(value, 6)} · at θ + 2π: ${f(wrapped, 6)} · wrap difference: ${Math.abs(value - wrapped).toExponential(1)}. Maximum coordinate discrepancy on 361 plot points: ${error.toExponential(1)}.`;
    $('phaseNote').textContent = k === 0 ? 'Zero amplitude: every angle ties. The phase setting changes no response and does not define a preferred direction.' : k < 0.01 ? 'Near-zero amplitude: the formal maximum still exists, but phase describes a negligible response; small coefficient perturbations can change it substantially.' : 'Observe the negative sign: a positive μ shifts the preferred input toward −μ modulo 2π. The plot keeps a fixed vertical scale.';
  }
  function locality() {
    const k = numeric('concentration'), center = numeric('center'), h = numeric('harmonic'), period = numeric('period'), offset = numeric('offset');
    $('concentrationValue').textContent = f(k, 2);
    $('centerValue').textContent = f(center) + ' rad';
    $('offsetValue').textContent = f(offset) + ' rad';
    linePlot('sumPlot', [{ fn: (x) => C.sum(x, 1, center, h), color: 'gray' }, { fn: (x) => C.sum(x, k, center, h), color: 'teal' }], { title: 'Kappa scales sum height without narrowing it', ymin: -12, ymax: 12, ylabel: 'Sum (arbitrary units)' });
    linePlot('kernelPlot', [{ fn: (x) => C.kernel(x, 1, center, h), color: 'gray' }, { fn: (x) => C.kernel(x, k, center, h), color: 'teal' }, { fn: (x) => C.chordal(x, k, center, h), color: 'amber', dash: true }], { title: 'Peak-normalized kernel and chordal Gaussian overlap', ymin: 0, ymax: 1, ylabel: 'Kernel (dimensionless)' });
    $('localityReadout').textContent = k === 0 ? `κ = 0: zero sum and constant-one kernel. No unique peak or fundamental period; T/h = ${f(period / h, 2)} hours is only the input feature’s repetition scale.` : `Sum amplitude: ${f(k, 2)}. Kernel peak: 1. Repetition: 2π/${h} = ${f(C.TAU / h)} rad, or ${f(period / h, 2)} hours for θ(t) = 2πt/${period}. ${h > 1 ? h + ' peaks per input circle.' : 'One peak per input circle.'}`;
    const theta = center + offset;
    const distance = (Math.cos(h * theta) - Math.cos(h * center)) ** 2 + (Math.sin(h * theta) - Math.sin(h * center)) ** 2;
    const v = C.kernel(theta, k, center, h), g = C.chordal(theta, k, center, h);
    const max = Math.max(...C.grid().map((x) => Math.abs(C.kernel(x, k, center, h) - C.chordal(x, k, center, h))));
    $('identityReadout').textContent = `At the probe: chord distance² = ${f(distance, 6)}; circular K = ${f(v, 6)}; Gaussian K = ${f(g, 6)}. Maximum identity discrepancy on 361 plot points: ${max.toExponential(1)}.`;
  }
  function history() {
    const view = $('historyView').value, family = $('family').value, labels = numeric('labels'), noise = numeric('noise'), seconds = numeric('seconds');
    $('family').disabled = view === 'balanced';
    $('labels').disabled = $('noise').disabled = view !== 'cell';
    const rows = E.models.map((model) => {
      const row = view === 'cell' ? E.cells.find((c) => c.model_id === model.id && c.family === family && c.total_labels === labels && c.noise_std === noise && c.allowance_seconds === seconds) : E.summaries.find((s) => s.model_id === model.id && s.allowance_seconds === seconds);
      const error = view === 'cell' ? row.normalized_clean_mse : view === 'family' ? row.family_errors[family] : row.balanced_periodic_error;
      return { model, row, error };
    });
    const split = E.label_splits[String(labels)];
    const description = view === 'balanced' ? 'Equal-weight mean over smooth, narrow and multimodal families, averaging labels, noise and initializations within each function.' : view === 'family' ? `${family} family, averaging all label budgets, noise levels and initializations within each function.` : `${family} family · ${labels} labels = ${split.train} training + ${split.validation} validation · noise SD ${noise} (synthetic target units) · averages eight functions and two initializations.`;
    $('historyDescription').textContent = `${seconds}-second allowance · ${description} ${family === 'nonperiodic' && view !== 'balanced' ? 'Diagnostic only: excluded from the locked primary gate.' : ''}`;
    const max = Math.max(...rows.map((r) => r.error));
    $('historyTable').innerHTML = `<table><caption>Saved normalized clean MSE (dimensionless; lower is better). Bars share a linear scale within this view; scales change between views.</caption><thead><tr><th scope="col">Model</th><th scope="col">Features / fitted scalars</th><th scope="col">Error · source link</th><th scope="col" class="bar-cell">Within-view comparison</th></tr></thead><tbody>${rows.map(({ model, row, error }) => `<tr${model.id === 'vp_both' ? ' class="selected"' : ''}><th scope="row">${model.id}</th><td class="numeric">${source(model.source, model.features + ' / ' + model.parameters)}</td><td class="numeric">${source(row.source, f(error, 8), view === 'balanced' ? '/balanced_periodic_error' : view === 'family' ? '/family_errors/' + family : '')}</td><td class="bar-cell"><div class="bar" style="width:${100 * error / max}%;background:${model.id === 'vp_both' ? colors.teal : model.id === 'joint_both' ? colors.purple : colors.gray}" aria-hidden="true"></div></td></tr>`).join('')}</tbody></table>`;
  }
  function historicalFixed() {
    const v16 = E.co_primary.fixed16, v48 = E.co_primary.fixed48, joint = E.optimizer_contrast;
    const percent = (x) => f(x * 100, 2) + '%';
    $('gateCards').innerHTML = metric(source(v16.source, percent(v16.relative_improvement), '/relative_improvement'), 'VP error reduction vs fixed16 · equal basis count') + metric(source(v16.source, '+' + percent(-v16.family_improvements.smooth), '/family_improvements/smooth'), 'Smooth error increase vs fixed16 · FAIL') + metric(source(v48.source, percent(v48.relative_improvement), '/relative_improvement'), 'VP error reduction vs fixed48 · equal scalar count');
    $('gateCards').insertAdjacentHTML('afterend', `<p class="note">VP vs fixed16: ${source(v16.source, '95% paired interval ' + percent(v16.bootstrap.lower) + ' to ' + percent(v16.bootstrap.upper), '/bootstrap')}. VP vs fixed48: ${source(v48.source, percent(v48.bootstrap.lower) + ' to ' + percent(v48.bootstrap.upper), '/bootstrap')}. VP error was ${source(joint.source, percent(-joint.relative_improvement) + ' higher than joint Adam', '/relative_improvement')}.</p>`);
    $('runtimeTable').innerHTML = `<table><caption>Historical confirmation runtime · all recorded fits per model</caption><thead><tr><th>Model</th><th>Median total fit (seconds)</th><th>Median completed updates</th></tr></thead><tbody>${E.models.map((m) => { const r = E.runtime_by_model[m.id]; return `<tr><th scope="row">${m.id}</th><td class="numeric">${source(r.source, f(r.median_fit_seconds, 5), '/median_fit_seconds')}</td><td class="numeric">${source(r.source, f(r.median_updates, 0), '/median_updates')}</td></tr>`; }).join('')}</tbody></table>`;
  }
  function selection() {
    if (!syntheticPool) syntheticPool = C.selectionPool();
    const development = numeric('development'), count = numeric('candidates');
    const rows = C.selectionSummary(syntheticPool, development), active = rows.find((r) => r.count === count);
    linePlot('selectionPlot', [{ values: rows.map((r) => r.development), color: 'teal' }, { values: rows.map((r) => r.reserved), color: 'purple' }, { values: rows.map(() => 1), color: 'gray', dash: true }], { title: 'Selection on development scores creates optimism', ymin: 0, ymax: 1.2, ylabel: 'Mean MSE (arbitrary squared units)', xvalues: [0, 1, 2, 3], ticks: [[0, '1'], [1, '4'], [2, '16'], [3, '64']], xlabel: 'Candidates searched (equally spaced log₄ scale)', probe: [1, 4, 16, 64].indexOf(count) });
    $('selectionTable').innerHTML = `<table><caption>New authored toy scores · ${development} development observations · average of 200 repetitions</caption><thead><tr><th>Candidates</th><th>Selected development MSE</th><th>Same-candidate reserved MSE</th></tr></thead><tbody>${rows.map((r) => `<tr${r.count === count ? ' class="selected"' : ''}><th scope="row">${r.count}</th><td class="numeric">${f(r.development)}</td><td class="numeric">${f(r.reserved)}</td></tr>`).join('')}</tbody></table>`;
    $('selectionReadout').textContent = `Searching ${count}: development ${f(active.development)}, reserved ${f(active.reserved)}. Their gap is ${f(active.reserved - active.development)} squared units. Every candidate’s true risk remains exactly 1; changing this view does not select again on reserved scores.`;
  }
  function setValues(values) { for (const [id, value] of Object.entries(values)) $(id).value = value; }
  function reset(number) {
    const area = $('lesson' + number);
    area.querySelectorAll('details').forEach((item) => { item.open = false; });
    area.querySelectorAll('.prompt select').forEach((item) => { item.selectedIndex = 0; });
    $('workspace' + number).hidden = true;
    area.querySelector('[data-reveal]').setAttribute('aria-expanded', 'false');
    if (number === 1) {
      const d = C.defaults('coordinates'); setValues({ amplitude: d.kappa, phase: d.phase, probe: d.theta }); coordinates();
    } else if (number === 2) {
      const d = C.defaults('locality'); setValues({ concentration: d.kappa, center: d.center, harmonic: d.harmonic, period: d.period, offset: d.offset }); locality();
    } else {
      const d = C.defaults('history'), s = C.defaults('selection');
      setValues({ historyView: d.view, family: d.family, labels: d.labels, noise: d.noise, seconds: d.seconds, candidates: s.candidates, development: s.development });
      $('selectionWorkspace').hidden = true;
      $('revealSelection').setAttribute('aria-expanded', 'false');
      // Clear any revealed synthetic results so reset returns to the initial DOM.
      for (const id of ['selectionPlot', 'selectionTable', 'selectionReadout']) $(id).replaceChildren();
      history();
    }
  }
  document.querySelectorAll('[data-lesson]').forEach((button) => button.addEventListener('click', () => {
    document.querySelectorAll('[data-lesson]').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
    document.querySelectorAll('.lesson').forEach((item) => { item.hidden = item.id !== 'lesson' + button.dataset.lesson; });
  }));
  document.querySelectorAll('[data-reveal]').forEach((button) => {
    button.setAttribute('aria-expanded', 'false'); button.setAttribute('aria-controls', 'workspace' + button.dataset.reveal);
    button.addEventListener('click', () => { $('workspace' + button.dataset.reveal).hidden = false; button.setAttribute('aria-expanded', 'true'); });
  });
  document.querySelectorAll('[data-reset]').forEach((button) => button.addEventListener('click', () => reset(Number(button.dataset.reset))));
  for (const id of ['amplitude', 'phase', 'probe']) $(id).addEventListener('input', coordinates);
  for (const id of ['concentration', 'center', 'harmonic', 'period', 'offset']) $(id).addEventListener('input', locality);
  for (const id of ['historyView', 'family', 'labels', 'noise', 'seconds']) $(id).addEventListener('change', history);
  for (const id of ['candidates', 'development']) $(id).addEventListener('change', selection);
  $('zero').addEventListener('click', () => { $('amplitude').value = 0; coordinates(); });
  $('nearzero').addEventListener('click', () => { $('amplitude').value = 0.001; coordinates(); });
  $('boundary').addEventListener('click', () => { $('probe').value = Math.PI; coordinates(); });
  $('revealSelection').setAttribute('aria-controls', 'selectionWorkspace');
  $('revealSelection').addEventListener('click', () => { $('selectionWorkspace').hidden = false; $('revealSelection').setAttribute('aria-expanded', 'true'); selection(); });
  historicalFixed();
  [1, 2, 3].forEach(reset);
})();
