/* View-only controls. All numbers and scientific decisions come from the export. */
(() => {
  'use strict';
  const evidence = globalThis.VMResearchEvidence;
  const byId = id => document.getElementById(id);
  const put = (id, text) => { byId(id).textContent = text; };
  let active = 0;

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function anchor(text, path) {
    const node = element('a', text); node.href = path; return node;
  }
  function format(metric, value) {
    return (value * metric.scale).toFixed(metric.decimals) + (metric.unit === '%' ? '%' : '');
  }
  function renderView() {
    const study = evidence.studies[active];
    const view = study.views[Number(byId('researchView').value)];
    put('viewDescription', view.description);
    const metrics = byId('researchMetrics'); metrics.replaceChildren();
    for (const metric of view.metrics) {
      const card = element('div', undefined, 'research-metric');
      card.append(element('span', metric.label));
      const value = element('strong');
      value.append(anchor(format(metric, metric.value), evidence.sources[metric.source].href));
      if (metric.unit && metric.unit !== '%') value.append(element('span', metric.unit, 'unit'));
      card.append(value);
      if (metric.interval) card.append(element('span', `Saved 95% interval: ${format(metric, metric.interval[0])} to ${format(metric, metric.interval[1])}`, 'interval'));
      metrics.append(card);
    }
    metrics.hidden = view.metrics.length === 0;
    const sources = byId('researchSources'); sources.replaceChildren();
    const refs = [{source: study.report, pointer: 'Original report'}];
    if (study.gate) refs.push(study.gate);
    for (const metric of view.metrics) {
      refs.push(metric);
      if (metric.interval_pointer) refs.push({source: metric.source, pointer: metric.interval_pointer});
    }
    for (const ref of refs) {
      const source = evidence.sources[ref.source];
      const row = element('div', undefined, 'source-entry');
      row.append(anchor(source.path, source.href), element('br'), element('code', ref.pointer), element('br'), element('code', `SHA256 ${source.sha256}`));
      sources.append(row);
    }
  }
  function selectStudy(index) {
    active = index;
    const study = evidence.studies[index];
    document.querySelectorAll('[data-study]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.study) === index)));
    put('studyId', `${study.id} / ${study.title}`);
    put('studyTitle', study.question); put('studyScope', study.scope);
    put('studyStatus', study.status);
    byId('studyStatus').classList.toggle('neutral', study.gate === null);
    for (const key of ['finding', 'limit', 'engineering']) put(key, study[key]);
    put('selfPrompt', study.prompt); put('selfAnswer', study.answer);
    for (const id of ['selfCheck', 'engineeringDetails', 'provenanceDetails']) byId(id).open = false;
    const selector = byId('researchView'); selector.replaceChildren();
    study.views.forEach((view, i) => { const option = element('option', view.label); option.value = i; selector.append(option); });
    selector.value = '0';
    const links = byId('researchLinks'); links.replaceChildren();
    links.append(anchor('Read the original report', evidence.sources[study.report].href));
    for (const link of study.links) links.append(anchor(link.label, '../../' + link.path));
    renderView();
  }
  evidence.studies.forEach((study, index) => {
    const button = element('button'); button.type = 'button'; button.dataset.study = index;
    button.setAttribute('aria-controls', 'studyPanel');
    button.append(element('span', study.id), document.createTextNode(study.title));
    button.addEventListener('click', () => selectStudy(index)); byId('studyNav').append(button);
  });
  byId('researchView').addEventListener('change', renderView);
  byId('resetResearch').addEventListener('click', () => selectStudy(0));
  selectStudy(0);
})();
