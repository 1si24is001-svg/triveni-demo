const T = require('./web/triveni.js');
const B = require('./web/bundle.json');
T.init(B);
(async () => {
  const res = await T.run(B.test);
  const py = B.python;
  const samePairs = res.pairs.length === py.pairs.length && res.pairs.every((p, i) => p[0] === py.pairs[i][0] && p[1] === py.pairs[i][1]);
  let d1 = 0, d3 = 0;
  if (samePairs) res.pairs.forEach((_, i) => { d1 = Math.max(d1, Math.abs(res.p1[i] - py.p1[i])); d3 = Math.max(d3, Math.abs(res.p3[i] - py.p3[i])); });
  let predDiff = 0; for (const id of Object.keys(py.pred)) { const a = (res.pred[id] || []).join(), b = py.pred[id].join(); if (a !== b) predDiff++; }
  const m = T.evaluate(res, B.test.truth);
  console.log('pairs identical:', samePairs, res.pairs.length, 'vs', py.pairs.length);
  console.log('max |p1 diff|', d1.toExponential(2), ' max |p3 diff|', d3.toExponential(2), ' entities with different prediction:', predDiff);
  console.log('JS  F0.5', m.f05.toFixed(4), 'P', m.precision.toFixed(4), 'R', m.recall.toFixed(4), 'blockRecall', m.blockingRecall.toFixed(4), m.country);
  console.log('PY  F0.5', py.metrics.f05.toFixed(4), 'P', py.metrics.precision.toFixed(4), 'R', py.metrics.recall.toFixed(4));
  console.log('runtime ms', res.ms);
})();
