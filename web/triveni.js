/* TRIVENI engine: a line-for-line port of demo_core.py, normalize.py,
   translit.py and decode.py, plus LightGBM tree evaluation. Runs in the
   browser and in Node (for the parity test against Python). */
(function (root) {
  'use strict';
  let TB, IDF, DFLT, B;

  function init(bundle) {
    B = bundle; TB = bundle.tables; IDF = bundle.idf; DFLT = bundle.idf_default;
    TB.loanRx = TB.loan.map(([p, r]) => [new RegExp(p), r]);
    TB.suffixSet = new Set(TB.suffixes);
    TB.stopSet = new Set(TB.leading_stop);
    TB.webSet = new Set(TB.web_tokens);
    TB.landmarkRx = new RegExp(TB.landmark_re, 'gi');
    TB.dbaSplit = new RegExp(TB.dba_re, 'i');
    TB.dbaFull = new RegExp('^(?:' + TB.dba_re + ')$', 'i');
  }

  // ------------------------------------------------------------ transliteration
  const DEVA = /[\u0900-\u097F]/, ANY_INDIC = /[\u0900-\u0D7F]/;
  function toDevanagari(text) {
    let out = '';
    for (const ch of text) {
      if (ch in TB.chillu) { out += TB.chillu[ch]; continue; }
      const cp = ch.codePointAt(0);
      if (cp === 0x0B9A) out += 'स';
      else if (cp >= 0x0980 && cp <= 0x0D7F) out += String.fromCodePoint(0x0900 + (cp - 0x0980) % 0x80);
      else out += ch;
    }
    return out;
  }
  function word(w) {
    w = w.split('ऱ्ऱ').join('ट्ट');
    const chars = Array.from(w), out = [];
    let i = 0;
    while (i < chars.length) {
      let c = chars[i];
      let nxt = i + 1 < chars.length ? chars[i + 1] : '';
      if (nxt === TB.nukta && c in TB.nukta_base) {
        c = TB.nukta_base[c]; i += 1;
        nxt = i + 1 < chars.length ? chars[i + 1] : '';
      }
      if (c in TB.cons) {
        out.push(TB.cons[c]);
        if (nxt in TB.matras) { out.push(TB.matras[nxt]); i += 1; }
        else if (nxt === TB.virama) { i += 1; }
        else out.push('a');
      } else if (c in TB.vowels) out.push(TB.vowels[c]);
      else if (c in TB.marks) out.push(TB.marks[c]);
      else if (c in TB.digits) out.push(TB.digits[c]);
      else if (c === TB.nukta) { /* skip */ }
      else out.push(c);
      i += 1;
    }
    let s = out.join('');
    const last = chars.length ? chars[chars.length - 1] : 'x';
    if (s.length > 2 && s.endsWith('a') && !s.endsWith('aa') && DEVA.test(last)) s = s.slice(0, -1);
    return s.split('aa').join('a').split('ee').join('i').split('oo').join('u');
  }
  function loan(w) { for (const [rx, rep] of TB.loanRx) if (rx.test(w)) return rep; return w; }
  function transliterate(text) {
    if (typeof text !== 'string' || !ANY_INDIC.test(text)) return text;
    text = toDevanagari(text.replace(/[\u200c\u200d]/g, ''));
    const words = text.split(/\s+/).filter(Boolean).map(w => DEVA.test(w) ? loan(word(w)) : w);
    return words.join(' ').replace(/\bpra\.?\s+li\.?(?=\s|$)/g, 'private limited');
  }

  // ------------------------------------------------------------ normalization
  function fold(text) {
    if (typeof text !== 'string') return '';
    text = transliterate(text.normalize('NFC')).normalize('NFKD').replace(/\p{Mn}/gu, '');
    return text.toLowerCase().split('&').join(' and ').split('+').join(' and ');
  }
  const clean = t => t.replace(/[^a-z0-9 ]+/g, ' ').replace(/\s+/g, ' ').trim();
  const SK = [['ksh', 'ks'], ['x', 'ks'], ['ph', 'f'], ['sh', 's'], ['kh', 'k'], ['gh', 'g'], ['th', 't'],
    ['dh', 'd'], ['bh', 'b'], ['ch', 'c'], ['ck', 'k'], ['q', 'k'], ['w', 'v'], ['z', 'j'], ['c', 'k']];
  function skeleton(token) {
    let t = token;
    for (const [a, b] of SK) t = t.split(a).join(b);
    if (!t) return t;
    t = t.split('g').join('k').split('b').join('p').split('d').join('t');
    t = t[0] + t.slice(1).replace(/[aeiouyh]/g, '');
    return t.replace(/(.)\1+/g, '$1');
  }
  const splitWs = s => s.split(/\s+/).filter(Boolean);
  const ab = (m, t) => (Object.prototype.hasOwnProperty.call(m, t) ? m[t] : t);

  function normName(raw) {
    const text = fold(raw);
    const parts = text.split(TB.dbaSplit).filter(p => p && !TB.dbaFull.test(p));
    const variants = [];
    for (const p of (parts.length ? parts : [text])) {
      const isWeb = /\.(com|net|org|in|co|fr|biz|info|io)\b|www\./.test(p);
      let toks = splitWs(splitWs(clean(p)).map(t => ab(TB.name_abbrev, t)).join(' '));
      if (isWeb) { const f = toks.filter(t => !TB.webSet.has(t)); toks = f.length ? f : toks; }
      const merged = []; let run = '';
      for (const t of toks) {
        if (t.length === 1 && /[a-z]/.test(t)) { run += t; continue; }
        if (run) { merged.push(run.length > 1 ? ab(TB.name_abbrev, run) : run); run = ''; }
        merged.push(t);
      }
      if (run) merged.push(run.length > 1 ? ab(TB.name_abbrev, run) : run);
      toks = merged;
      while (toks.length > 1 && TB.stopSet.has(toks[0])) toks = toks.slice(1);
      const suffix = toks.filter(t => TB.suffixSet.has(t));
      const core = toks.filter(t => !TB.suffixSet.has(t));
      variants.push([core.length ? core : toks, suffix]);
    }
    const [coreT, sufT] = variants[0];
    return { core: coreT.join(' '), alt: variants.length > 1 ? variants[1][0].join(' ') : '',
      suffix: sufT.join(' '), skel: coreT.map(skeleton).join(' '),
      acr: coreT.filter(Boolean).map(t => t[0]).join(''), translit: transliterate(raw || '') };
  }

  function normAddress(raw) {
    let text = fold(raw);
    const landmarks = [];
    TB.landmarkRx.lastIndex = 0;
    let m;
    while ((m = TB.landmarkRx.exec(text)) !== null) { landmarks.push(m[0]); if (m[0] === '') TB.landmarkRx.lastIndex++; }
    text = text.replace(TB.landmarkRx, ' ');
    const rawT = splitWs(clean(text)), toks = [];
    rawT.forEach((t, k) => {
      const edge = k === rawT.length - 1 || k === 0;
      if (t.length === 2 && (edge || ['tx', 'hr', 'dl', 'ka', 'mh'].includes(t)))
        t = TB.us_states[t] || TB.in_states[t] || ab(TB.addr_abbrev, t);
      else t = ab(TB.addr_abbrev, t);
      toks.push(t);
    });
    const all = splitWs(toks.join(' '));
    const nums = all.join(' ').match(/\d+/g) || [];
    return { anorm: all.join(' '), awords: all.filter(t => !/^\d+$/.test(t)),
      pc: nums.filter(x => x.length === 5 || x.length === 6),
      nums: nums.filter(x => !(x.length === 5 || x.length === 6)).map(x => x.replace(/^0+/, '') || '0'),
      landmark: clean(landmarks.join(' ')) };
  }

  function prep(r) {
    const n = normName(r.business_name), a = normAddress(r.business_address);
    return { id: r.entity_id, country: r.country, src: (r.entity_id || '').slice(0, 2),
      name: r.business_name, addr: r.business_address, core: n.core, alt: n.alt, suffix: n.suffix,
      skel: n.skel, acr: n.acr, translit: n.translit, toks: splitWs(n.core), sk: splitWs(n.skel),
      anorm: a.anorm, awords: a.awords, pc: a.pc, nums: a.nums, landmark: a.landmark };
  }

  // ------------------------------------------------------------ similarity
  function jaroWinkler(s, t) {
    if (s === t) return s ? 1.0 : 0.0;
    const ls = s.length, lt = t.length;
    if (!ls || !lt) return 0.0;
    const rng = Math.max(Math.floor(Math.max(ls, lt) / 2) - 1, 0);
    const sm = new Array(ls).fill(false), tm = new Array(lt).fill(false);
    let m = 0;
    for (let i = 0; i < ls; i++) {
      const lo = Math.max(0, i - rng), hi = Math.min(i + rng + 1, lt);
      for (let j = lo; j < hi; j++) if (!tm[j] && s[i] === t[j]) { sm[i] = tm[j] = true; m++; break; }
    }
    if (!m) return 0.0;
    let k = 0, tr = 0;
    for (let i = 0; i < ls; i++) if (sm[i]) { while (!tm[k]) k++; if (s[i] !== t[k]) tr++; k++; }
    const jaro = (m / ls + m / lt + (m - tr / 2) / m) / 3;
    let p = 0;
    while (p < Math.min(4, ls, lt) && s[p] === t[p]) p++;
    return jaro + p * 0.1 * (1 - jaro);
  }
  function levRatio(s, t) {
    if (!s && !t) return 1.0;
    let prev = Array.from({ length: t.length + 1 }, (_, j) => j);
    for (let i = 1; i <= s.length; i++) {
      const cur = [i];
      for (let j = 1; j <= t.length; j++)
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (s[i - 1] !== t[j - 1] ? 1 : 0));
      prev = cur;
    }
    return 1 - prev[t.length] / Math.max(s.length, t.length);
  }
  function grams(x) {
    x = ' ' + x + ' '; const d = new Map();
    for (let i = 0; i < x.length - 2; i++) { const g = x.slice(i, i + 3); d.set(g, (d.get(g) || 0) + 1); }
    return d;
  }
  const sortedKeys = m => Array.from(m.keys()).sort();
  function trigramCos(s, t) {
    if (!s || !t) return 0.0;
    const a = grams(s), b = grams(t);
    let dot = 0.0;
    for (const g of sortedKeys(a)) if (b.has(g)) dot += a.get(g) * b.get(g);
    let na = 0, nb = 0;
    for (const g of sortedKeys(a)) na += a.get(g) * a.get(g);
    for (const g of sortedKeys(b)) nb += b.get(g) * b.get(g);
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  }
  function jaccard(A, B) {
    const a = new Set(A), b = new Set(B);
    if (!a.size || !b.size) return 0.0;
    let i = 0; for (const x of a) if (b.has(x)) i++;
    return i / (a.size + b.size - i);
  }
  function overlap(A, B) {
    const a = new Set(A), b = new Set(B);
    if (!a.size || !b.size) return 0.0;
    let i = 0; for (const x of a) if (b.has(x)) i++;
    return i / Math.min(a.size, b.size);
  }
  const idfOf = t => (Object.prototype.hasOwnProperty.call(IDF, t) ? IDF[t] : DFLT);
  function softMatch(A, B) {
    if (!A.length || !B.length) return [0.0, 0.0];
    let tot = 0.0, worst = 0.0;
    for (const [X, Y] of [[A, B], [B, A]]) {
      let acc = 0.0;
      for (const x of X) {
        let best = 0.0;
        for (const y of Y) { const v = jaroWinkler(x, y); if (v > best) best = v; }
        acc += best;
        if (best < 0.88) worst = Math.max(worst, idfOf(x));
      }
      tot += acc / X.length;
    }
    return [tot / 2, worst];
  }
  function idfJaccard(A, B) {
    const a = Array.from(new Set(A)).sort(), bs = new Set(B);
    if (!a.length || !bs.size) return 0.0;
    let inter = 0; for (const t of a) if (bs.has(t)) inter += idfOf(t);
    const uni = Array.from(new Set([...a, ...bs])).sort();
    let u = 0; for (const t of uni) u += idfOf(t);
    return u ? inter / u : 0.0;
  }
  const inter = (A, B) => { const b = new Set(B); return A.some(x => b.has(x)); };

  function pairFeatures(a, b) {
    const [me, worst] = softMatch(a.toks, b.toks);
    const ca = a.core.split(' ').join(''), cb = b.core.split(' ').join('');
    const altsA = [a.toks].concat(a.alt ? [splitWs(a.alt)] : []), altsB = [b.toks].concat(b.alt ? [splitWs(b.alt)] : []);
    let altBest = -Infinity;
    for (const x of altsA) for (const y of altsB) altBest = Math.max(altBest, overlap(x, y));
    const pcb = a.pc.length && b.pc.length ? 1.0 : 0.0, pce = pcb && inter(a.pc, b.pc) ? 1.0 : 0.0;
    const nb = a.nums.length && b.nums.length ? 1.0 : 0.0, ne = nb && inter(a.nums, b.nums) ? 1.0 : 0.0;
    const acr = (a.acr.length >= 2 && a.acr === cb) || (b.acr.length >= 2 && b.acr === ca) ? 1.0 : 0.0;
    return [jaroWinkler(a.core, b.core), levRatio(a.core, b.core), jaccard(a.toks, b.toks), overlap(a.toks, b.toks),
      trigramCos(a.core, b.core), jaccard(a.sk, b.sk), a.skel && a.skel === b.skel ? 1.0 : 0.0,
      a.toks.length && b.toks.length && a.toks[0] === b.toks[0] ? 1.0 : 0.0,
      levRatio(ca, cb), me, idfJaccard(a.toks, b.toks), worst,
      a.suffix && b.suffix ? 1.0 : 0.0, a.suffix && a.suffix === b.suffix ? 1.0 : 0.0,
      acr, altBest, jaccard(a.awords, b.awords), trigramCos(a.anorm, b.anorm),
      pcb, pce, pcb - pce, nb, ne, nb - ne, !a.anorm || !b.anorm ? 1.0 : 0.0,
      Math.abs(a.core.length - b.core.length), Math.abs(a.toks.length - b.toks.length)];
  }

  // ------------------------------------------------------------ blocking
  const STREET = /(?:^|\s)(\d+)\s+([a-z]{3,})/;
  function keys(r) {
    const out = [];
    for (const t of r.sk) if (t.length >= 2) out.push('t|' + t);
    for (const t of r.toks) if (t.length >= 4) out.push('q|' + t.slice(0, 4));
    if (r.sk.length >= 2) out.push('n|' + r.sk.slice(0, 2).sort().join('_'));
    const comp = r.core.split(' ').join('');
    if (comp.length >= 4) out.push('c|' + comp.slice(0, 6));
    if (r.pc.length && r.sk.length) out.push('p|' + r.pc[0] + '|' + r.sk[0]);
    const m = STREET.exec(r.anorm);
    if (m) out.push('a|' + m[1] + '|' + skeleton(m[2]));
    return Array.from(new Set(out));
  }
  function buildIndex(pool) {
    const inv = new Map();
    pool.forEach((r, i) => { for (const k of keys(r)) { if (!inv.has(k)) inv.set(k, []); inv.get(k).push(i); } });
    const n = pool.length, cap = Math.max(50, Math.floor(0.05 * n)), w = new Map();
    for (const [k, v] of inv) if (v.length <= cap) w.set(k, Math.log(1 + n / v.length));
    return { inv, w, n, cap };
  }
  function block(s, pool, index) {
    const score = new Map();
    for (const k of keys(s)) {
      if (!index.w.has(k)) continue;
      const w = index.w.get(k);
      for (const j of index.inv.get(k)) score.set(j, (score.get(j) || 0.0) + w);
    }
    if (!score.size) return { kept: [], keyed: 0 };
    const ranked = Array.from(score.entries()).sort((x, y) => (y[1] - x[1]) || (x[0] - y[0])).slice(0, B.k_retrieve);
    const top = ranked[0][1];
    const res = ranked.map(([j, kb]) => {
      const p = pool[j];
      return [j, kb, trigramCos(s.core, p.core) + 0.5 * trigramCos(s.anorm, p.anorm) + 0.2 * kb / top];
    });
    res.sort((x, y) => (y[2] - x[2]) || (x[0] - y[0]));
    return { kept: res.slice(0, B.k_max), keyed: score.size };
  }

  // ------------------------------------------------------------ context (stage 2)
  function context(pairs, p1, recs) {
    const byS = new Map(), byT = new Map();
    pairs.forEach(([s, t], i) => {
      if (!byS.has(s)) byS.set(s, []); byS.get(s).push(i);
      if (!byT.has(t)) byT.set(t, []); byT.get(t).push(i);
    });
    const n = pairs.length, rankS = new Array(n), rankT = new Array(n), smax = new Array(n), s2 = new Array(n),
      ssum = new Array(n), tbest = new Array(n), tn = new Array(n);
    const ord = idx => idx.slice().sort((a, b) => (p1[b] - p1[a]) || (a - b));
    for (const idx of byS.values()) {
      const o = ord(idx), vals = o.map(i => p1[i]);
      o.forEach((i, r) => { rankS[i] = r + 1; smax[i] = vals[0]; s2[i] = vals.length > 1 ? vals[1] : 0.0; });
      let tot = 0.0; for (const i of idx) tot += p1[i];
      for (const i of idx) ssum[i] = tot;
    }
    for (const idx of byT.values()) {
      const o = ord(idx);
      o.forEach((i, r) => { rankT[i] = r + 1; tn[i] = idx.length; const ot = o.filter(k => k !== i); tbest[i] = ot.length ? p1[ot[0]] : 0.0; });
    }
    return pairs.map(([s, t], i) => {
      const x = recs.get(t);
      let tri = 0.0;
      if (p1[i] >= 0.02) for (const k of byS.get(s)) {
        if (k === i || p1[k] < 0.2) continue;
        const y = recs.get(pairs[k][1]);
        const nn = overlap(x.toks, y.toks), aa = x.awords.length && y.awords.length ? overlap(x.awords, y.awords) : nn;
        const v = (0.6 * nn + 0.4 * aa) * p1[k];
        if (v > tri) tri = v;
      }
      return [p1[i], rankS[i], smax[i], smax[i] - p1[i], smax[i] - s2[i], ssum[i], rankT[i], tbest[i], p1[i] - tbest[i], tn[i], tri];
    });
  }

  // ------------------------------------------------------------ models
  function predict(trees, x) {
    let raw = 0.0;
    for (const nodes of trees) {
      let i = 0;
      while (nodes[i][0] !== -1) { const [f, thr, l, r] = nodes[i]; i = x[f] <= thr ? l : r; }
      raw += nodes[i][1];
    }
    return 1 / (1 + Math.exp(-raw));
  }
  function iso(p) {
    const xs = B.iso.x, ys = B.iso.y, n = xs.length;
    if (p <= xs[0]) return ys[0];
    if (p >= xs[n - 1]) return ys[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= p) lo = mid; else hi = mid; }
    return ys[lo] + (p - xs[lo]) * (ys[hi] - ys[lo]) / (xs[hi] - xs[lo]);
  }

  // ------------------------------------------------------------ decoding
  function poissonBinomial(ps) {
    let d = [1.0];
    for (const p of ps) {
      const nd = new Array(d.length + 1).fill(0.0);
      for (let t = 0; t < d.length; t++) { nd[t] += d[t] * (1 - p); nd[t + 1] += d[t] * p; }
      d = nd;
    }
    return d;
  }
  /** Expected F-beta for every k (0 = predict no match). Returns {best, scores}. */
  function expectedF(probs, beta = 0.5, missRate = 0.0, maxConsider = 25, minP = 1e-3) {
    const ps = probs.slice().sort((a, b) => b - a).filter(p => p >= minP);
    const b2 = beta * beta, extra = missRate > 0 ? [missRate] : [];
    let e0 = 1 - missRate; for (const p of ps) e0 *= (1 - p);
    const scores = [e0]; let best = 0, bestE = e0;
    for (let k = 1; k <= Math.min(ps.length, maxConsider); k++) {
      const tp = poissonBinomial(ps.slice(0, k)), fn = poissonBinomial(ps.slice(k).concat(extra));
      let e = 0.0;
      for (let t = 0; t <= k; t++) for (let f = 0; f < fn.length; f++) {
        const den = k + b2 * (t + f);
        if (den > 0) e += tp[t] * fn[f] * ((1 + b2) * t / den);
      }
      scores.push(e);
      if (e > bestE) { best = k; bestE = e; }
    }
    return { best, scores };
  }

  function fbeta(pred, truth, b2 = 0.25) {
    const P = new Set(pred), T = new Set(truth);
    if (!T.size) return P.size ? 0.0 : 1.0;
    if (!P.size) return 0.0;
    let tp = 0; for (const x of P) if (T.has(x)) tp++;
    if (!tp) return 0.0;
    const p = tp / P.size, r = tp / T.size;
    return (1 + b2) * p * r / (b2 * p + r);
  }

  // ------------------------------------------------------------ full run
  async function run(data, onStage) {
    const say = onStage || (() => {});
    const t0 = Date.now();
    const s1 = data.s1.map(prep), pool = data.pool.map(prep);
    const recs = new Map(); for (const r of s1.concat(pool)) recs.set(r.id, r);
    await say('normalize', { ms: Date.now() - t0, n: s1.length + pool.length });
    const byC = new Map(); for (const p of pool) { if (!byC.has(p.country)) byC.set(p.country, []); byC.get(p.country).push(p); }
    const idx = new Map(); for (const [c, v] of byC) idx.set(c, buildIndex(v));
    const pairs = [], blockInfo = new Map(); let allPairs = 0, keyed = 0;
    for (const s of s1) {
      const P = byC.get(s.country) || pool, I = idx.get(s.country) || buildIndex(pool);
      allPairs += P.length;
      const r = block(s, P, I); keyed += r.keyed;
      blockInfo.set(s.id, r.kept.map(([j, kb, bs]) => ({ t: P[j].id, kb, bs })));
      for (const [j] of r.kept) pairs.push([s.id, P[j].id]);
    }
    await say('block', { allPairs, keyed, kept: pairs.length });
    const X1 = pairs.map(([s, t]) => pairFeatures(recs.get(s), recs.get(t)));
    const p1 = X1.map(x => predict(B.m1, x));
    await say('stage1', { n: pairs.length });
    const X2 = context(pairs, p1, recs);
    const p3 = X2.map(x => iso(predict(B.m2, x)));
    await say('stage2', { n: pairs.length });
    const groups = new Map();
    pairs.forEach(([s, t], i) => { if (!groups.has(s)) groups.set(s, []); groups.get(s).push({ t, p: p3[i], p1: p1[i], i }); });
    const pred = {}, dec = {};
    for (const s of s1) {
      const g = (groups.get(s.id) || []).sort((a, b) => b.p - a.p);
      let k = 0, scores = [1.0];
      if (g.length && g[0].p >= 1e-3) { const r = expectedF(g.map(x => x.p)); k = r.best; scores = r.scores; }
      else if (g.length) { scores = [g.reduce((a, x) => a * (1 - x.p), 1.0)]; }
      pred[s.id] = g.slice(0, k).map(x => x.t);
      dec[s.id] = { cands: g, k, scores };
    }
    await say('decode', { ms: Date.now() - t0 });
    return { s1, pool, recs, pairs, X1, p1, X2, p3, pred, dec, blockInfo, allPairs, keyed, ms: Date.now() - t0 };
  }

  function evaluate(res, truth) {
    const ids = res.s1.map(r => r.id);
    let tp = 0, np = 0, nt = 0, sing = 0, singOk = 0, sumF = 0;
    const byC = {};
    for (const r of res.s1) {
      const pr = res.pred[r.id] || [], tr = truth[r.id] || [];
      const f = fbeta(pr, tr); sumF += f;
      (byC[r.country] = byC[r.country] || []).push(f);
      const T = new Set(tr); for (const x of pr) if (T.has(x)) tp++;
      np += pr.length; nt += tr.length;
      if (!tr.length) { sing++; if (!pr.length) singOk++; }
    }
    const tset = new Set(); for (const [s, ts] of Object.entries(truth)) for (const t of ts) tset.add(s + '>' + t);
    let hit = 0; for (const [s, t] of res.pairs) if (tset.has(s + '>' + t)) hit++;
    const country = {}; for (const [c, v] of Object.entries(byC)) country[c] = v.reduce((a, b) => a + b, 0) / v.length;
    return { f05: sumF / ids.length, precision: tp / Math.max(np, 1), recall: tp / Math.max(nt, 1),
      singletonAcc: sing ? singOk / sing : null, blockingRecall: hit / Math.max(tset.size, 1), country, n: ids.length };
  }

  /** Score one hand-typed pair with the stage-1 model only. */
  function scorePair(ra, rb) {
    const a = prep(ra), b = prep(rb), x = pairFeatures(a, b);
    return { a, b, x, p: predict(B.m1, x) };
  }

  const api = { init, prep, transliterate, normName, normAddress, skeleton, keys, pairFeatures, predict,
    iso, expectedF, fbeta, run, evaluate, scorePair, context, block, buildIndex };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Triveni = api;
})(typeof self !== 'undefined' ? self : this);
