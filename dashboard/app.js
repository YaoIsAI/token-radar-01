// Token Radar's night-scope dashboard. It reads window.RADAR (written beside dashboard/ by
// main.ts) and renders the ranked board, a filterable table, a top-3 spotlight, a per-protocol
// detail drawer and a live-token view. No network: every byte it shows is in data.js. Views by
// URL (?view=board|detail|filtered|validation, &slug=, &category=, &q=, &tab=live) or hash
// (#p=<slug>, #q=<text>, #category=<name>), so a screenshot can be taken of each state.
(function () {
  'use strict';

  var DATA = window.RADAR;
  if (!DATA) return;
  var WATCH = {};
  (DATA.watchlist || []).forEach(function (s) { WATCH[s] = true; });
  var app = document.getElementById('board');
  var state = { q: '', category: 'all', chain: 'all', watchlist: 'all', tab: 'pre-token', sort: 'implied', dir: -1, selected: null, view: 'board' };

  var usd = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
  function fmtUsd(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    if (Math.abs(n) >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (Math.abs(n) >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return '$' + Math.round(n / 1e3) + 'K';
    return usd.format(n);
  }
  function fmtPct(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    return (n > 0 ? '+' : '') + n.toFixed(2) + '%';
  }
  function fmtRatio(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    if (Math.abs(n) < 0.005) return '≈ 0×';
    return n.toFixed(2) + '×';
  }
  function fmtAge(days) {
    if (days === null || days === undefined) return 'unknown';
    if (days >= 365) return (days / 365).toFixed(1) + 'y';
    return Math.max(0, Math.round(days)) + 'd';
  }

// The radar-5 model in one function: the page recomputes a row from the snapshot figures, the
  // fitted coefficients data.js carries and the fixed constants, and checks the number board.json
  // carries. This mirrors lib/model.ts scoreCore exactly: same features, same order, same rounding
  // (a one-dollar difference is allowed: Math.log can differ by a last bit between engines). A
  // row's launch probability comes from its own cross-fit fold (row.model.fold).
  function impliedOf(row) {
    var fitted = DATA.fitted;
    var cal = DATA.calibration;
    var model = DATA.model;
    var age = (row.age_days === null || row.age_days === undefined) ? null : row.age_days;
    var missingAge = age === null;
    var missingGrowth = row.change_7d === null || row.change_7d === undefined;
    var ageScore = age === null ? 0 : 1 - Math.exp(-age / 365);
    var growthScore = missingGrowth ? 0 : Math.tanh(Math.min(model.growth_clamp_pct, Math.max(-model.growth_clamp_pct, row.change_7d)) / model.growth_scale);
    var chainScore = 1 - Math.exp(-Math.log1p((row.chains || []).length) / Math.log(9));
    var prior = cal.categories[row.category || 'Other'] || { launch_prior: cal.global_launch_rate, value_prior: cal.global_value_log_median };
    var logV = fitted.value.intercept
      + fitted.value.log_tvl * Math.log(Math.max(row.tvl_usd, 1))
      + fitted.value.value_prior * prior.value_prior
      + fitted.value.growth * growthScore
      + fitted.value.chain * chainScore
      + fitted.value.missing_age * (missingAge ? 1 : 0)
      - (missingGrowth ? model.missing_growth_penalty : 0);
    var vConditional = Math.exp(Math.max(-30, Math.min(30, logV)));
    var delay = (age !== null && age > model.delay_threshold_days)
      ? Math.min(model.delay_cap, model.delay_scale * Math.log(age / model.delay_threshold_days))
      : 0;
    var fold = (row.model && typeof row.model.fold === 'number') ? row.model.fold : 0;
    var launchFold = fitted.launch_folds[fold];
    var foldPrior = fitted.launch_fold_priors[fold];
    var launchPrior = (foldPrior.categories[row.category || 'Other'] !== undefined)
      ? foldPrior.categories[row.category || 'Other']
      : foldPrior.global_launch_rate;
    var logit = launchFold.intercept
      + launchFold.age * ageScore
      + launchFold.growth * growthScore
      + launchFold.chain * chainScore
      + launchFold.launch_prior * launchPrior
      + launchFold.missing_age * (missingAge ? 1 : 0)
      - (missingGrowth ? model.missing_growth_penalty : 0)
      - delay;
    var pLaunch = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, logit))));
    return Math.max(1, Math.round(pLaunch * vConditional));
  }
  // A recompute hook for verification: the page's own arithmetic, callable from the console.
  window.RADAR_TEST = { impliedOf: impliedOf };

  function el(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === 'class') node.className = attrs[k];
        else if (k === 'text') node.textContent = attrs[k];
        else if (k === 'html') node.innerHTML = attrs[k];
        else if (k.slice(0, 2) === 'on') node.addEventListener(k.slice(2), attrs[k]);
        else if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
      });
    }
    for (var i = 2; i < arguments.length; i++) {
      var child = arguments[i];
      if (child === null || child === undefined) continue;
      if (Array.isArray(child)) child.forEach(function (c) { if (c) node.appendChild(c); });
      else if (typeof child === 'string') node.appendChild(document.createTextNode(child));
      else node.appendChild(child);
    }
    return node;
  }

  // URL state: query params first, then the hash, so ?view=detail&slug=x and #p=x both work.
  function readParams() {
    var q = new URLSearchParams(window.location.search);
    var hash = new URLSearchParams(window.location.hash.replace(/^#/, ''));
    function get(name) { return q.get(name) || hash.get(name); }
    var view = get('view');
    if (view === 'detail' || view === 'filtered' || view === 'board' || view === 'validation') state.view = view;
    var slug = get('slug') || get('p');
    if (slug) { state.selected = slug; state.view = 'detail'; }
    var category = get('category');
    if (category) { state.category = category; if (!view) state.view = 'filtered'; }
    var chain = get('chain');
    if (chain) state.chain = chain;
    var text = get('q');
    if (text) { state.q = text; if (!view) state.view = 'filtered'; }
    var tab = get('tab');
    if (tab === 'live' || tab === 'pre-token') { state.tab = tab; state.sort = tab === 'live' ? 'mcap' : 'implied'; state.dir = -1; }
    var wl = get('wl');
    if (wl === 'watch' || wl === 'outside' || wl === 'all') state.watchlist = wl;
  }
  function writeHash() {
    var params = new URLSearchParams();
    if (state.q) params.set('q', state.q);
    if (state.category !== 'all') params.set('category', state.category);
    if (state.chain !== 'all') params.set('chain', state.chain);
    if (state.watchlist !== 'all') params.set('wl', state.watchlist);
    if (state.selected) params.set('p', state.selected);
    if (state.tab !== 'pre-token') params.set('tab', state.tab);
    var next = params.toString();
    if (next) window.history.replaceState(null, '', '#' + next);
    else window.history.replaceState(null, '', window.location.pathname + window.location.search);
  }

  function hayOf(r) {
    return [r.name, r.slug, r.category, (r.chains || []).join(' ')].join(' ').toLowerCase();
  }
  function matchFilters(r, q) {
    if (state.category !== 'all' && r.category !== state.category) return false;
    if (state.chain !== 'all' && !(r.chains || []).includes(state.chain)) return false;
    if (!q) return true;
    return hayOf(r).indexOf(q) >= 0;
  }
  function preRows() {
    var q = state.q.trim().toLowerCase();
    return DATA.rows.filter(function (r) {
      if (state.watchlist === 'watch' && !WATCH[r.slug]) return false;
      if (state.watchlist === 'outside' && WATCH[r.slug]) return false;
      return matchFilters(r, q);
    });
  }
  function liveRowsFiltered() {
    var q = state.q.trim().toLowerCase();
    return DATA.live.filter(function (r) { return matchFilters(r, q); });
  }
  function visibleRows() {
    return state.tab === 'live' ? liveRowsFiltered() : preRows();
  }
  function sortedRows(list) {
    var key = state.sort;
    return list.slice().sort(function (a, b) {
      var av, bv;
      if (key === 'rank') { av = a.rank; bv = b.rank; }
      else if (key === 'name') { av = a.name.toLowerCase(); bv = b.name.toLowerCase(); }
      else if (key === 'tvl') { av = a.tvl_usd; bv = b.tvl_usd; }
      else if (key === 'growth') { av = a.change_7d === null ? -Infinity : a.change_7d; bv = b.change_7d === null ? -Infinity : b.change_7d; }
      else if (key === 'age') { av = a.age_days === null ? Infinity : a.age_days; bv = b.age_days === null ? Infinity : b.age_days; }
      else if (key === 'mcap') { av = a.mcap_usd === null ? -Infinity : a.mcap_usd; bv = b.mcap_usd === null ? -Infinity : b.mcap_usd; }
      else { av = a.implied_value_usd; bv = b.implied_value_usd; }
      if (av < bv) return -state.dir;
      if (av > bv) return state.dir;
      return a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0;
    });
  }

  var SORT_LABELS = { rank: 'Rank', name: 'Protocol', tvl: 'TVL', growth: '7d', age: 'Age', implied: 'Implied value', mcap: 'Market cap' };
  function sortLabel() {
    return (SORT_LABELS[state.sort] || state.sort) + (state.dir < 0 ? ' ▼ high to low' : ' ▲ low to high');
  }

  function header() {
    return el('header', { class: 'top' },
      el('div', { class: 'brand' },
        el('span', { class: 'sweep', 'aria-hidden': 'true' }),
        el('span', { class: 'brand-name' }, el('b', { text: 'Token Radar' }), ' — which launch to farm next')
      ),
      el('div', { class: 'top-meta' },
        el('span', { text: 'Snapshot ' + DATA.as_of }),
        el('span', { text: 'DefiLlama protocol index' }),
        el('span', { text: DATA.model_version + ' model' })
      )
    );
  }

  function lede() {
    return el('section', { class: 'lede' },
      el('div', null,
        el('h1', { text: DATA.watchlist_size + ' watchlist protocols, ranked by implied launch value' }),
        el('p', { text: 'The watchlist is the floor: every protocol with at least $' + Math.round(DATA.floor_usd / 1e6) + 'M of TVL and no token on record. The early radar adds ' + DATA.early_size + ' more pre-token protocols outside the watchlist, so the board is the whole pre-token universe the snapshot can see — not a TVL table. Implied value is a model, not a promise.' })
      ),
      el('div', { class: 'facts' },
        el('div', { class: 'fact' }, el('span', { text: 'Watchlist slug check' }), el('b', { text: DATA.watchlist_covered + ' / ' + DATA.watchlist_size })),
        el('div', { class: 'fact' }, el('span', { text: 'Pre-token rows scored' }), el('b', { text: String(DATA.board_size) })),
        el('div', { class: 'fact' }, el('span', { text: 'Early rows outside the watchlist' }), el('b', { text: String(DATA.early_size) })),
        el('div', { class: 'fact' }, el('span', { text: 'Live-token protocols set aside' }), el('b', { text: String(DATA.live_size) })),
        el('div', { class: 'fact' }, el('span', { text: 'Rows missing age' }), el('b', { text: String(DATA.missing_age_count) })),
        el('div', { class: 'fact' }, el('span', { text: 'Rows missing 7d growth' }), el('b', { text: String(DATA.missing_growth_count) }))
      )
    );
  }

  function spotlight() {
    var top = DATA.rows.slice(0, 3);
    var cards = top.map(function (r) {
      return el('button', { class: 'spot-card', type: 'button', onclick: function () { openDetail(r.slug); } },
        el('span', { class: 'rank', text: 'Farm first · Rank ' + r.rank }),
        el('b', { text: r.name }),
        el('span', { class: 'val', text: fmtUsd(r.implied_value_usd) + ' implied' }),
        el('span', { class: 'sub', text: (r.category || 'Other') + ' · TVL ' + fmtUsd(r.tvl_usd) + ' · ' + fmtPct(r.change_7d) + ' in 7d' })
      );
    });
    return el('section', { class: 'spot', 'aria-label': 'Top three protocols to farm first' }, cards);
  }

  function controls() {
    var source = state.tab === 'live' ? DATA.live : DATA.rows;
    var categories = Array.from(new Set(source.map(function (r) { return r.category || 'Other'; }))).sort();
    var chains = Array.from(new Set(source.reduce(function (acc, r) { return acc.concat(r.chains || []); }, []))).sort();
    var search = el('input', { type: 'search', id: 'q', placeholder: 'Search name, category or chain', value: state.q, 'aria-label': 'Search the board' });
    search.addEventListener('input', function () { state.q = search.value; writeHash(); render(); });
    var cat = el('select', { id: 'category', 'aria-label': 'Filter by category' },
      el('option', { value: 'all', text: 'All categories' }),
      categories.map(function (c) { return el('option', { value: c, text: c, selected: c === state.category ? 'selected' : null }); })
    );
    cat.addEventListener('change', function () { state.category = cat.value; writeHash(); render(); });
    var chain = el('select', { id: 'chain', 'aria-label': 'Filter by chain' },
      el('option', { value: 'all', text: 'All chains' }),
      chains.map(function (c) { return el('option', { value: c, text: c, selected: c === state.chain ? 'selected' : null }); })
    );
    chain.addEventListener('change', function () { state.chain = chain.value; writeHash(); render(); });
    var pre = el('button', { type: 'button', 'aria-pressed': state.tab === 'pre-token' ? 'true' : 'false', text: 'Pre-token', onclick: function () { state.tab = 'pre-token'; state.sort = 'implied'; state.dir = -1; render(); } });
    var live = el('button', { type: 'button', 'aria-pressed': state.tab === 'live' ? 'true' : 'false', text: 'Live token', onclick: function () { state.tab = 'live'; state.sort = 'mcap'; state.dir = -1; render(); } });
    var fields = [
      el('div', { class: 'field' }, el('label', { for: 'q', text: 'Search' }), search),
      el('div', { class: 'field' }, el('label', { for: 'category', text: 'Category' }), cat),
      el('div', { class: 'field' }, el('label', { for: 'chain', text: 'Chain' }), chain)
    ];
    if (state.tab === 'pre-token') {
      var wl = el('select', { id: 'watchlist', 'aria-label': 'Filter by watchlist' },
        el('option', { value: 'all', text: 'All rows' }),
        el('option', { value: 'watch', text: 'Watchlist only', selected: state.watchlist === 'watch' ? 'selected' : null }),
        el('option', { value: 'outside', text: 'Outside the watchlist', selected: state.watchlist === 'outside' ? 'selected' : null })
      );
      wl.addEventListener('change', function () { state.watchlist = wl.value; writeHash(); render(); });
      fields.push(el('div', { class: 'field' }, el('label', { for: 'watchlist', text: 'Watchlist' }), wl));
    }
    fields.push(el('div', { class: 'field' }, el('span', { class: 'sub', text: 'Token status' }), el('div', { class: 'tabs' }, pre, live)));
    fields.push(el('span', { class: 'sort-note', text: 'Sorted by ' + sortLabel() }));
    fields.push(el('span', { class: 'count', text: state.tab === 'pre-token' ? preRows().length + ' rows shown' : liveRowsFiltered().length + ' live-token protocols' }));
    return el('div', { class: 'controls' }, fields);
  }

  function sortHeader(key, label) {
    return el('th', { scope: 'col' },
      el('button', { type: 'button', onclick: function () {
        if (state.sort === key) state.dir = -state.dir;
        else { state.sort = key; state.dir = key === 'rank' ? 1 : -1; }
        render();
      } },
        label,
        state.sort === key ? el('span', { 'aria-hidden': 'true', text: state.dir < 0 ? ' ▼' : ' ▲' }) : null
      )
    );
  }

  function boardTable() {
    var list = sortedRows(preRows());
    if (list.length === 0) {
      return el('div', { class: 'panel' }, el('div', { class: 'empty' },
        el('h2', { text: 'No protocol matches that filter' }),
        el('p', { text: 'Clear the search or pick another category to see the board again.' }),
        el('button', { class: 'button primary', type: 'button', text: 'Clear filters', onclick: function () { state.q = ''; state.category = 'all'; state.chain = 'all'; state.watchlist = 'all'; writeHash(); render(); } })
      ));
    }
    var max = 1;
    list.forEach(function (r) { if (r.implied_value_usd > max) max = r.implied_value_usd; });
    var tbody = el('tbody');
    list.forEach(function (r) {
      var growth = r.change_7d === null ? 'n/a' : fmtPct(r.change_7d);
      var growthClass = r.change_7d === null ? '' : r.change_7d >= 0 ? 'pos' : 'neg';
      var nameCell = el('div', { class: 'name' },
        el('b', { text: r.name }),
        el('span', { class: 'sub', text: (r.category || 'Other') + ' on ' + (r.chains && r.chains.length ? r.chains.slice(0, 2).join(', ') : 'an unnamed chain') })
      );
      if (WATCH[r.slug]) nameCell.appendChild(el('span', { class: 'tag', text: 'Watchlist' }));
      var tr = el('tr', {
        'data-selected': state.selected === r.slug ? 'true' : 'false',
        tabindex: '0',
        onclick: function () { openDetail(r.slug); },
        onkeydown: function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openDetail(r.slug); } }
      },
        el('td', { class: 'num', text: String(r.rank) }),
        el('td', null, nameCell),
        el('td', { class: 'num', text: fmtUsd(r.tvl_usd) }),
        el('td', { class: 'num ' + growthClass, text: growth }),
        el('td', { class: 'num', text: fmtAge(r.age_days) }),
        el('td', null, el('div', { class: 'implied' },
          el('div', { class: 'bar', 'aria-hidden': 'true' }, el('i', { style: 'width:' + Math.max(3, Math.round((r.implied_value_usd / max) * 100)) + '%' })),
          el('span', { class: 'num', text: fmtUsd(r.implied_value_usd) })
        )),
        el('td', null, el('span', { class: 'status', text: 'no token' }))
      );
      tbody.appendChild(tr);
    });
    return el('div', { class: 'panel' },
      el('div', { class: 'table-wrap' },
        el('table', null,
          el('caption', { class: 'sub', text: 'Pre-token protocols, strongest implied value first. Select a row for its model and farm plan.' }),
          el('thead', null, el('tr', null,
            sortHeader('rank', 'Rank'),
            sortHeader('name', 'Protocol'),
            sortHeader('tvl', 'TVL'),
            sortHeader('growth', '7d'),
            sortHeader('age', 'Age'),
            sortHeader('implied', 'Implied value'),
            el('th', { scope: 'col', text: 'Token' })
          )),
          tbody
        )
      )
    );
  }

  function liveTable() {
    var list = sortedRows(liveRowsFiltered());
    if (list.length === 0) {
      return el('div', { class: 'panel' }, el('div', { class: 'empty' },
        el('h2', { text: 'No live-token protocol matches that filter' }),
        el('p', { text: 'Clear the search or pick another category to see the live table again.' }),
        el('button', { class: 'button primary', type: 'button', text: 'Clear filters', onclick: function () { state.q = ''; state.category = 'all'; state.chain = 'all'; writeHash(); render(); } })
      ));
    }
    var tbody = el('tbody');
    list.forEach(function (r) {
      var tr = el('tr', {
        tabindex: '0',
        onclick: function () { openLiveDetail(r.slug); },
        onkeydown: function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openLiveDetail(r.slug); } }
      },
        el('td', null, el('span', { class: 'status live', text: 'already trades' })),
        el('td', null, el('div', { class: 'name' }, el('b', { text: r.name }), el('span', { class: 'sub', text: (r.category || 'Other') + ' on ' + (r.chains && r.chains.length ? r.chains.join(', ') : 'an unnamed chain') }))),
        el('td', { class: 'num', text: fmtUsd(r.tvl_usd) }),
        el('td', { class: 'num ' + (r.change_7d !== null && r.change_7d >= 0 ? 'pos' : 'neg'), text: fmtPct(r.change_7d) }),
        el('td', { class: 'num', text: fmtUsd(r.mcap_usd) }),
        el('td', { class: 'num', text: fmtRatio(r.mcap_usd === null || r.tvl_usd === 0 ? null : r.mcap_usd / r.tvl_usd) })
      );
      tbody.appendChild(tr);
    });
    return el('div', { class: 'panel' },
      el('div', { class: 'table-wrap' },
        el('table', null,
          el('caption', { class: 'sub', text: 'Every protocol with a market cap on record, outside CEX: ' + DATA.live.length + ' rows. Market cap is a fact, not a model. Protocols that record a token symbol but no market cap appear in neither view.' }),
          el('thead', null, el('tr', null,
            el('th', { scope: 'col', text: 'Status' }),
            sortHeader('name', 'Protocol'),
            sortHeader('tvl', 'TVL'),
            sortHeader('growth', '7d'),
            sortHeader('mcap', 'Market cap'),
            el('th', { scope: 'col', text: 'Mcap / TVL' })
          )),
          tbody
        )
      )
    );
  }

  function fmtNet(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    return (n < 0 ? '−$' : '$') + Math.abs(n).toFixed(2);
  }

  function modelPanel() {
    var f = DATA.fitted;
    var cal = DATA.calibration;
    var v = DATA.validation;
    return el('section', { class: 'panel model' },
      el('h2', { text: 'How implied value is computed — model ' + DATA.model_version }),
      el('p', { class: 'sub', text: 'Every row is scored by one function of the snapshot figures. Both halves are fit by regression on this snapshot’s own ' + v.live_fit_n + ' live-token protocols, at runtime — no coefficient is hand-set except the two the data cannot identify (see the research note). No protocol name is special-cased, and no row is hand-picked.' }),
      el('div', { class: 'formula' },
        'implied = round(P_launch x V_conditional)' +
        '\nlog V_conditional = ' + f.value.intercept.toFixed(2) + ' + ' + f.value.log_tvl.toFixed(2) + ' x log(TVL) + ' + f.value.value_prior.toFixed(2) + ' x category value prior + ' + f.value.growth.toFixed(2) + ' x growth + ' + f.value.chain.toFixed(2) + ' x chains + ' + f.value.missing_age.toFixed(2) + ' if age unknown − 0.05 if growth unreported' +
        '\nP_launch = sigmoid(' + f.launch.intercept.toFixed(2) + ' + ' + f.launch.age.toFixed(2) + ' x age + ' + f.launch.growth.toFixed(2) + ' x growth + ' + f.launch.chain.toFixed(2) + ' x chains + ' + f.launch.launch_prior.toFixed(2) + ' x category launch prior + ' + f.launch.missing_age.toFixed(2) + ' if age unknown − 0.05 if growth unreported − delay past 3y)'
      ),
      el('p', { class: 'sub', text: 'Both halves are ridge regressions fit at runtime (value ridge ' + v.ridge.value_lambda + ', launch ridge ' + v.ridge.launch_lambda + ', each chosen by five-fold cross-validation). Age is not a value feature any more: it broke the time-split ranking in the last version. A row’s P_launch comes from the fold model trained without that row (5-fold cross-fit, fold priors of its own); the launch line above is the full fit, for reference, and the five fold fits are in validation.json.' }),
      el('p', { class: 'sub', text: 'Category priors are shrunk toward the global rates (' + (cal.global_launch_rate * 100).toFixed(1) + '% launch rate, median log(market cap / TVL) ' + cal.global_value_log_median.toFixed(2) + '). Time-split holdout: Spearman ' + v.time_split.value_spearman.toFixed(2) + ', top-10 hit ' + v.time_split.value_top10_hit + '/' + v.time_split.value_top10_n + ', median log error ' + v.time_split.value_median_abs_log_error.toFixed(2) + ', launch AUC ' + v.time_split.launch_auc.toFixed(2) + '. Expanding-window mean Spearman ' + v.expanding_window.mean_spearman.toFixed(2) + '; out-of-fold value Spearman ' + v.oof.value_spearman.toFixed(2) + ', launch AUC ' + v.oof.launch_auc.toFixed(2) + '. Full backtest below, and the research note for the method, the assumptions and the limits.' })
    );
  }

  function coefTable(title, rows) {
    var tbody = el('tbody');
    rows.forEach(function (r) {
      tbody.appendChild(el('tr', null,
        el('td', { text: r.name }),
        el('td', { class: 'num', text: r.fitted.toFixed(2) }),
        el('td', { class: 'num', text: r.low.toFixed(2) + ' – ' + r.high.toFixed(2) }),
        el('td', { class: 'num', text: r.median.toFixed(2) })
      ));
    });
    return el('div', { class: 'panel model' },
      el('h3', { text: title }),
      el('div', { class: 'table-wrap' },
        el('table', null,
          el('thead', null, el('tr', null,
            el('th', { scope: 'col', text: 'Coefficient' }),
            el('th', { scope: 'col', text: 'Fitted' }),
            el('th', { scope: 'col', text: 'Bootstrap range (B = ' + DATA.validation.bootstrap_b + ')' }),
            el('th', { scope: 'col', text: 'Median' })
          )),
          tbody
        )
      )
    );
  }

  function validationPanel() {
    var v = DATA.validation;
    var t = v.time_split, r = v.random_split;
    function splitRows(s, label) {
      return [
        [label + ' — train / test', s.train_n + ' / ' + s.test_n + ' live protocols (cutoff ' + s.cutoff_age_days + ' days)'],
        [label + ' — value Spearman', s.value_spearman.toFixed(2)],
        [label + ' — value top-10 hit rate', s.value_top10_hit + ' / ' + s.value_top10_n],
        [label + ' — value median |log error|', s.value_median_abs_log_error.toFixed(2) + ' (about x' + Math.exp(s.value_median_abs_log_error).toFixed(1) + ')'],
        [label + ' — launch AUC', s.launch_auc.toFixed(2) + ' (' + s.launch_test_n + ' protocols, ' + s.launch_test_live_n + ' live)']
      ];
    }
    var rows = splitRows(t, 'Time split (oldest 70% fit, newest 30% held out)').concat(splitRows(r, 'Random split (70/30, fixed seed)'));
    rows.unshift(['In-sample value fit (' + v.live_fit_n + ' live protocols)', 'RMSE (log) ' + v.insample_value_rmse_log.toFixed(2) + ', median |log error| ' + v.insample_value_median_abs_log_error.toFixed(2)]);
    rows.unshift(['Ridge, chosen by 5-fold cross-validation (grid 0.03–30, smallest λ within 0.01 of the best wins)', 'value λ ' + v.ridge.value_lambda + ' (CV Spearman ' + v.ridge.value_cv_spearman.toFixed(2) + '), launch λ ' + v.ridge.launch_lambda + ' (CV AUC ' + v.ridge.launch_cv_auc.toFixed(2) + ')']);
    rows.push(['Expanding window — mean over 4 eras (each era predicted by a model fit only on older eras)', 'value Spearman ' + v.expanding_window.mean_spearman.toFixed(2) + ' (folds ' + v.expanding_window.folds.map(function (f) { return f.value_spearman.toFixed(2); }).join(', ') + '), mean median |log error| ' + v.expanding_window.mean_median_abs_log_error.toFixed(2)]);
    rows.push(['Out-of-fold value (' + v.live_fit_n + ' live protocols, 5 folds, priors recalibrated per fold)', 'Spearman ' + v.oof.value_spearman.toFixed(2) + ', median |log error| ' + v.oof.value_median_abs_log_error.toFixed(2) + ', top-10 hit ' + v.oof.value_top10_hit + ' / ' + v.oof.value_top10_n]);
    rows.push(['Out-of-fold launch (' + v.oof.launch_n + ' protocols, ' + v.oof.launch_live_n + ' live, 5 folds, priors recalibrated per fold)', 'AUC ' + v.oof.launch_auc.toFixed(2)]);
    rows.push(['Holdout coverage', v.known_age_live_n + ' live protocols with a recorded listing date; ' + v.missing_age_live_n + ' live protocols have none and cannot be placed in time, so they are counted here, not hidden.']);
    var tbody = el('tbody');
    rows.forEach(function (row) {
      tbody.appendChild(el('tr', null, el('td', { text: row[0] }), el('td', { class: 'num', text: row[1] })));
    });
    var panel = el('section', { class: 'panel model' },
      el('h2', { text: 'Backtest — does the fitted model predict what it has not seen?' }),
      el('p', { class: 'sub', text: 'The model is fit on this snapshot’s live-token protocols, then tested on held-out ones it never saw. The time split is the hard test: it fits the oldest 70% and predicts the newest 30%, so it must extrapolate to younger protocols. The random split is the easier test: same distribution, new rows. The expanding window repeats the time test four times, era by era. The out-of-fold rows predict every row from a model trained without it. Read the time split first.' }),
      el('div', { class: 'table-wrap' },
        el('table', { class: 'checks' },
          el('thead', null, el('tr', null, el('th', { scope: 'col', text: 'Check' }), el('th', { scope: 'col', text: 'Result' }))),
          tbody
        )
      ),
      el('p', { class: 'sub', text: 'What the numbers say: the value model orders held-out market caps moderately (Spearman ' + t.value_spearman.toFixed(2) + ' on the time split, ' + r.value_spearman.toFixed(2) + ' on the random split, ' + v.oof.value_spearman.toFixed(2) + ' out-of-fold) and its dollar error is wide — a median log error of ' + t.value_median_abs_log_error.toFixed(2) + ' is a factor of about x' + Math.exp(t.value_median_abs_log_error).toFixed(1) + ' on the time split. The youngest era is still unrankable: the expanding window’s last fold, the newest 19 protocols, scores near zero for every variant tried. Treat implied values as an ordering, not a price. The bootstrap ranges below show which coefficients the data actually pins down: log(TVL) and the category value prior are stable; growth and chain are not.' })
    );
    var wrap = el('div', null, panel, coefTable('Value coefficients — log(market cap)', v.value_bootstrap), coefTable('Launch coefficients — logit of launched-or-not', v.launch_bootstrap));
    return wrap;
  }

  function researchSection() {
    return el('section', { class: 'panel model' },
      el('h2', { text: 'Research note' }),
      el('p', { class: 'sub', text: 'The full note — the model, how the coefficients were fit, the holdout results, the farm cost model and its assumptions, the limits, and what the numbers are not — ships with this dashboard as RESEARCH.md, and the validation numbers as validation.json.' }),
      el('p', null, el('a', { href: 'RESEARCH.md', text: 'Read the research note (RESEARCH.md)' }), el('span', { class: 'sub', text: ' · ' }), el('a', { href: 'validation.json', text: 'Validation numbers (validation.json)' }))
    );
  }

  var dialog = el('dialog', { class: 'detail', id: 'detail', 'aria-labelledby': 'detail-title' });
  dialog.addEventListener('close', function () { state.selected = null; writeHash(); });

  function qualityText(m) {
    if (m.data_quality === 'complete') return 'complete — age and 7d growth both on record.';
    if (m.data_quality === 'partial') return 'partial — ' + (m.missing_age ? 'age is not reported' : '7d growth is not reported') + '.';
    return 'thin — age and 7d growth are both missing.';
  }

  function openDetail(slug) {
    var row = DATA.rows.filter(function (r) { return r.slug === slug; })[0];
    if (!row) { openLiveDetail(slug); return; }
    state.selected = slug;
    writeHash();
    dialog.textContent = '';
    var m = row.model;
    var plan = row.farm_plan || { capital_usd: 0, setup_cost_usd: 0, weekly_cost_usd: 0, hold_days: 0, weeks: 0, net_cost_usd: 0, sunk_cost_usd: 0, expected_gross_usd: 0, expected_net_usd: 0, exit_steps: [], risk_flags: [] };
    var steps = row.farm_steps || [];
    var recomputed = impliedOf(row);
    dialog.appendChild(el('div', { class: 'detail-head' },
      el('div', null,
        el('span', { class: 'brand-name', text: 'Rank ' + row.rank + ' of ' + DATA.board_size + (WATCH[row.slug] ? ' · Watchlist' : '') }),
        el('h2', { id: 'detail-title', text: row.name }),
        el('p', { class: 'sub', text: (row.category || 'Other') + ' on ' + (row.chains && row.chains.length ? row.chains.join(', ') : 'an unnamed chain') })
      ),
      el('button', { class: 'close', type: 'button', 'aria-label': 'Close details', text: 'Close', onclick: function () { dialog.close(); } })
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Implied value' }),
      el('div', { class: 'equation' },
        el('div', null, el('span', { text: 'TVL' }), el('b', { class: 'num', text: fmtUsd(row.tvl_usd) })),
        el('div', null, el('span', { text: 'Seven-day growth' }), el('b', { class: 'num', text: fmtPct(row.change_7d) })),
        el('div', null, el('span', { text: 'Age with no token' }), el('b', { class: 'num', text: fmtAge(row.age_days) })),
        el('div', null, el('span', { text: 'Chains deployed' }), el('b', { class: 'num', text: String((row.chains || []).length) })),
        el('div', null, el('span', { text: 'Launch probability' }), el('b', { class: 'num', text: (m.p_launch * 100).toFixed(1) + '%' })),
        el('div', null, el('span', { text: 'Conditional value' }), el('b', { class: 'num', text: fmtUsd(m.v_conditional) })),
        el('div', null, el('span', { text: 'Confidence band' }), el('b', { class: 'num', text: fmtUsd(m.implied_low) + ' – ' + fmtUsd(m.implied_high) })),
        el('div', null, el('span', { text: 'Implied value' }), el('b', { class: 'num', text: fmtUsd(row.implied_value_usd) }))
      ),
      el('p', { class: 'sub', text: 'Band: TVL ±30% and 7d growth ±20% through the same function. Model ' + DATA.model_version + ': the same function scores every protocol. The page recomputes this row as ' + fmtUsd(recomputed) + (Math.abs(recomputed - row.implied_value_usd) <= 1 ? ', the same number board.json carries (to the dollar).' : ', which differs from the board: the data would be stale.') + ' This is an estimate of what a launch could be worth, not a market-cap promise.' })
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'What drives the score' }),
      el('ul', { class: 'plain-list' }, (m.drivers || []).map(function (d) { return el('li', { text: d }); }))
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'What could be wrong' }),
      (m.risks && m.risks.length)
        ? el('ul', { class: 'plain-list' }, m.risks.map(function (d) { return el('li', { text: d }); }))
        : el('p', { class: 'sub', text: 'No material risk flagged for this row.' })
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Data quality' }),
      el('p', { class: 'sub', text: 'Data quality: ' + qualityText(m) })
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Farm plan — money, netted to the dollar' }),
      el('div', { class: 'equation' },
        el('div', null, el('span', { text: 'Net cost to run' }), el('b', { class: 'num', text: fmtNet(plan.net_cost_usd) })),
        el('div', null, el('span', { text: 'Capital (returned on exit)' }), el('b', { class: 'num', text: '$' + plan.capital_usd })),
        el('div', null, el('span', { text: 'Spent: setup + weekly gas' }), el('b', { class: 'num', text: fmtNet(plan.sunk_cost_usd) + ' over ' + plan.weeks + ' weeks' })),
        el('div', null, el('span', { text: 'Expected gross (pro-rata share)' }), el('b', { class: 'num', text: fmtNet(plan.expected_gross_usd) })),
        el('div', null, el('span', { text: 'Expected net' }), el('b', { class: 'num', text: fmtNet(plan.expected_net_usd) })),
        el('div', null, el('span', { text: 'Minimum hold' }), el('b', { class: 'num', text: plan.hold_days + ' days' }))
      ),
      el('p', { class: 'sub', text: 'Expected gross = launch probability x conditional value x (capital / TVL): it assumes any future token is distributed pro-rata to TVL share. The snapshot carries no airdrop terms, so this is a scenario, not a promise — a $50–$250 position in a $50M+ protocol is a dust share, and the expected net is usually negative. Full assumption in the research note.' }),
      el('h3', { text: 'Exit' }),
      el('ol', { class: 'steps' }, (plan.exit_steps || []).map(function (s) { return el('li', null, el('b', { text: s })); })),
      el('h3', { text: 'Risk flags' }),
      (plan.risk_flags && plan.risk_flags.length)
        ? el('ul', { class: 'plain-list' }, plan.risk_flags.map(function (d) { return el('li', { text: d }); }))
        : el('p', { class: 'sub', text: 'No risk flag for this plan.' })
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Farm plan — ' + steps.length + ' steps' }),
      el('ol', { class: 'steps' }, steps.map(function (s) {
        var at = s.indexOf(' Cost: ');
        var action = at >= 0 ? s.slice(0, at) : s;
        var tail = at >= 0 ? s.slice(at + 1) : '';
        return el('li', null, el('b', { text: action }), el('span', { class: 'cost', text: tail }));
      }))
    ));
    if (typeof dialog.showModal === 'function') {
      if (!dialog.open) dialog.showModal();
    } else {
      dialog.setAttribute('open', '');
    }
  }

  function openLiveDetail(slug) {
    var row = DATA.live.filter(function (r) { return r.slug === slug; })[0];
    if (!row) return;
    state.selected = slug;
    writeHash();
    dialog.textContent = '';
    var ratio = row.mcap_usd === null || row.tvl_usd === 0 ? null : row.mcap_usd / row.tvl_usd;
    dialog.appendChild(el('div', { class: 'detail-head' },
      el('div', null,
        el('span', { class: 'brand-name', text: 'Live token · not ranked' }),
        el('h2', { id: 'detail-title', text: row.name }),
        el('p', { class: 'sub', text: (row.category || 'Other') + ' on ' + (row.chains && row.chains.length ? row.chains.join(', ') : 'an unnamed chain') })
      ),
      el('button', { class: 'close', type: 'button', 'aria-label': 'Close details', text: 'Close', onclick: function () { dialog.close(); } })
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Market value' }),
      el('div', { class: 'equation' },
        el('div', null, el('span', { text: 'Market cap' }), el('b', { class: 'num', text: fmtUsd(row.mcap_usd) })),
        el('div', null, el('span', { text: 'TVL' }), el('b', { class: 'num', text: fmtUsd(row.tvl_usd) })),
        el('div', null, el('span', { text: 'Market cap / TVL' }), el('b', { class: 'num', text: fmtRatio(ratio) })),
        el('div', null, el('span', { text: 'Seven-day growth' }), el('b', { class: 'num', text: fmtPct(row.change_7d) })),
        el('div', null, el('span', { text: 'Age' }), el('b', { class: 'num', text: fmtAge(row.age_days) })),
        el('div', null, el('span', { text: 'Chains deployed' }), el('b', { class: 'num', text: String((row.chains || []).length) }))
      ),
      el('p', { class: 'sub', text: 'This token already trades, so its market cap is a fact, not a model. It is set aside from the pre-token ranking and carries no farm plan.' })
    ));
    if (typeof dialog.showModal === 'function') {
      if (!dialog.open) dialog.showModal();
    } else {
      dialog.setAttribute('open', '');
    }
  }

  function render() {
    app.textContent = '';
    var shell = el('div', { class: 'shell' });
    shell.appendChild(header());
    shell.appendChild(lede());
    if (state.view === 'validation') {
      shell.appendChild(validationPanel());
      shell.appendChild(researchSection());
      shell.appendChild(el('p', { class: 'footnote', text: 'Implied values are this radar’s own model, not a promise and not a price. A protocol records no token in the snapshot; that is DefiLlama’s record, not proof it has none. Source: ' + DATA.source + '.' }));
      app.appendChild(shell);
      app.setAttribute('aria-busy', 'false');
      return;
    }
    if (state.view !== 'detail') shell.appendChild(spotlight());
    shell.appendChild(controls());
    shell.appendChild(state.tab === 'pre-token' ? boardTable() : liveTable());
    if (state.tab === 'pre-token') { shell.appendChild(modelPanel()); shell.appendChild(validationPanel()); shell.appendChild(researchSection()); }
    shell.appendChild(el('p', { class: 'footnote', text: 'Implied values are this radar’s own model, not a promise and not a price. A protocol records no token in the snapshot; that is DefiLlama’s record, not proof it has none. Source: ' + DATA.source + '.' }));
    app.appendChild(shell);
    app.setAttribute('aria-busy', 'false');
    if (!dialog.isConnected) app.appendChild(dialog);
    if (state.selected) openDetail(state.selected);
  }

  readParams();
  render();
  window.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && dialog.open) dialog.close();
  });
})();
