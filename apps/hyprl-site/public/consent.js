/* Optional scripts are registered here only after a privacy/CSP review. Disabled by default. */
(function (root, factory) {
  'use strict';
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else {
    var api = factory();
    root.HyprLConsent = api;
    document.addEventListener('DOMContentLoaded', function () {
      // Do not access localStorage, mount a banner or load any optional resource when disabled.
      if (!api.ENABLED) return;
      api.mount(document, root.localStorage, []);
    });
  }
})(typeof window === 'undefined' ? globalThis : window, function () {
  'use strict';
  var ENABLED = false;
  var KEY = 'hyprl_optional_consent_v1';
  var PURPOSES = ['functional', 'analytics'];

  function afterMonths(at, count) {
    var d = new Date(at);
    var day = d.getUTCDate();
    d.setUTCDate(1);
    d.setUTCMonth(d.getUTCMonth() + count);
    var last = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 0)).getUTCDate();
    d.setUTCDate(Math.min(day, last));
    return d.toISOString();
  }

  function create(options) {
    options = options || {};
    var enabled = options.enabled === true;
    var storage = options.storage;
    var now = options.now || Date.now;
    var choice = null;
    var loaded = new Set();
    var scripts = options.scripts || [];
    function valid(c) {
      var decided = c && Date.parse(c.decided_at);
      return c && c.version === 1 && Number.isFinite(decided) && decided <= now() &&
        c.expires_at === afterMonths(decided, 6) && now() < Date.parse(c.expires_at) &&
        now() < Date.parse(afterMonths(decided, 13)) &&
        PURPOSES.every(function (p) { return typeof c.purposes?.[p] === 'boolean'; });
    }
    function read() {
      if (!enabled) return null;
      try {
        var c = JSON.parse(storage.getItem(KEY));
        if (valid(c)) return c;
        storage.removeItem(KEY);
      } catch (_) { /* Unavailable/corrupt storage fails closed. */ }
      return null;
    }
    choice = read();
    function current() {
      if (!enabled) return null;
      if (!valid(choice)) {
        choice = null;
        try { storage.removeItem(KEY); } catch (_) { /* No persistence required. */ }
      }
      return choice;
    }
    function loadAllowed() {
      var c = current();
      scripts.forEach(function (s) {
        if (c && PURPOSES.includes(s.purpose) && c.purposes[s.purpose] && !loaded.has(s)) {
          loaded.add(s);
          s.load();
        }
      });
    }
    function choose(purposes) {
      if (!enabled) return null;
      var previous = current();
      choice = {
        version: 1, decided_at: new Date(now()).toISOString(), expires_at: afterMonths(now(), 6),
        purposes: Object.fromEntries(PURPOSES.map(function (p) { return [p, purposes[p] === true]; }))
      };
      try { storage.setItem(KEY, JSON.stringify(choice)); } catch (_) { /* Session-only consent. */ }
      var revoked = previous && PURPOSES.some(function (p) { return previous.purposes[p] && !choice.purposes[p]; });
      if (revoked) {
        scripts.forEach(function (s) { if (loaded.has(s)) s.unload?.(); });
        options.onRevoke?.(); // Browser reload stops previously executed third-party code.
      } else loadAllowed();
      return choice;
    }
    return { current: current, choose: choose, loadAllowed: loadAllowed };
  }

  function mount(doc, storage, scripts) {
    if (!ENABLED) return null;
    var panel = doc.createElement('section');
    panel.className = 'consent-panel';
    panel.setAttribute('aria-labelledby', 'optional-consent-title');
    var title = doc.createElement('h2');
    title.id = 'optional-consent-title';
    title.textContent = 'Vos choix pour les services facultatifs';
    var text = doc.createElement('p');
    text.textContent = 'Les services facultatifs restent bloqués avant votre accord. Votre choix vaut six mois, puis sera redemandé (avant treize mois). Vous pouvez le modifier à tout moment.';
    var fields = doc.createElement('fieldset');
    var legend = doc.createElement('legend');
    legend.textContent = 'Finalités facultatives';
    fields.appendChild(legend);
    var inputs = {};
    PURPOSES.forEach(function (purpose) {
      var label = doc.createElement('label');
      label.className = 'check';
      var input = doc.createElement('input');
      input.type = 'checkbox';
      input.checked = false;
      inputs[purpose] = input;
      label.append(input, doc.createTextNode(purpose === 'functional' ? 'Contenus tiers facultatifs' : 'Mesure d’audience facultative'));
      fields.appendChild(label);
    });
    var actions = doc.createElement('div');
    actions.className = 'actions';
    var reopen = doc.createElement('button');
    reopen.type = 'button';
    reopen.className = 'btn btn-quiet';
    reopen.textContent = 'Modifier mes choix de cookies';
    var engine = create({ enabled: true, storage: storage, scripts: scripts, onRevoke: function () { location.reload(); } });
    function finish(purposes) {
      engine.choose(purposes);
      panel.hidden = true;
      reopen.hidden = false;
      reopen.focus();
    }
    [['Tout refuser', function () { finish({}); }],
      ['Tout accepter', function () { finish({ functional: true, analytics: true }); }],
      ['Enregistrer mes choix', function () {
        finish(Object.fromEntries(PURPOSES.map(function (p) { return [p, inputs[p].checked]; })));
      }]].forEach(function (action) {
      var button = doc.createElement('button');
      button.type = 'button';
      button.className = 'btn btn-secondary';
      button.textContent = action[0];
      button.addEventListener('click', action[1]);
      actions.appendChild(button);
    });
    panel.append(title, text, fields, actions);
    reopen.addEventListener('click', function () {
      var c = engine.current();
      PURPOSES.forEach(function (p) { inputs[p].checked = c?.purposes[p] === true; });
      panel.hidden = false;
      actions.querySelector('button').focus();
    });
    panel.hidden = !!engine.current();
    reopen.hidden = !panel.hidden;
    doc.body.append(panel, reopen);
    engine.loadAllowed();
    // Long-lived tabs also lose consent at expiry; reload removes executed optional code.
    var hadChoice = !!engine.current();
    setInterval(function () {
      var hasChoice = !!engine.current();
      if (hadChoice && !hasChoice) location.reload();
      hadChoice = hasChoice;
    }, 60000);
    window.addEventListener('storage', function (event) { if (event.key === KEY) location.reload(); });
    return engine;
  }
  return { ENABLED: ENABLED, KEY: KEY, create: create, mount: mount, afterMonths: afterMonths };
});
