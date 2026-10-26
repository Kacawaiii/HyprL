// Espace d'administration HyprL. Les données sont toujours insérées avec textContent (jamais innerHTML).
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var CHANNELS = { discord: 'Discord', slack: 'Slack', telegram: 'Telegram', teams: 'Microsoft Teams', email: 'E-mail', webhook: 'Webhook', api: 'API' };
  var fmt = new Intl.DateTimeFormat('fr-FR', {
    timeZone: 'Europe/Paris', year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit'
  });

  function api(method, url, body) {
    var opts = { method: method, credentials: 'same-origin', headers: { 'Accept': 'application/json' } };
    if (body !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    return fetch(url, opts).then(function (res) {
      return res.text().then(function (t) {
        var data = null;
        try { data = t ? JSON.parse(t) : null; } catch (e) { data = null; }
        return { status: res.status, data: data, headers: res.headers };
      });
    });
  }

  function show(el, text, kind) {
    el.textContent = text;
    el.className = 'status-msg' + (kind ? ' ' + kind : '');
    el.hidden = !text;
  }

  function showLogin(message) {
    $('admin-panel').hidden = true;
    $('admin-login').hidden = false;
    show($('login-msg'), message || '', 'error');
    $('password').value = '';
    $('password').focus();
  }

  function td(text, cls) {
    var c = document.createElement('td');
    if (cls) c.className = cls;
    c.textContent = text;
    return c;
  }

  function render(list) {
    var tbody = $('rows');
    tbody.replaceChildren();
    $('count').textContent = '(' + list.length + ')';
    if (!list.length) {
      var tr = document.createElement('tr');
      var c = td('Aucune inscription.', 'empty');
      c.colSpan = 6;
      tr.appendChild(c);
      tbody.appendChild(tr);
      return;
    }
    list.forEach(function (s) {
      var tr = document.createElement('tr');
      tr.appendChild(td(fmt.format(new Date(s.created_at)), 'mono nowrap'));
      tr.appendChild(td(s.email));
      tr.appendChild(td((s.channels || []).map(function (k) { return CHANNELS[k] || k; }).join(', ') || '—'));
      tr.appendChild(td(s.actions || '—', 'mono'));
      var consent = td('', 'consent-cell');
      var when = document.createElement('div');
      when.className = 'mono';
      when.textContent = 'Coché le ' + fmt.format(new Date(s.consent.checked_at)) + ' · ' + s.consent.version;
      var text = document.createElement('div');
      text.textContent = '« ' + s.consent.text + ' »';
      consent.append(when, text);
      tr.appendChild(consent);
      var actions = td('', 'nowrap');
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn btn-danger';
      btn.textContent = 'Supprimer';
      btn.setAttribute('aria-label', 'Supprimer ' + s.email);
      btn.addEventListener('click', function () { remove(s, btn); });
      actions.appendChild(btn);
      tr.appendChild(actions);
      tbody.appendChild(tr);
    });
  }

  function load() {
    return api('GET', '/api/admin/subscribers').then(function (r) {
      if (r.status === 401) return showLogin(r.data && r.data.error === 'session_expired' ? 'Session expirée après 30 minutes d’inactivité.' : '');
      if (r.status !== 200) return show($('panel-msg'), 'Erreur de chargement.', 'error');
      $('admin-login').hidden = true;
      $('admin-panel').hidden = false;
      render(r.data.subscribers);
    });
  }

  function remove(s, btn) {
    if (!window.confirm('Supprimer définitivement ' + s.email + ' ? Cette action est irréversible.')) return;
    btn.disabled = true;
    api('DELETE', '/api/admin/subscribers/' + encodeURIComponent(s.id)).then(function (r) {
      if (r.status === 401) return showLogin('Session expirée. Reconnectez-vous.');
      if (r.status === 204 || r.status === 404) {
        show($('panel-msg'), 'Inscription supprimée : ' + s.email, 'ok');
        return load();
      }
      btn.disabled = false;
      show($('panel-msg'), 'La suppression a échoué.', 'error');
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    $('login-form').addEventListener('submit', function (e) {
      e.preventDefault();
      var pwd = $('password').value;
      if (!pwd) return;
      api('POST', '/api/admin/login', { password: pwd }).then(function (r) {
        $('password').value = '';
        if (r.status === 200) { show($('login-msg'), ''); return load(); }
        if (r.status === 429) {
          var min = Math.ceil((Number(r.headers.get('Retry-After')) || 900) / 60);
          return show($('login-msg'), 'Trop de tentatives. Réessayez dans ' + min + ' min.', 'error');
        }
        var left = r.data && typeof r.data.remaining === 'number' ? ' Tentatives restantes : ' + r.data.remaining + '.' : '';
        show($('login-msg'), 'Mot de passe incorrect.' + left, 'error');
        $('password').focus();
      });
    });
    $('logout').addEventListener('click', function () {
      api('POST', '/api/admin/logout').then(function () { showLogin(''); });
    });
    load();
  });
})();
