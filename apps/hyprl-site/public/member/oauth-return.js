'use strict';
(async () => {
  const params = new URLSearchParams(location.search);
  const code = params.get('code'), state = params.get('state');
  history.replaceState(null, '', '/auth/discord/callback');
  try {
    if (!code || !state || params.has('error') || params.getAll('code').length !== 1 || params.getAll('state').length !== 1) throw Error();
    const response = await fetch('/auth/discord/complete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ code, state }), credentials: 'same-origin', cache: 'no-store', signal: AbortSignal.timeout(20000) });
    if (!response.ok) throw Error();
    location.replace('/membre');
  } catch { document.getElementById('result').textContent = 'Connexion non aboutie. Vérifiez votre rôle dans le serveur HyprL, puis recommencez la connexion.'; }
})();
