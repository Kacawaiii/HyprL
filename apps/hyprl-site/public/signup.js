/* Same-origin, cookie-free form protection. No third-party script or storage. */
'use strict';
(() => {
  const form = document.querySelector('form[action="/inscription"]');
  if (!form) return;
  const button = form.querySelector('button[type="submit"]');
  const message = document.getElementById('signup-status');
  const encoder = new TextEncoder();
  let proof, preparing;
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
  function say(text) { message.textContent = text; }
  async function prepare() {
    button.disabled = true;
    say('Préparation du formulaire sécurisé…');
    const r = await fetch('/api/form-token', { cache: 'no-store', credentials: 'omit' });
    const challenge = await r.json(), started = Date.now();
    if (!r.ok || !challenge.token || challenge.bits !== 16 || !crypto.subtle) throw Error('Formulaire indisponible. Réessayez plus tard ou utilisez le lien Discord ci-dessous.');
    let nonce = 0;
    for (; nonce < 10000000; nonce++) {
      const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', encoder.encode(challenge.token + ':' + nonce)));
      if (digest[0] === 0 && digest[1] === 0) break;
    }
    if (nonce === 10000000) throw Error('La vérification a échoué. Réessayez ou utilisez Discord.');
    await delay(Math.max(0, challenge.min_ms - (Date.now() - started)));
    proof = { token: challenge.token, work: String(nonce), expires: started + challenge.expires_ms - 5000 };
    button.disabled = false;
    say('Formulaire prêt. Confirmez ensuite votre inscription dans l’unique e-mail reçu.');
  }
  function ensureProof() {
    if (proof && proof.expires > Date.now()) return Promise.resolve();
    if (!preparing) preparing = prepare().finally(() => { preparing = null; });
    return preparing;
  }
  form.addEventListener('focusin', event => {
    if (event.target === message || event.target === button) return;
    if (!proof && !preparing) ensureProof().catch(error => { say(error.message); button.disabled = false; });
  });
  form.addEventListener('submit', async event => {
    event.preventDefault();
    button.disabled = true;
    try {
      await ensureProof();
      button.disabled = true;
      const body = new URLSearchParams(new FormData(form));
      body.set('form_token', proof.token); body.set('form_work', proof.work);
      proof = null;
      const r = await fetch('/api/subscribe', { method: 'POST', credentials: 'omit', headers: { Accept: 'application/json' }, body });
      const response = await r.json();
      if (!r.ok) throw Error(response.detail || 'Vérifiez le formulaire et réessayez.');
      location.assign('/merci');
    } catch (error) {
      say(error.message || 'Envoi impossible. Réessayez ou utilisez Discord.');
      message.focus();
      button.disabled = false;
    }
  });
  // The static submit button stays disabled without JavaScript.
  button.disabled = false;
})();
