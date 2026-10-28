'use strict';
// A separate timer asks the owning HTTP process to send. It never opens the subscriber store.
async function main(env = process.env) {
  const port = Number(env.PORT || 3000);
  if (!Number.isInteger(port) || port < 1 || port > 65535 || !/^[a-f0-9]{64}$/.test(env.NEWSLETTER_WORKER_KEY || '')) throw Error('Newsletter worker configuration invalid');
  const res = await fetch('http://127.0.0.1:' + port + '/api/internal/newsletter/send', {
    method: 'POST', headers: { Authorization: 'Bearer ' + env.NEWSLETTER_WORKER_KEY }, redirect: 'error', signal: AbortSignal.timeout(240000),
  });
  if (!res.ok) throw Error('Newsletter request failed (' + res.status + ')');
  const { edition, sent, uncertain, skipped } = await res.json();
  console.log(JSON.stringify({ edition, sent, uncertain, skipped }));
}
if (require.main === module) main().catch(() => { console.error('Weekly newsletter failed; review privately'); process.exitCode = 1; });
module.exports = { main };
