'use strict';
const crypto = require('node:crypto');
const { LIMITS } = require('../server/lib/signup-guard');
function solve(token) {
  for (let i = 0; ; i++) if (crypto.createHash('sha256').update(token + ':' + i).digest('hex').startsWith('0000')) return String(i);
}
async function proof(base, advance) {
  const r = await fetch(base + '/api/form-token');
  const c = await r.json();
  if (!c.token) throw Error('Test challenge exhausted');
  const work = solve(c.token);
  advance(LIMITS.minMs);
  return { form_token: c.token, form_work: work };
}
module.exports = { proof, solve };
