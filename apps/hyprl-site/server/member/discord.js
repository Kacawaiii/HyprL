'use strict';
const { fail } = require('./security');
const API = 'https://discord.com/api/v10';
class Discord {
  constructor(config, publicUrl, request = fetch) { Object.assign(this, { config, publicUrl, request }); }
  async call(url, options = {}) {
    let response;
    try { response = await this.request(url, { ...options, redirect: 'error', signal: AbortSignal.timeout(5000) }); }
    catch { throw fail(503, 'discord_unavailable'); }
    if (!response.ok) throw fail(response.status === 401 || response.status === 403 || response.status === 404 ? 401 : 503, 'discord_unavailable');
    const raw = await response.text();
    if (raw.length > 65536) throw fail(503, 'discord_unavailable');
    try { return JSON.parse(raw); } catch { throw fail(503, 'discord_unavailable'); }
  }
  async token(fields) {
    const result = await this.call(API + '/oauth2/token', {
      method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ client_id: this.config.clientId, client_secret: this.config.clientSecret, ...fields }).toString(),
    });
    if (typeof result.access_token !== 'string' || !result.access_token || typeof result.refresh_token !== 'string' || !Number.isFinite(result.expires_in) || result.expires_in <= 0 || result.token_type?.toLowerCase() !== 'bearer') throw fail(503, 'discord_unavailable');
    return result;
  }
  exchange(code, verifier) { return this.token({ grant_type: 'authorization_code', code, code_verifier: verifier, redirect_uri: this.publicUrl + '/auth/discord/callback' }); }
  refresh(token) { return this.token({ grant_type: 'refresh_token', refresh_token: token }); }
  user(token) { return this.call(API + '/users/@me', { headers: { Authorization: 'Bearer ' + token } }); }
  async roles(token) {
    const result = await this.call(API + '/users/@me/guilds/' + this.config.guildId + '/member', { headers: { Authorization: 'Bearer ' + token } });
    if (!Array.isArray(result.roles) || result.pending === true || result.roles.some(r => !/^\d{17,20}$/.test(r))) throw fail(403, 'membership_required');
    return result.roles;
  }
}
module.exports = { Discord };
