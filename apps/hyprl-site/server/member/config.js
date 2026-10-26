'use strict';
const path = require('node:path');
function memberConfig(env, site) {
  if (env.MEMBER_ENABLED !== 'true') return { enabled: false };
  const roles = Object.fromEntries(['early', 'essentiel', 'premium', 'fondateur', 'equipe'].map(t => [t, env['DISCORD_ROLE_' + t.toUpperCase()]]));
  const config = {
    enabled: true, clientId: env.DISCORD_CLIENT_ID, clientSecret: env.DISCORD_CLIENT_SECRET,
    guildId: env.DISCORD_GUILD_ID, roles, identityKey: env.MEMBER_IDENTITY_KEY,
    stateFile: env.MEMBER_STATE_FILE, snapshotDir: env.MEMBER_SNAPSHOT_DIR,
    routerSocket: env.MEMBER_ROUTER_SOCKET || '', routerKey: env.MEMBER_ROUTER_KEY || '',
    billingOrigin: env.MEMBER_BILLING_ORIGIN || '',
  };
  if (![config.clientId, config.guildId, ...Object.values(roles)].every(x => /^\d{17,20}$/.test(x || '')) ||
      new Set(Object.values(roles)).size !== 5 || !config.clientSecret || !/^[a-f0-9]{64}$/.test(config.identityKey || '')) throw Error('Member configuration: Discord identity, distinct role IDs and 32-byte hex identity key required');
  for (const v of [config.stateFile, config.snapshotDir]) {
    if (!v || !path.isAbsolute(v) || (site.production && (v === site.root || v.startsWith(site.root + path.sep)))) throw Error('Member configuration: dedicated absolute external paths required');
  }
  if (config.routerSocket && (!path.isAbsolute(config.routerSocket) || !/^[a-f0-9]{64}$/.test(config.routerKey))) throw Error('Member configuration: private router socket/key required');
  if (config.billingOrigin && (new URL(config.billingOrigin).origin !== config.billingOrigin || !config.billingOrigin.startsWith('https://'))) throw Error('Member configuration: exact HTTPS billing origin required');
  return config;
}
module.exports = { memberConfig };
