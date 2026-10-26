'use strict';
const http = require('node:http');
const crypto = require('node:crypto');
const { random, fail } = require('./security');
class RouterClient {
  constructor(config) { this.config = config; }
  call(customer, action, fields = {}) {
    if (!this.config.routerSocket) throw fail(503, 'router_unavailable');
    const raw = JSON.stringify({ customer, action, ...fields });
    const at = String(Math.floor(Date.now() / 1000)), nonce = random();
    const signature = crypto.createHmac('sha256', Buffer.from(this.config.routerKey, 'hex')).update(at + '.' + nonce + '.' + raw).digest('hex');
    return new Promise((resolve, reject) => {
      const request = http.request({ socketPath: this.config.routerSocket, path: '/member/v1', method: 'POST', headers: {
        'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(raw), 'X-HyprL-At': at, 'X-HyprL-Nonce': nonce, 'X-HyprL-Signature': signature,
      } }, response => {
        let data = '';
        response.on('data', chunk => { data += chunk; if (data.length > 65536) request.destroy(); });
        response.on('end', () => {
          if (response.statusCode !== 200) return reject(fail([400, 403, 404].includes(response.statusCode) ? response.statusCode : 503, 'router_refused'));
          try { resolve(JSON.parse(data)); } catch { reject(fail(503, 'router_unavailable')); }
        });
        response.on('error', () => reject(fail(503, 'router_unavailable')));
      });
      request.setTimeout(5000, () => request.destroy());
      request.on('error', () => reject(fail(503, 'router_unavailable')));
      request.end(raw);
    });
  }
}
module.exports = { RouterClient };
