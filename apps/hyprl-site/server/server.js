'use strict';
// Point d'entrée : node server/server.js (les variables d'environnement doivent être définies).

const http = require('node:http');
const { loadConfig } = require('./config');
const { createApp } = require('./app');

let config;
try {
  config = loadConfig();
} catch (err) {
  console.error(err.message);
  process.exit(1);
}

let app;
try {
  app = createApp(config);
} catch (err) {
  console.error(err.message);
  process.exit(1);
}

const server = http.createServer(app);
server.headersTimeout = 15000;
server.requestTimeout = 20000;

server.listen(config.port, config.host, () => {
  console.log('HyprL en écoute sur http://' + config.host + ':' + config.port + ' (URL publique : ' + config.publicUrl + ')');
});

function shutdown() {
  app.close();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 5000).unref();
}
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
