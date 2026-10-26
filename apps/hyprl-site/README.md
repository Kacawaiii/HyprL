# HyprL site

Imported from the standalone site, excluding all runtime data, installed dependencies,
private configuration and operator contact details. Node.js >= 20.6.

```sh
npm ci
npm test
```

Configure the runtime using `.env.example`; keep the populated file outside Git.
`NOTIFY_EMAIL` has no code default. Subscriber storage is private runtime data.
