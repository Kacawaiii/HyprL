# HyprL V2 Production Deployment (Docker)

This guide shows how to promote the local docker-compose.v2 stack to a hardened VPS deployment using pre-built images, .env secrets, and a reverse proxy.

## 1. Prerequisites
- VPS (Ubuntu/Debian/RHEL) with Docker Engine ≥ 24 and docker compose plugin.
- DNS A/AAAA record pointing `hyprl.example.com` (or your domain) to the VPS.
- Container registry access (Docker Hub, GHCR, etc.).
- Secrets for HYPRL_ADMIN_TOKEN, HYPRL_API_TOKEN(s), and Discord bot token.

## 2. Build & Push Images
Run from the dev machine inside the repo root:

```sh
HYPRL_REGISTRY_PREFIX=ghcr.io/YOUR_ORG/hyprl \
HYPRL_TAG=v2-prod \
sh deploy/build_and_push_v2.sh
```

This script builds `api`, `portal`, and `bot` images from `deploy/Dockerfile.*` and pushes:
- `${HYPRL_REGISTRY_PREFIX}-api:${HYPRL_TAG}`
- `${HYPRL_REGISTRY_PREFIX}-portal:${HYPRL_TAG}`
- `${HYPRL_REGISTRY_PREFIX}-bot:${HYPRL_TAG}`

## 3. Prepare VPS Directory
On the VPS:

```sh
sudo mkdir -p /opt/hyprl
sudo chown "$USER" /opt/hyprl
cd /opt/hyprl
# copy the repo (git clone or rsync) and keep deploy/ + docs/ + requirements/ as needed
```

Populate env files **without committing them**:

`deploy/.env.api`
```
HYPRL_DB_URL=sqlite:///./hyprl_v2.db   # or postgres URI
HYPRL_ADMIN_TOKEN=prod_admin_token_here
REDIS_URL=redis://redis:6379/0
HYPRL_PREDICT_IMPL=stub
```

`deploy/.env.portal`
```
HYPRL_API_BASE=http://api:8000
HYPRL_API_TOKEN=portal_prod_token
HYPRL_PORTAL_TITLE=HyprL Control Panel
```

`deploy/.env.bot`
```
HYPRL_API_BASE=http://api:8000
HYPRL_API_TOKEN=bot_prod_token   # can reuse portal token or provision a dedicated one
DISCORD_BOT_TOKEN=real_discord_token
```

Generate tokens via `/v2/tokens` using the admin token before deploying.

## 4. Launch Prod Compose Stack

```sh
cd /opt/hyprl/deploy
export HYPRL_REGISTRY_PREFIX=ghcr.io/YOUR_ORG/hyprl
export HYPRL_TAG=v2-prod

docker compose -f docker-compose.v2.prod.yml pull
docker compose -f docker-compose.v2.prod.yml up -d

docker compose -f docker-compose.v2.prod.yml ps
```

`docker-compose.v2.prod.yml` uses the registry images and reuses `.env.*` files. Data/log volumes still point to `../data` and `../logs` relative to deploy/; adjust bind mounts if you relocate directories.

## 5. Reverse Proxy & TLS
Pick Caddy or NGINX (examples in `deploy/Caddyfile.example` and `deploy/nginx.conf.example`).

### Caddy
1. Copy `deploy/Caddyfile.example` to `/etc/caddy/Caddyfile`.
2. Replace `hyprl.example.com` with your domain.
3. Ensure the Caddy container/service runs on the same Docker network so it can reach `api` and `portal`.
4. Caddy handles TLS certificates automatically via ACME once port 443 is accessible.

### NGINX
1. Copy `deploy/nginx.conf.example` to `/etc/nginx/nginx.conf` (or include it as a site).
2. Replace domain + certificate paths, then reload NGINX.
3. Terminate TLS at NGINX or front it with a separate TLS terminator.

After the proxy is live:
- `https://hyprl.example.com/` → Streamlit portal
- `https://hyprl.example.com/api/health` → `{ "ok": true }`

## 6. Security Checklist
- Only expose ports 80/443 publicly; keep 8000/8501 internal to Docker network.
- Rotate HYPRL_ADMIN_TOKEN and HYPRL_API_TOKEN regularly.
- Store `data/` and `logs/` on persistent volumes and back them up.
- Enable firewall rules (ufw/nftables) to restrict SSH and other services.
- Monitor container logs (`docker compose logs -f api portal bot`) for auth errors.

## 7. Troubleshooting
- `401 Unauthorized` on `/v2/tokens`: confirm HYPRL_ADMIN_TOKEN inside the `api` container matches the .env file.
- Portal cannot fetch usage: ensure `.env.portal` token scopes include `read:usage` and reverse proxy forwards Authorization headers.
- Discord bot flapping: validate Discord token scopes and that HYPRL_API_TOKEN includes `read:usage` + `read:predict` + `write:session`.

With these steps, HyprL V2 can be promoted from local docker-compose to a VPS with TLS, registry-hosted images, and clearly separated secrets.
