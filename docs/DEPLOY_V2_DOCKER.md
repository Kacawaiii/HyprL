# HyprL V2 – Déploiement docker-compose

Ce guide assemble l’API FastAPI V2, Redis, le portail Streamlit et le bot Discord dans une stack docker-compose unique. Objectif : démarrer la stack en local ou sur un VPS en éditant uniquement les secrets.

## Prérequis

- Docker ≥ 24 et docker compose plugin.
- Token admin HyprL (permet de créer des tokens clients) et token utilisateur (scopes `read:predict`, `read:usage`, `write:session` pour portail/bot).
- Secrets Discord (bot token, application configurée avec les slash commands).

## Préparation des variables d’environnement

Dans `deploy/`, des fichiers `.env.*.example` sont fournis. Copiez-les puis remplacez les valeurs par vos secrets réels :

```bash
cd deploy
cp .env.api.example .env.api
cp .env.portal.example .env.portal
cp .env.bot.example .env.bot
# Éditez les fichiers copiés avec vos valeurs (tokens, titres…)
```

- `.env.api` : contient `HYPRL_ADMIN_TOKEN` (utilisé pour bootstrapper /v2/tokens) et éventuellement `HYPRL_PREDICT_IMPL=real`.
- `.env.portal` : `HYPRL_API_TOKEN` issu de `/v2/tokens` + titre custom.
- `.env.bot` : même `HYPRL_API_TOKEN` (scopes read/predict/usage/write session) et `DISCORD_BOT_TOKEN`.

> Conseil : les tokens issus de `/v2/tokens` se présentent sous la forme `tok_xxx.yyyzzz...`. Gardez-les secrets.

## Génération d’un token utilisateur

1. Démarrer l’API localement (hors Docker) ou via la stack pour disposer d’un admin token (`HYPRL_ADMIN_TOKEN`).
2. Appeler :

```bash
curl -X POST "$HYPRL_API_BASE/v2/tokens" \
  -H "Authorization: Bearer $HYPRL_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "account_id": "hyprl_portal_bot",
        "scopes": ["read:predict","read:usage","write:session"],
        "credits_total": 100000,
        "label": "portal+bot"
      }'
```

3. Copier `token_plain` dans `.env.portal` et `.env.bot`.

## docker-compose : build & run

Depuis `deploy/` :

```bash
# Validation du fichier compose
docker compose -f docker-compose.v2.yml config > /dev/null

# Build + démarrage détaché
docker compose -f docker-compose.v2.yml up --build -d
```

La stack démarre quatre services :

| Service | Port | Description |
| --- | --- | --- |
| `redis` | interne | Rate-limit distribué + jobs RT |
| `api` | 8000 | FastAPI V2 (tokens/predict/sessions/autorank) |
| `portal` | 8501 | Streamlit (Usage, Sessions, Predict Monitor) |
| `bot` | N/A | Bot Discord (slash commands hyprl_ping, predict, sessions…) |

Les volumes `../data` et `../logs` sont montés dans le conteneur API pour conserver les sessions et exports sur l’hôte.

## Vérifications rapides

1. API health : http://localhost:8000/health → doit renvoyer `{"ok": true}`.
2. Portal : http://localhost:8501 → pages Usage/Sessions/Predict Monitor accessibles.
3. Bot : vérifier dans Discord la commande `/hyprl_ping` (dans un serveur où l’application est enregistrée).

## Sécurité & bonnes pratiques

- Ne jamais commiter les `.env` réels.
- Ajouter un reverse proxy TLS (Caddy, NGINX…) devant l’API/portal pour exposition publique.
- Restreindre l’accès à l’API via firewall ou allowlist IP si possible.
- Utiliser `HYPRL_PREDICT_IMPL=real` uniquement lorsque votre modèle est packagé dans l’image (sinon laisser `stub`).
- Pour Redis en production, activer la persistance ou pointer vers un service managé.

## Arrêt & nettoyage

```bash
docker compose -f docker-compose.v2.yml down
# Supprimer les volumes anonymes si nécessaire
docker compose -f docker-compose.v2.yml down -v
```

Les répertoires `data/` et `logs/` restent sur l’hôte ; sauvegardez-les avant toute suppression.
