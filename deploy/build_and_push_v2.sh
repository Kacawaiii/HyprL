#!/bin/sh
set -eu

: "${HYPRL_REGISTRY_PREFIX:=your-registry/hyprl}"
: "${HYPRL_TAG:=v2-dev}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

API_IMAGE="${HYPRL_REGISTRY_PREFIX}-api:${HYPRL_TAG}"
PORTAL_IMAGE="${HYPRL_REGISTRY_PREFIX}-portal:${HYPRL_TAG}"
BOT_IMAGE="${HYPRL_REGISTRY_PREFIX}-bot:${HYPRL_TAG}"

printf 'Building images with prefix %s and tag %s\n' "$HYPRL_REGISTRY_PREFIX" "$HYPRL_TAG"

cd "$REPO_ROOT"

docker build -t "$API_IMAGE" -f "$SCRIPT_DIR/Dockerfile.api" .
docker build -t "$PORTAL_IMAGE" -f "$SCRIPT_DIR/Dockerfile.portal" .
docker build -t "$BOT_IMAGE" -f "$SCRIPT_DIR/Dockerfile.bot" .

printf 'Pushing %s\n' "$API_IMAGE"
docker push "$API_IMAGE"
printf 'Pushing %s\n' "$PORTAL_IMAGE"
docker push "$PORTAL_IMAGE"
printf 'Pushing %s\n' "$BOT_IMAGE"
docker push "$BOT_IMAGE"
