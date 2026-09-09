#!/bin/sh
set -eu

token_file="${ZEVO_UI_TOKEN_FILE:-/run/zevo-ui-auth/token}"
template=/etc/nginx/templates/default.conf.template
target=/etc/nginx/conf.d/default.conf

attempt=0
while [ ! -s "$token_file" ]; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 100 ]; then
    echo "[web] UI access token was not created: $token_file" >&2
    exit 1
  fi
  sleep 0.1
done

ZEVO_UI_ACCESS_TOKEN=$(cat "$token_file")
export ZEVO_UI_ACCESS_TOKEN
# Substitute only the credential. Nginx variables such as `$host` remain.
envsubst '${ZEVO_UI_ACCESS_TOKEN}' < "$template" > "$target"
exec nginx -g 'daemon off;'
