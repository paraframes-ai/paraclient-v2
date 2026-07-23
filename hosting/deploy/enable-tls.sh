#!/usr/bin/env bash
# Get real Let's Encrypt certs for both platform hosts and switch nginx to
# HTTPS. Run this AFTER provision-vm.sh and AFTER both hostnames resolve to the
# VM's public IP (certbot proves control via an HTTP challenge on port 80).
# certbot installs a systemd timer, so renewal is automatic afterward.
#
#   ./enable-tls.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -f config.env ] || { echo "copy config.env.example -> config.env first" >&2; exit 1; }
# shellcheck disable=SC1091
. ./config.env
: "${PROJECT:?}" "${ZONE:?}" "${VM_NAME:?}" "${CERTBOT_EMAIL:?set CERTBOT_EMAIL in config.env}"
: "${HOST_PROD:?}" "${HOST_EDU:?}"

echo "==> Requesting certs for $HOST_PROD and $HOST_EDU"
gcloud compute ssh "$VM_NAME" --project="$PROJECT" --zone="$ZONE" --command \
  "sudo certbot --nginx --non-interactive --agree-tos \
     -m '${CERTBOT_EMAIL}' \
     -d '${HOST_PROD}' -d '${HOST_EDU}' \
     --redirect \
   && sudo nginx -t && sudo systemctl reload nginx \
   && echo 'TLS enabled; auto-renew timer:' && systemctl list-timers 'certbot*' --no-pager | head -3"

echo "==> Done. Now run ./deploy.sh, then test from anywhere:"
echo "    curl -fsSL https://${HOST_PROD}/install.sh | bash"
