#!/usr/bin/env bash
# Publish the install payload and push it to the VM's web root. Run this after
# provision-vm.sh, and again whenever paraclient.py / install.sh / the
# requirements change. Clients pick up the update by re-running the one-liner.
#
#   ./deploy.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
[ -f "$HERE/config.env" ] || { echo "copy config.env.example -> config.env first" >&2; exit 1; }
# shellcheck disable=SC1091
. "$HERE/config.env"
: "${PROJECT:?}" "${ZONE:?}" "${VM_NAME:?}"

echo "==> Building payload"
"$ROOT/hosting/publish.sh" "$ROOT/hosting/dist" >/dev/null
ls -1 "$ROOT/hosting/dist"

echo "==> Copying to $VM_NAME:/tmp/dist"
gcloud compute ssh "$VM_NAME" --project="$PROJECT" --zone="$ZONE" \
  --command 'rm -rf /tmp/dist'
gcloud compute scp --recurse --project="$PROJECT" --zone="$ZONE" \
  "$ROOT/hosting/dist" "$VM_NAME:/tmp/dist"

echo "==> Installing to /var/www/paraclient and reloading nginx"
gcloud compute ssh "$VM_NAME" --project="$PROJECT" --zone="$ZONE" --command \
  'sudo mkdir -p /var/www/paraclient \
   && sudo rsync -a --delete /tmp/dist/ /var/www/paraclient/ \
   && sudo nginx -t \
   && sudo systemctl reload nginx \
   && echo "deployed: $(ls /var/www/paraclient | tr "\n" " ")"'

echo "==> Verify:"
echo "    curl -fsSL https://platform.prod.internal.paraframes.org/install.sh | bash"
echo "    curl -fsSL https://platform.edu.internal.paraframes.org/install.sh | bash -s -- --env edu"
