#!/usr/bin/env bash
# Provision the ParaClient install-endpoint VM (internal-only e2-micro), a
# firewall rule for 80/443, and — if DNS_ZONE is set — the A records for both
# platform hosts. Idempotent-ish: safe to re-run (create-or-update on records,
# create failures on existing VM/firewall are surfaced, not swallowed).
#
#   cp config.env.example config.env && $EDITOR config.env
#   ./provision-vm.sh
#
# Then run ./deploy.sh to push the files.
set -euo pipefail
cd "$(dirname "$0")"
[ -f config.env ] || { echo "copy config.env.example -> config.env first" >&2; exit 1; }
# shellcheck disable=SC1091
. ./config.env
: "${PROJECT:?set PROJECT in config.env}"
: "${ZONE:?set ZONE in config.env}"
: "${VPC:?set VPC in config.env}"
: "${VM_NAME:?set VM_NAME in config.env}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-micro}"

NET_ARGS=(--network="$VPC")
[ -n "${SUBNET:-}" ] && NET_ARGS+=(--subnet="$SUBNET")

echo "==> Creating VM $VM_NAME ($MACHINE_TYPE, internal-only) in $ZONE"
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --image-family=debian-12 --image-project=debian-cloud \
  --no-address \
  "${NET_ARGS[@]}" \
  --tags=paraclient-web \
  --metadata-from-file=startup-script=startup-script.sh

echo "==> Allowing 80/443 from ${ALLOW_CIDR:-10.0.0.0/8} to tag paraclient-web"
gcloud compute firewall-rules create paraclient-web-allow \
  --project="$PROJECT" --network="$VPC" \
  --direction=INGRESS --action=ALLOW --rules=tcp:80,tcp:443 \
  --target-tags=paraclient-web \
  --source-ranges="${ALLOW_CIDR:-10.0.0.0/8}" \
  || echo "   (firewall rule may already exist — continuing)"

IP=$(gcloud compute instances describe "$VM_NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --format='get(networkInterfaces[0].networkIP)')
echo "==> VM internal IP: $IP"

if [ -n "${DNS_ZONE:-}" ]; then
  for h in prod edu; do
    fqdn="platform.$h.internal.paraframes.org."
    echo "==> DNS $fqdn -> $IP (zone $DNS_ZONE)"
    gcloud dns record-sets create "$fqdn" --project="$PROJECT" --zone="$DNS_ZONE" \
      --type=A --ttl=300 --rrdatas="$IP" 2>/dev/null \
    || gcloud dns record-sets update "$fqdn" --project="$PROJECT" --zone="$DNS_ZONE" \
      --type=A --ttl=300 --rrdatas="$IP"
  done
else
  echo "==> DNS_ZONE not set — create A records yourself:"
  echo "    platform.prod.internal.paraframes.org -> $IP"
  echo "    platform.edu.internal.paraframes.org  -> $IP"
fi

echo "==> Done. Wait ~1 min for the startup script, then: ./deploy.sh"
