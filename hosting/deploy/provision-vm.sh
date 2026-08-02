#!/usr/bin/env bash
# Provision the PUBLIC ParaClient install-endpoint: a reserved static public IP,
# an e2-micro VM (nginx + certbot via startup-script), a firewall opening 80/443
# to the internet, and — if DNS_ZONE is set — the public A records for both
# platform hosts. Safe to re-run (create-or-update on records; existing
# VM/IP/firewall are reported, not fatal).
#
#   cp config.env.example config.env && $EDITOR config.env
#   ./provision-vm.sh
#
# Then: point DNS at the printed IP (auto if DNS_ZONE set), run ./enable-tls.sh,
# then ./deploy.sh.
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
REGION="${ZONE%-*}"
IP_NAME="${VM_NAME}-ip"

NET_ARGS=(--network="$VPC")
[ -n "${SUBNET:-}" ] && NET_ARGS+=(--subnet="$SUBNET")

echo "==> Reserving static public IP $IP_NAME in $REGION"
gcloud compute addresses create "$IP_NAME" --project="$PROJECT" --region="$REGION" \
  || echo "   (address may already exist — continuing)"
IP=$(gcloud compute addresses describe "$IP_NAME" --project="$PROJECT" \
  --region="$REGION" --format='get(address)')
echo "==> Public IP: $IP"

echo "==> Creating VM $VM_NAME ($MACHINE_TYPE, public) in $ZONE"
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --image-family=debian-12 --image-project=debian-cloud \
  --address="$IP" \
  "${NET_ARGS[@]}" \
  --tags=paraclient-web \
  --metadata-from-file=startup-script=startup-script.sh \
  || echo "   (VM may already exist — continuing)"

echo "==> Opening 80/443 to the internet (tag paraclient-web)"
gcloud compute firewall-rules create paraclient-web-allow \
  --project="$PROJECT" --network="$VPC" \
  --direction=INGRESS --action=ALLOW --rules=tcp:80,tcp:443 \
  --target-tags=paraclient-web \
  --source-ranges=0.0.0.0/0 \
  || echo "   (firewall rule may already exist — continuing)"

if [ -n "${DNS_ZONE:-}" ]; then
  for fqdn in "${HOST_PROD:?}." "${HOST_EDU:?}."; do
    echo "==> DNS $fqdn -> $IP (public zone $DNS_ZONE)"
    gcloud dns record-sets create "$fqdn" --project="$PROJECT" --zone="$DNS_ZONE" \
      --type=A --ttl=300 --rrdatas="$IP" 2>/dev/null \
    || gcloud dns record-sets update "$fqdn" --project="$PROJECT" --zone="$DNS_ZONE" \
      --type=A --ttl=300 --rrdatas="$IP"
  done
else
  echo "==> DNS_ZONE not set — add these PUBLIC A records yourself:"
  echo "      ${HOST_PROD:?}  A  $IP"
  echo "      ${HOST_EDU:?}   A  $IP"
fi

cat <<EOF

==> VM provisioned. Next:
    1. Make sure both hostnames resolve to $IP  (dig +short ${HOST_PROD})
    2. ./enable-tls.sh     # gets real Let's Encrypt certs for both hosts
    3. ./deploy.sh         # pushes the installer files
EOF
