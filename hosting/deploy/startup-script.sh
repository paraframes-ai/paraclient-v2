#!/usr/bin/env bash
# Runs as root on the VM at first boot (passed as the instance startup-script).
# Installs nginx, an interim self-signed cert for both platform hosts, and the
# vhost that serves /var/www/paraclient at the web root. deploy.sh fills that
# dir with the actual files afterward.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update -y
apt-get install -y nginx openssl rsync

install -d -m 755 /var/www/paraclient

# --- interim TLS: self-signed cert with a SAN covering both hosts ----------
# Replace with an internal-CA-issued cert for platform.{prod,edu}.internal...
# once you have one (drop it at the same paths and `systemctl reload nginx`).
CRT=/etc/ssl/paraframes
install -d -m 700 "$CRT"
if [ ! -f "$CRT/platform.crt" ]; then
  openssl req -x509 -nodes -newkey rsa:2048 -days 825 \
    -keyout "$CRT/platform.key" -out "$CRT/platform.crt" \
    -subj "/CN=platform.prod.internal.paraframes.org" \
    -addext "subjectAltName=DNS:platform.prod.internal.paraframes.org,DNS:platform.edu.internal.paraframes.org"
  chmod 600 "$CRT/platform.key"
fi

cat > /etc/nginx/sites-available/paraclient <<'NGINX'
server {
    listen 80;
    listen 443 ssl;
    server_name platform.prod.internal.paraframes.org platform.edu.internal.paraframes.org;

    ssl_certificate     /etc/ssl/paraframes/platform.crt;
    ssl_certificate_key /etc/ssl/paraframes/platform.key;

    root /var/www/paraclient;
    location / { try_files $uri =404; }
}
NGINX

ln -sf /etc/nginx/sites-available/paraclient /etc/nginx/sites-enabled/paraclient
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable nginx
systemctl restart nginx
