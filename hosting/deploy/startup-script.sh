#!/usr/bin/env bash
# Runs as root on the VM at first boot (passed as the instance startup-script).
# Installs nginx + certbot and serves /var/www/paraclient over HTTP. TLS is
# added later by enable-tls.sh (certbot needs DNS pointing here first, which
# can't be true at boot). deploy.sh fills the web root with the actual files.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update -y
apt-get install -y nginx rsync certbot python3-certbot-nginx

install -d -m 755 /var/www/paraclient

cat > /etc/nginx/sites-available/paraclient <<'NGINX'
server {
    listen 80;
    listen [::]:80;
    server_name platform.prod.internal.paraframes.org platform.edu.internal.paraframes.org;

    root /var/www/paraclient;
    location / { try_files $uri =404; }
}
NGINX

ln -sf /etc/nginx/sites-available/paraclient /etc/nginx/sites-enabled/paraclient
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable nginx
systemctl restart nginx
