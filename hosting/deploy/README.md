# Deploy the PUBLIC install endpoint (e2-micro on GCP)

Stands up a small public VM that serves the installer at
`https://platform.{prod,edu}.internal.paraframes.org/install.sh` — reachable
from anywhere (your Mac included) with a real, trusted HTTPS cert.

> "internal" is just part of the hostname. The record lives in your **public**
> paraframes.org DNS, so it resolves everywhere. Both hostnames point at the
> same VM; users pick the env with `--env edu` at install time.

## One-time setup (run in order)

```bash
cd hosting/deploy
cp config.env.example config.env      # fill in PROJECT/ZONE/VPC/CERTBOT_EMAIL (config.env is gitignored)
gcloud auth login                     # if not already authenticated

./provision-vm.sh                     # static IP + VM (nginx+certbot) + firewall + optional DNS
#   -> prints the PUBLIC IP

# Point BOTH hostnames at that IP in your PUBLIC paraframes.org DNS.
#   - if you set DNS_ZONE (a public Cloud DNS zone), provision-vm.sh did it for you
#   - otherwise add the A records now, and wait for them to resolve:
#       dig +short platform.prod.internal.paraframes.org   # should show the IP

./enable-tls.sh                       # real Let's Encrypt certs for both hosts (auto-renews)
./deploy.sh                           # publishes the installer files to the VM
```

Test from your Mac:

```bash
curl -fsSL https://platform.prod.internal.paraframes.org/install.sh | bash
curl -fsSL https://platform.edu.internal.paraframes.org/install.sh | bash -s -- --env edu
```

No `PARACLIENT_INSECURE` needed — the cert is trusted.

## Updating the client later

Edit `paraclient.py` / `install.sh` / `requirements-client.txt`, then:

```bash
cd hosting/deploy && ./deploy.sh
```

Users update by re-running the one-liner. Nothing else to touch (cert renews on
its own).

## What each file does

| file                 | role                                                              |
|----------------------|------------------------------------------------------------------|
| `config.env.example` | copy to `config.env`; environment settings                       |
| `startup-script.sh`  | VM boot: nginx + certbot, serves `/var/www/paraclient` over HTTP  |
| `provision-vm.sh`    | reserve static IP, create VM + firewall (80/443 open) + DNS       |
| `enable-tls.sh`      | certbot → real HTTPS for both hostnames (run after DNS resolves)  |
| `deploy.sh`          | `publish.sh` + push `dist/` to the web root, reload nginx         |

## Ordering matters

`enable-tls.sh` must run **after** DNS points at the VM — Let's Encrypt proves
you control the names via an HTTP challenge on port 80. If you run it too early
it fails harmlessly; just fix DNS and re-run.

## Teardown

```bash
gcloud compute instances delete "$VM_NAME" --zone "$ZONE"
gcloud compute firewall-rules delete paraclient-web-allow
gcloud compute addresses delete "${VM_NAME}-ip" --region "${ZONE%-*}"
# and remove the two A records
```
