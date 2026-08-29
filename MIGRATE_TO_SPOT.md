# Migrating the L4 box to g2-standard-4 Spot

Goal: same box, cheaper. `g2-standard-8` (on-demand, ~$640/mo at 24/7) →
`g2-standard-4` **Spot** (~$150–175/mo at 24/7). Reuse the existing boot disk so
nothing is copied; models, adapters, environments, secrets, account data, and
Tailscale state remain on the existing disk.

## Why it's a recreate, not an edit
You can change an instance's machine type in place, but you CANNOT flip an
existing instance from Regular to Spot — provisioning model is fixed at
creation. So the only way to get Spot is to delete the instance (keeping the
disk) and create a new one from that disk.

## Prep already done on the disk (so it's correct after the move)
- `paraclient-v3` disabled (the 14B CPU tier — dropped to fit 16 GB).
- 8 GB swapfile + `/etc/fstab` entry added (safety net so v4 + v2 + guard fit
  16 GB without an OOM under load).
- Everything else (v4/vLLM, v2, guard, gateway) still enabled and will start.

## What runs on the new box
v4 (GPU) + v2 (7B CPU) + guard + gateway. Measured fit on 16 GB: ~14.5 GB used,
with the 8 GB swap as the cushion. If v2 ever causes memory pressure, disable it
too (`sudo systemctl disable --now paraclient-v2`) and you're at ~11 GB.

## Cutover

Facts you'll need:  project `paraframes-app`, zone `us-east1-b`,
instance `paraclient-l4`, boot disk device `persistent-disk-0`
(get the disk's RESOURCE name from the console: VM → Storage → Boot disk).

### Console path (simplest)
1. Compute Engine → VM instances → **Stop** `paraclient-l4`.
2. Open it → Storage → note the **boot disk name**.
3. **Delete** the instance — in the delete dialog, **uncheck "Delete boot disk"**.
4. **Create instance**:
   - Name `paraclient-l4`, region/zone `us-east1` / `us-east1-b`.
   - Machine: **GPUs → NVIDIA L4 ×1**, then machine type **g2-standard-4**.
   - **Availability policies → VM provisioning model → Spot**.
   - **Boot disk → Change → Existing disks →** select the disk you kept.
   - Create.
5. It boots with everything. Tailscale rejoins automatically (state on disk).

### gcloud equivalent
```bash
gcloud compute instances stop paraclient-l4 --zone=us-east1-b
gcloud compute instances delete paraclient-l4 --zone=us-east1-b --keep-disks=boot
gcloud compute instances create paraclient-l4 \
  --zone=us-east1-b --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --disk=name=<BOOT_DISK_NAME>,boot=yes,auto-delete=no \
  --maintenance-policy=TERMINATE
```

## Verify after boot
```bash
nvidia-smi                                   # L4 present
free -h                                      # swap active
systemctl is-active paraclient-vllm paraclient-v2 paraclient-guard paraclient-gateway
systemctl is-enabled paraclient-v3           # should say: disabled
curl -s -o /dev/null -w '%{http_code}\n' http://100.122.196.7:8080/healthz
```

## Preemption note (Spot)
A Spot VM does not auto-restart when Google reclaims it — it STOPs. For a
stateful box like this, that's the right behavior (no data loss); just start it
again. The watchdog on the always-on e2 box can start it automatically.

## Post-migration fixes applied on the new box (2026-08-09)
The recreate was the box's first reboot in ~19 days and surfaced latent issues:
- **NVIDIA driver**: running kernel 6.8.0-1065-gcp had no driver module (an apt
  kernel upgrade had been sitting unbooted). Installed
  `linux-modules-nvidia-580-server-open-6.8.0-1065-gcp`, added
  `/etc/modules-load.d/nvidia.conf` (nvidia, nvidia_uvm) so it auto-loads on
  every Spot restart, and `apt-mark hold`ed the kernel so a future upgrade
  can't strand the GPU again on an unattended preemption-reboot.
- **v2 dropped too**: on 15 GB, v4(vLLM ~9.4G)+v2+guard forced heavy swap and
  the guard's moderation calls timed out (fail-closed -> everything blocked).
  Disabled paraclient-v2 as well; the box now runs v4 + guard + gateway.
- **guard threads**: serve_guard.sh `-t 8` -> `-t 4` to match the 4 vCPUs;
  moderation went from a 20 s timeout to ~1 s warm.
- **SA scope**: recreated with cloud-platform -> Vertex Gemini works. Claude
  still needs a Vertex quota/Model-Garden subscription (429 quota, not scope).
