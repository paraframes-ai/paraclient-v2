# CSAIL ParaClient Muse runbook

The detailed commands, data schema, and safety limitations are in [README.md](README.md).
All work is remote at `/data/scratch/ashwin/paraclient-muse`, branch `muse-glimmer`.
Use `ssh -o BatchMode=yes slurm-agent` for every remote command. Never place code,
data, environments, or weights in AFS, and never load/train models on the login node.

1. Run `slurm/setup_env_gh200.sh` through a GH200 Slurm job. It creates
   `/data/scratch/ashwin/envs/muse-arm`, installs cu128 ARM wheels, and verifies CUDA.
   Read `artifacts/environment.json` and the resolved requirements lock. If ARM
   compatibility fails, consider the NVIDIA NGC PyTorch/Apptainer fallback in README.
2. Submit `slurm/inspect_muse.sbatch`. Read `artifacts/muse_inspection.json` for the
   pinned model revision, actual module names, and executed native chat-template
   examples. Stop and ask if Hugging Face authentication becomes necessary.
3. Run the one authorized `slurm/smoke_muse.sbatch`. It builds math with
   `--limit 200`, adds four tool fixtures, runs six steps, and tests SIGUSR1 save and
   automatic resume in a single 45-minute allocation. Read the job-specific summary.
   Do not submit another training smoke job without approval.
4. Prepare reviewed `data/raw/<subject>.synth.jsonl` and optional
   `<subject>.tools.jsonl` for the study. Civics retains mandatory human review.
   Preserve source/generator provenance and license/attribution requirements.
5. The four-day `slurm/train_muse.sbatch` subject array is **not authorized for
   automatic submission**. Ask before running it or any other additional GPU work.
   It trains one adapter per GH200, saves bounded checkpoints, and requeues after
   graceful SIGUSR1 saving. Missing reviewed data fails the corresponding task.
6. Later, copy final adapters and the inspection report to the serving host. Run
   `serve_tardy.sh` there using a compatible x86_64 vLLM environment. Only GPUs 2/3
   are allowed. The script validates installed LoRA/parser support before launch;
   do not silently merge adapters or switch architectures to bypass a failed check.
7. Run `eval_adapter.py --base muse` against an authorized local research backend,
   then human-review a representative sample. Automated numeric answer and recovery
   checks are heuristics, not validation for student deployment.

The safety/content-filter layer remains independent. The proxy extracts only
student-facing `to=user` content before output moderation; private `to=self` and
tool content never enter the filter or student response. Keep vLLM localhost-only.
The current proxy has no application tool execution loop; live tool probes use
controlled fixtures against the research backend.

Scratch is unbacked. Commit code in the remote clone without pushing, archive
compact reports and final adapter files separately, and omit optimizer checkpoints
from serving copies. Existing Qwen workflows retain the original requirements
stack and default NF4 training; do not replace the Muse ARM environment with it.
