#!/usr/bin/env bash
#SBATCH --job-name=muse-smoke
#SBATCH --account=quanta
#SBATCH --partition=quanta-gh200
#SBATCH --qos=quanta-main
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200000M
#SBATCH --time=00:45:00
#SBATCH --chdir=/data/scratch/ashwin/paraclient-muse
#SBATCH --output=/data/scratch/ashwin/logs/%x_%j.out
# One allocation: 200 source problems, bounded training, real USR1 save + resume.
set -euo pipefail
source slurm/env.sh
run_dir="artifacts/smoke-$SLURM_JOB_ID"
data_dir="data/smoke-$SLURM_JOB_ID"
output_dir="adapters/smoke-$SLURM_JOB_ID"
mkdir -p "$run_dir"
python -m pytest -q tests --junitxml="$run_dir/unit-tests.xml"
python -u build_dataset.py --base muse --subject math --limit 200 \
    --tool-data examples/math.tools.jsonl --out-dir "$data_dir"
args=(--base muse --subject math --data "$data_dir/math.jsonl" --out-dir "$output_dir"
      --max-steps 6 --max-seq-len 512 --batch-size 1 --grad-accum 2
      --save-steps 1 --save-total-limit 2 --logging-steps 1)
python -u train_adapter.py "${args[@]}" > "$run_dir/phase1.log" 2>&1 &
trainer_pid=$!
(
    # Only signal our own child, after its first complete checkpoint.
    while kill -0 "$trainer_pid" 2>/dev/null; do
        if [[ -f "$output_dir/math/progress.json" ]]; then
            kill -USR1 "$trainer_pid"
            exit 0
        fi
        sleep 1
    done
    exit 1
) &
watcher_pid=$!
set +e
wait "$trainer_pid"
status=$?
wait "$watcher_pid"
watcher_status=$?
set -e
cat "$run_dir/phase1.log"
[[ $status == 75 && $watcher_status == 0 ]] || { echo "Signal-save phase failed: trainer=$status watcher=$watcher_status"; exit 1; }
cp "$output_dir/math/train_results.json" "$run_dir/signal-save.json"
python -u train_adapter.py "${args[@]}" > "$run_dir/phase2.log" 2>&1
cat "$run_dir/phase2.log"
cp "$output_dir/math/train_results.json" "$run_dir/resume.json"
python - "$output_dir/math" "$run_dir" <<'PY'
import json
from pathlib import Path
import sys
output, report = map(Path, sys.argv[1:])
first = json.loads((report / 'signal-save.json').read_text())
second = json.loads((report / 'resume.json').read_text())
assert first['stopped_by'] == 'SIGUSR1' and first['global_step'] < 6
assert second['resumed_from'] and second['global_step'] == 6
assert (output / 'adapter_model.safetensors').stat().st_size > 0
summary = {'signal_save': first, 'resume': second, 'adapter_bytes': (output / 'adapter_model.safetensors').stat().st_size}
(report / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY
