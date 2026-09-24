#!/usr/bin/env bash
#SBATCH --job-name=muse-subject
#SBATCH --account=quanta
#SBATCH --partition=quanta-gh200
#SBATCH --qos=quanta-main
#SBATCH --array=0-3
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200000M
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@600
#SBATCH --chdir=/data/scratch/ashwin/paraclient-muse
#SBATCH --output=/data/scratch/ashwin/logs/%x_%A_%a.out
# Full training: prepared for explicit user submission/approval; not a smoke job.
set -euo pipefail
source slurm/env.sh
subjects=(math language_arts civics general)
subject=${subjects[$SLURM_ARRAY_TASK_ID]}
data_dir=${DATA_DIR:-data/muse}
output_dir=${ADAPTER_DIR:-adapters/muse}
if [[ ! -f "$data_dir/$subject.jsonl" ]]; then
    srun python -u build_dataset.py --base muse --subject "$subject" --out-dir "$data_dir"
fi
set +e
srun python -u train_adapter.py --base muse --subject "$subject" \
    --data "$data_dir/$subject.jsonl" --out-dir "$output_dir" \
    --rank 16 --alpha 32 --dropout 0.05 --save-steps 50 --save-total-limit 2
status=$?
set -e
if [[ $status == 75 ]]; then
    # Python exits 75 only after saving model, optimizer, scheduler, and RNG state.
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    exit "$status"
fi
