# Validation record

All work was performed remotely under `/data/scratch/ashwin`; no project files
were created on the laptop. Initial work was committed on `muse-glimmer`;
`paraclient-v7` is the user-authorized branch prepared for pull-request review.

## Storage and environment

- Storage benchmark 2197295: direct 2,000 MiB writes measured 577 MB/s on
  `/data/scratch` and 166 MB/s on `/data/scratch-oc40`. Selected
  `/data/scratch/ashwin`; both temporary test files were removed.
- Initial setup jobs 2197316/2197372 exposed missing `ensurepip` and an incomplete
  venv. The setup script now bootstraps pip in its own NFS environment.
- Setup 2197400 installed the packages but stopped on NVIDIA's nonstandard SBSA
  wheel tag. Setup **2197519 completed** after checking the exact warning, actual
  AArch64 ELF libraries, CUDA availability, bf16 support, and a CUDA operation.
- No package source build failed. The remaining `pip check` warning is specific
  to `nvidia-cusparselt-cu12==0.7.1` / `manylinux2014_sbsa`; all other failures
  remain fatal. The fallback is a compatible ARM NVIDIA NGC PyTorch image through
  Apptainer, if needed; no container was deployed.
- Exact versions/hardware: `environment.json`, `../slurm/requirements-gh200.lock`.

## Runtime model inspection

- Full BF16 load and finite forward pass: **2197527**, completed.
- Model commit: `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`.
- `AutoModelForImageTextToText` resolves to `MuseGlimmerForConditionalGeneration`.
- 29,776,626,688 parameters total; 1,852,639,744 in `model.vision_tower`.
- 416 language linear targets; attention includes its additional `gate_proj`.
  Vision tower, adapter, and projection remain outside LoRA.
- Native template assertions exercised private `to=self`, student `to=user`,
  ATEM calls, correlated tool results, generation prefix, and rejection of JSON
  strings where native dictionary tool arguments are required.
- Adapter preflight **2197664**, completed: 104,792,064 trainable LoRA parameters,
  no trainable vision parameters, 505 tokens in the rendered example, successful
  Qwen tokenizer regression, and compatible gradient checkpointing.
- A preflight caught Transformers 5's removal of `warmup_ratio`; training now
  adapts to fractional `warmup_steps` while retaining the Transformers 4 field.
- Full target names and rendered example: `muse_inspection.json`.

## One training smoke allocation

**Job 2197693: COMPLETED, exit 0:0, elapsed 00:01:51, requested limit 00:45:00.**

- `build_dataset.py --base muse --subject math --limit 200`, plus four native
  tool examples: 404 dialogues, exactly 202 for each mode.
- BF16, r=16, alpha=32, dropout=0.05, sequence length 512, batch 1, accumulation 2.
- Actual SIGUSR1 after the first checkpoint caused a complete save/exit at step 2.
  The second process automatically resumed from checkpoint-2 and finished step 6
  within the same allocation. This was the only training job submitted.
- Logged per-step losses: 2.135, 1.807, 1.203, 0.9098, 0.6417, 0.9157. All finite.
  Do not interpret six steps as a quality evaluation. Trainer's aggregate loss
  after resume has different accounting; per-step values are clearer evidence.
- Peak allocated CUDA memory: 62,817,204,736 bytes.
- Final adapter weights: 419,293,392 bytes (~400 MiB), plus ~28 MB tokenizer data.
  Only checkpoint-5 and checkpoint-6 remain. Serving copies need the final adapter
  and tokenizer/config files, not the optimizer checkpoint directories.
- Safetensors headers verified **832 A/B tensors, all language-only**. PEFT 0.21
  minimized the saved target list. Full paths and the pinned model revision were
  restored in final/checkpoint JSON metadata; tensor files were unchanged. The
  trainer now preserves those fields on future saves.
- Adapter: `../adapters/smoke-2197693/math`.
- Compact machine-readable results: `smoke_result.json`.
- Detailed logs: `smoke-2197693/phase1.log`, `smoke-2197693/phase2.log` (ignored by Git).

## Safety and other validation

- **24 regression tests passed**, including actual FastAPI proxy requests with
  mocked tutor responses. Tests assert private reasoning/tool content never
  reaches `screen_output` or the student; malformed channels fail closed.
- Existing input escalation and independent output blocking passed.
- Dataset validation covers native call schemas, IDs/results, invalid inputs,
  both supported conversation paths, and incomplete-checkpoint avoidance.
- Python compilation, shell syntax, and `git diff --check` passed.
- `content_filter.py`, `merge_lora.py`, and legacy `requirements.txt` are unchanged.
- The final metadata-only preservation change was syntax-checked and verified
  against actual saved tensor headers; no additional training job was submitted.

## Open work requiring later inputs or authorization

- Full subject-array training is prepared but not submitted; obtain explicit
  approval before that or any additional GPU work.
- Language arts, civics, and general still need reviewed datasets. The four math
  tool fixtures are illustrative, not a representative research corpus.
- No live behavioral eval was run: no serving workload was authorized. The
  expanded evaluator is implemented; its model-dependent outcomes remain unknown.
- `serve_tardy.sh` was source/syntax checked only. Tardy was not contacted, so its
  installed vLLM architecture/LoRA/parser support and two-A100 memory fit remain
  unverified. The script checks support before starting.
- The independent proxy strips framing/structured private fields; unmarked
  reasoning embedded in ordinary text still requires behavioral evaluation.
  The current proxy has no application tool-execution loop.
- Scratch is unbacked. The user subsequently authorized publishing `paraclient-v7` for review. Archive
  final adapter/report files to durable storage separately; they are excluded from Git.
