# ThinkMorph grounded GRPO

Native ThinkMorph interleaved text/image GRPO with a Gemini or Qwen VLM judge, multi-GPU training, exact sample accounting, online W&B media, and seeded MMVP/VStar evaluation. This `thinkmorph` branch captures the November 13, 2026 setup.

**Start with [setup/README.md](setup/README.md).** It covers the environment, every model/data download, credentials, training, resume, and evaluation commands. No separate MathCanvas checkout is required.

## Included configurations

| Profile | Source | Text rollout batch | Image batch | VAE | Replay checkpointing | Evidence |
|---|---|---:|---:|---|---|---|
| `production` | `snapshots/production/thinkmorph_grpo` | 2 | 1 | CPU FP32 | on | Current A100 training implementation |
| `optimized` (default) | `src/thinkmorph_grpo` | 8 | 2 | CUDA FP32 | auto | Two RTX GPUs, two synchronized optimizer updates and reload checks |

Both use eight rollouts **per question**, one question per GPU per wave, LoRA rank 64/alpha 128, direct observation judging, and parallel text replay for both old and new likelihoods. RL sampling and replay disable CFG (`cfg_text_scale=cfg_img_scale=1`). Evaluation retains the established CFG text 4/image 2 settings.

The optimized trial saw four questions/32 rollouts: three question groups contributed gradients; one Jigsaw group was skipped after judge validation failed. Image batching does not promise a 2× end-to-end speedup. Multi-question-per-GPU experiments are included as library code but are not enabled by these profiles. H200 throughput has not been measured.

The 64 training source files are preserved byte-for-byte, with hashes in [provenance/source_snapshots.json](provenance/source_snapshots.json). Portable launch/download/evaluation wrappers are new; their validation scope is recorded in [validation/release_checks.json](validation/release_checks.json). This repository does not change any existing training run.

## Contents

- `src/`, `snapshots/production/`: native sampling, image SDE likelihoods, text replay, GRPO, LoRA, distributed coordination, checkpoint/RNG restoration, sample history, judge prompts/validators/transports.
- `configs/`: frozen training recipe, preview settings, execution profiles.
- `manifests/`: exact ordered 2,000-question selection and four-question smoke selection, source row identities, revisions, question/content/image hashes. Questions/images are reconstructed from the pinned public datasets.
- `scripts/`: pinned downloads, selective data export, training, setup checks, adapter export, seeded evaluation, missing-answer rescoring.
- `logging/`: independent CPU W&B logger with input images, generated images, rollout tables, rewards, accuracy, timing and memory.
- `evaluation/`, `patches/`: established benchmark settings and the exact VLMEvalKit modifications used by the completed evaluations.
- `tests/`, `validation/`: CPU regression tests, historical numerical oracle, batching measurements and release checks.
- `docs/method.md`: grounding state, reward assignment, limitations and reproduction boundaries.

Model weights, dataset images/parquets, training outputs, trained adapters, API credentials and machine-specific service files are excluded. A full checkpoint is required to resume training; an adapter export is sufficient for evaluation. See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream origins and licenses.
