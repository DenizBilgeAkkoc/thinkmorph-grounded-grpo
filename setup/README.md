# Setup and reproduction

Run these commands from the repository root. The release contains the code and download identities; it does not contain pretrained weights, private credentials, or trained LoRA weights.

## 1. Machine and Python environment

The validated stack uses Linux x86_64, Python 3.10, PyTorch 2.10.0 with CUDA 12.8, torchvision 0.25.0, FlashAttention 2.8.3, transformers 4.49.0 and PEFT 0.18.1. Ubuntu 24.04 is suitable with a matching NVIDIA driver/toolkit and a Python 3.10 environment; its default Python 3.12 is not this pinned environment.

Use a GPU image with the CUDA 12.8 development toolkit available (`nvcc --version`), a driver capable of running CUDA 12.8, and Git, curl, a C++ compiler, libGL and libglib. PyTorch wheels provide CUDA runtime libraries; building FlashAttention also needs the toolkit/compiler. On an Ubuntu machine you administer:

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential libgl1 libglib2.0-0
```

Create Python 3.10 with your environment manager (for example `conda create -n thinkmorph python=3.10` followed by `conda activate thinkmorph`). Then:

```bash
git clone --branch thinkmorph git@github.com:DenizBilgeAkkoc/thinkmorph-grounded-grpo.git
cd thinkmorph-grounded-grpo
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r setup/requirements.txt
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install --no-deps -e .
source setup/env.sh
```

Keep these versions together. The installed environment used for the previous GPU runs supplied the versions listed here; release checks reuse that environment. A completely fresh Ubuntu image and every hardware/compiler combination have not been tested by this packaging step.

Each worker loads one complete BF16 ThinkMorph model and FP32 trainable LoRA state on one GPU. Production was exercised on A100 80 GB; the optimized image-batch-2 trial peaked at about 56.2 GiB allocated / 60 GiB reserved on RTX PRO 6000 96 GB. Those measurements do not establish a worst-case memory bound. Use a short smoke run before scaling a new device. A40s were used for inference, not validated for this training configuration.

Store `runs/` on persistent storage. Each committed wave writes full LoRA + Adam state, per-rank RNG and audit artifacts; storage grows throughout training. Preserve full checkpoint triplets and their sample history. Do not delete an active run's files to reclaim space.

## 2. Download the native source and model

```bash
python scripts/download_assets.py
```

This fetches native ThinkMorph into `vendor/ThinkMorph` and downloads approximately 29.56 GB of model assets into `models/ThinkMorph-7B`. Every model file is checked against its recorded digest. Public downloads do not require a Hugging Face write token.

| Asset | Repository | Pinned revision |
|---|---|---|
| Native source | [ThinkMorph/ThinkMorph](https://github.com/ThinkMorph/ThinkMorph) | `c1a48adfa212259c8ad79dfd9d05d87c27340cef` |
| Base/SFT policy + VAE/tokenizer/configs | [ThinkMorph/ThinkMorph-7B](https://huggingface.co/ThinkMorph/ThinkMorph-7B) | `ad1a4abccca4df166f2c9d2178eb33ed61ae367e` |
| Evaluation source | [VLMEvalKit_Thinkmorph](https://github.com/hychaochao/VLMEvalKit_Thinkmorph) | `cddb6a38ac5dfaf64020d6aa3964506e26006568` + `patches/vlmeval.patch` |

The base model is already the ThinkMorph policy; a separate BAGEL initialization or private SFT adapter is not needed. Model file sizes/hashes are in `provenance/model.json`. For an existing model directory:

```bash
python scripts/download_assets.py --source-only
python scripts/download_assets.py --verify-only --model-dir /persistent/models/ThinkMorph-7B
```

Pass the same `--model-dir` to training/evaluation, and `--native-source` to training if you place the native checkout elsewhere. Keep these locations stable during a run: paths are part of its frozen configuration. Downloads require network access; clear inherited `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` when necessary.

## 3. Rebuild the exact training data

```bash
python scripts/prepare_data.py
# Optional separate, four-question smoke export:
python scripts/prepare_data.py --selection manifests/smoke_4.selection.json --output data/smoke
```

The full command writes `data/train2000/manifest.json` and original input PNGs. It reads only `pid`, `question`, `answer`, and `problem_image_0` from pinned parquet byte ranges, caches ranges under `cache/parquet-ranges`, and verifies original image bytes, RGB pixels, question identity, answer/content identity and exported PNG bytes. It does not export reference reasoning or generated images. Re-running requires a fresh output directory; existing manifests are not overwritten.

| Training dataset | HF repository | Revision | Selected |
|---|---|---|---:|
| Visual search | [ThinkMorph/Visual_Search](https://huggingface.co/datasets/ThinkMorph/Visual_Search) | `f4e1839ca893a151355b7c843b737bf907cf0a05` | 500 |
| Chart refocus | [ThinkMorph/Chart_Refocus](https://huggingface.co/datasets/ThinkMorph/Chart_Refocus) | `f088c9c5b1084e1287fd4f969e6e3bcbcfd0d0e6` | 500 |
| Jigsaw assembly | [ThinkMorph/Jigsaw_Assembly](https://huggingface.co/datasets/ThinkMorph/Jigsaw_Assembly) | `9f437eb85a7b62cc65dadd930f6c0a665f0ea6b4` | 500 |
| Spatial navigation | [ThinkMorph/Spatial_Navigation](https://huggingface.co/datasets/ThinkMorph/Spatial_Navigation) | `b7567d9f2fcd096f5bf225cc4e64fc5c2b360897` | 500 |

All selected rows are from `train`. The first 1,000 came from seed 20261111; the next 1,000 from seed 20261112, excluding the first selection and 14 judge-calibration examples. The committed selector is authoritative: do not reshuffle or regenerate it from a seed alone. Source metadata and row/group indices are included. With 8 GPUs, this schedule mostly groups the same task in a wave; another world size changes wave composition.

The decoded/exported images are much smaller than the upstream datasets, but range caching still reads entire selected parquet column chunks and can require substantial disk space. `download_audit.jsonl` records bytes transferred and cache hits. No image or answer is inferred from a hash.

## 4. Credentials and preflight

Create local files `secrets/gemini_api_key` and optionally `secrets/wandb_api_key`, each containing only its respective key. Enter keys using your editor or secret manager, then restrict permissions:

```bash
mkdir -p secrets
chmod 700 secrets
chmod 600 secrets/gemini_api_key secrets/wandb_api_key
python scripts/check_setup.py --gpu
python -m pytest -q tests
```

If W&B is unused, omit its file from `chmod`. Gemini also accepts `GEMINI_API_KEY`. No credential values belong in command arguments, JSON configuration or Git. `secrets/` is ignored. The training launcher uses the key-file path only, and RPC auditing redacts credentials and inline image payloads.

The recorded judge ID is `gemini-3.8-flash`, temperature 0, thinking `low`. Startup verifies the provider's model identity and records its returned version. This ID must be available to the account. An unavailable model fails explicitly; choosing `--gemini-model ANOTHER_ID` starts a different judge configuration requiring its own calibration. There is no silent substitution. Gemini is accessed through the native REST client; training does not require the Google SDK.

## 5. Smoke run, production profile, and optimized profile

Prepare without starting GPUs or calling the judge:

```bash
python scripts/train.py --run-dir runs/smoke --manifest data/smoke/manifest.json \
  --profile optimized --gpus 2 --questions 4 --max-updates 2 \
  --judge-workers 1 --judge-http2 --prepare-only
```

Then start that same run, with the same arguments and without `--prepare-only`:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py \
  --run-dir runs/smoke --manifest data/smoke/manifest.json \
  --profile optimized --gpus 2 --questions 4 --max-updates 2 \
  --gemini-key-file secrets/gemini_api_key --judge-workers 1 --judge-http2 \
  --wandb-key-file secrets/wandb_api_key --wandb-entity YOUR_ENTITY
```

Use identical judge settings in preparation and execution; they are frozen in the run contract. The historical optimized four-question pilot used these conservative judge settings. The ordinary default is eight workers per rank.

For the production A100 configuration and first 1,000 selected questions:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/train.py \
  --run-dir /persistent/runs/thinkmorph-1000 --manifest data/train2000/manifest.json \
  --profile production --gpus 8 --questions 1000 \
  --gemini-key-file secrets/gemini_api_key --judge-workers 8 \
  --wandb-key-file secrets/wandb_api_key --wandb-entity YOUR_ENTITY
```

Use `--profile optimized` in a **new run directory** to use text batch 8, image batch 2, CUDA VAE and automatic replay checkpointing. `--questions 2000` consumes the full fixed manifest. Eight GPUs process up to eight distinct questions/64 rollouts per wave, with gradients averaged over accepted question objectives. One optimizer update is not one sample. Fully unscorable N8 groups are skipped after bounded semantic retries; API/network failures are not converted into zero rewards. Skipped questions still count as seen.

The judge starts scoring ready rollouts while other rollouts are generated. `--judge-workers` limits concurrent requests **per rank**, so 8 GPUs × 8 workers can create up to 64 concurrent requests. Direct observation mode removes a separate claim-extraction request. It does not combine all questions into one Gemini batch API call. Increase concurrency only after checking account quotas and latency. `--judge-http2` enables the tested curl HTTP/2 transport; it requires `/usr/bin/curl` with HTTP2 support (`/usr/bin/curl --version`).

For local Qwen, serve `Qwen/Qwen3-VL-32B-Instruct` on an OpenAI-compatible endpoint using separate judge GPUs, then pass `--judge-provider qwen --judge-url http://JUDGE_HOST:PORT/v1`. This is an optional alternate deployment; the current production recipe uses Gemini. Downloading/hosting Qwen is unnecessary for Gemini training.

The wrapper supports one node. The distributed library includes global-rank coordination, but a two-node 16-H200 deployment needs a shared manifest/run path and a coordinated multi-node torchrun launch; the single-node wrapper is not a validated two-node orchestrator. Do not start two independent single-node wrappers on the same run. Start a new run when changing world size; existing checkpoint restores deliberately reject that change.

## 6. Monitor and resume

Training stdout is in `RUN/logs/phase_000000.log`. `Controller_Status.json` reports committed questions and optimizer updates; `phase_*/summary.json` and `wave_*/update.json` contain timing, memory, rewards, skips and rank agreement. Online W&B status and URL are in `W&B_Status.json`, with the logger's own output in `logs/wandb.log`.

The independent W&B reader uploads scalars, question thumbnails, all rollout summaries and generated images for a rotating question each committed wave. `samples/best_image` means the highest image reward **within that question's eight candidates**; it is not a thresholded quality endorsement. See the documented blanket-highlighting case in `docs/method.md`.

A resumable wave has all of:

```text
phase_000000/wave_000123/checkpoint.pt
phase_000000/wave_000123/checkpoint.pt.rng.pt
phase_000000/wave_000123/checkpoint.pt.complete.json
phase_000000/wave_000123/checkpoint_validation.json
phase_000000/wave_000123/update.json
```

Select the latest committed checkpoint by its marker `group` (not mtime), confirm the reload proof, and repeat the original launch command with:

```bash
--resume RUN/phase_000000/wave_000123/checkpoint.pt
```

Keep the same profile, paths, recipe, manifest, GPU count, judge and concurrency. The launcher creates a new phase; it rejects restarting an existing run from base or rolling back past a newer committed checkpoint. `--max-updates` is the **absolute cumulative update target**, not additional updates after resume. To continue from 1,000 to 2,000 questions, the original run must already use the full 2,000-entry manifest; increase `--questions` to 2000 and resume its latest checkpoint. Preserve previous phase directories for attempted/committed sample history and W&B continuity.

Full checkpoints include adapters, Adam moments, sample history, and per-rank Python/NumPy/Torch RNG. They require strict configuration identity. Moving paths, changing source, switching profile/world size or importing a historical external run requires an audited migration, not bypassing restore checks. The portable wrapper does not automatically restart failed runs or prune checkpoints; inspect the failure and explicitly resume. Its process group cleanup only targets processes it launched.

## 7. Export adapters and evaluate MMVP / VStar

Download the pinned evaluation checkout and apply the recorded patch, then install the evaluation extras:

```bash
python scripts/download_assets.py --source-only --eval
python -m pip install -r setup/requirements-eval.txt
source setup/env.sh
python scripts/check_setup.py --eval
```

The optional Gradio web UI is intentionally excluded because its installed version conflicts with the pinned Hugging Face Hub dependency. These command-line image evaluations do not use Gradio or PyAV. VLMEvalKit may print an optional-video or missing-`.env` notice during import; credentials for our scripts are supplied through the explicit key-file arguments above.

Read `policy_digest`, `optimizer_updates`, and `group` from the completed checkpoint marker. The question count is `group + 1`. Export on CPU:

```bash
python scripts/export_adapters.py \
  --checkpoint RUN/phase_000000/wave_000123/checkpoint.pt \
  --output runs/export-update124 \
  --expected-digest DIGEST_FROM_MARKER --expected-update 124 --expected-questions 992
```

The numeric values above illustrate an all-full 8-GPU run; use the actual marker values. No optimizer or base weights are copied into the export. Evaluation loads unmerged FP32 adapters and checks their file checksum, names, shapes, policy digest and enabled layers.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --benchmark MMVP \
  --adapters runs/export-update124 --output runs/eval-update124-MMVP
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --benchmark VStarBench \
  --adapters runs/export-update124 --output runs/eval-update124-VStar
```

VLMEvalKit automatically downloads [MMVP.tsv](http://opencompass.openxlab.space/utils/VLMEval/MMVP.tsv) (300 questions, MD5 `8cb732b141a0cba5b42159df2839e557`) and [VStarBench.tsv](https://huggingface.co/datasets/xjtupanda/VStar_Bench/resolve/main/VStarBench.tsv) (191 questions, MD5 `b18854d7075574be06b631cd5f7d2d6a`) into `data/benchmarks`. Its pinned loader validates these hashes; the evaluation protocol records the downloaded TSV SHA256 too. Source URLs are not immutable, so keep the validated cache for future reproduction.

Seeds are `42 + original row position` for Python, NumPy, CPU Torch and CUDA. Inference uses temperature 0.3, 4,096 text tokens, up to three rounds, 50 image integration points, and CFG text 4/image 2. Each worker repeats its first question and checks text plus generated-image pixel hashes. Cross-hardware bitwise equivalence is not promised. Raw text, prompts, images, seeds, latency and memory are saved per question. A partial `--limit` run must have its own output directory.

For two GPUs, run the same benchmark/output with `--shards 2 --shard 0` on one visible GPU and `--shards 2 --shard 1` on the other. Wait for both `workers/*/COMPLETE.json`, then run the same command with `--score-only`. Do not change shard assignment while resuming an output directory. Scoring uses the native exact matcher and reports question accuracy plus MMVP both-correct pair accuracy.

Optional format-failure recovery, without regenerating answers:

```bash
python scripts/rescore_missing.py --benchmark MMVP \
  --predictions runs/eval-update124-MMVP/checkpoint_MMVP_exact_matching_result.xlsx \
  --output runs/eval-update124-MMVP-fallback --gemini-key-file secrets/gemini_api_key
```

This only considers missing final answers or exact-matcher `Z` results. The judge receives the question/options and terminal response, **never the image or answer key**. Two extractions must agree; disagreements receive a third vote, otherwise abstain. Original results remain unchanged and revised scores are labeled separately. The paper used GPT-5 for answer extraction; this Gemini fallback is not an exact reproduction of that judge.
