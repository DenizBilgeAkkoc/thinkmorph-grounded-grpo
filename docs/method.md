# Method and reproduction boundaries

## State and action grounding

The policy starts with the question and original image. It generates native interleaved text and images for up to three rounds, then a terminal answer. Each generated image becomes part of the context for later text and image actions. Native token sequences, image conditioning, latent transitions, SDE windows and RNG states are retained for replay.

The process judge separates three responsibilities:

1. **Proposal:** assess what the pre-image text intends to inspect or transform using evidence available before the action. A sensible plan can earn credit even if the generated image later fails.
2. **Image transition:** compare original/previous/current images and the proposal. Targeting, preservation, execution and usefulness are assessed separately. Tight but imperfect boxes can receive partial credit; there is no blanket 0.5 cap.
3. **Observation:** assess the following text directly against visible state and task constraints. A plan is not evidence that a claimed visual result occurred. Observation feedback is attached to the text action that makes the claim, rather than rewarding an earlier proposal for a later unsupported statement.

Direct observation mode performs the observation assessment in one judge request instead of a separate text-claim extraction pass followed by visual judging. The schema retains evidence and typed task checks; bounded repair handles recoverable JSON/coverage mistakes. Navigation, chart, search and jigsaw checks remain in the code.

Process judging never receives the reference solution trace or correct final answer. The separately scoped terminal judge can use the answer key to assess final-answer correctness. The terminal policy version is `semantic_target_v2`. Evaluation answer extraction is another separate operation: it sees the model's terminal response and answer choices but no answer key or image.

## Reward and update

The configured process return discounts future image and text rewards with gamma 0.7 and normalizes by valid trajectory lengths. The combined return uses image reward weight 1, text reward weight 0.3 and terminal accuracy weight 0.5; unfinished trajectories receive penalty 0.5. Advantages are normalized within a question's eight rollouts with minimum standard deviation 0.02 and clipped to ±5. PPO clip range is 0.2. The velocity regularizer has coefficient 1e-4 and is a velocity squared-error penalty, not a categorical KL.

Both text and image expert projections have rank-64 LoRA adapters with alpha 128; image/text learning rates are 5e-5/3e-5. The base weights stay frozen. Each distributed wave assigns one question per rank and averages accepted question objectives. Accepted flat groups count in the denominator; invalid groups do not. A rank with no local gradient can still receive the shared global update.

Image sampling uses 15 schedule points (14 transitions), shift 3, noise level 1 and a shared three-transition SDE window per image position across the question's eight rollouts. Outside the window, integration is deterministic. CFG is disabled throughout RL sampling and replay. Only stochastic transitions have an image likelihood for PPO.

Text replay uses full causal sequence scoring for both the old-policy snapshot and new policy. Incremental sampling probabilities are retained separately. BF16 kernels and batching can change these numerical results; matched old/new replay checks do **not** prove exact equality with the sampling distribution. The earlier comparison observed summed action discrepancies of several log-probability units, with mean absolute action discrepancy divided by tokens around 0.0042–0.0044. This is a recorded approximation/limitation, not a claimed bitwise equivalence.

Image batches preserve separate rollout RNG streams and conditioning. Generated image artifacts record the packed layout; replay uses the matching layout and validates its behavior likelihood. Singleton likelihoods must not be reused for wider batches. Text replay remains in pairs, even when text generation batches all eight rollouts.

## Invalid judgments and sample accounting

After bounded semantic assessment retries, an unscorable rollout causes its entire N8 question group to be skipped. Missing rewards are not fabricated. Transport/authentication/quota failures remain operational failures; they are not evidence that the output deserves zero. Validly assessed bad components can receive zero and still participate in training.

`sample_history.py` records source row/revision, question/image identities, planned seeds, sampled artifacts, attempt status and checkpoint/optimizer commits. A question can be seen, fully sampled, skipped, accepted with flat advantages, or gradient-contributing. These are different counts. Full checkpoints bind sample history by digest; previous failed attempts remain visible when explicitly included during resume. `sample_history.json`, `sample_history_summary.json`, individual `sample_history/attempt_*.json`, and checkpoint RNG sidecars provide the records needed to identify which samples were seen.

## Known limitations to inspect

- Reward is relative within each N8 group. In the reviewed chart example, shading every bar earned image reward 0.1875 while all seven alternatives earned zero. That candidate received positive advantage (~1.91), despite low absolute image quality. The unsupported observation still earned zero, and all eight terminal answers were correct. Current code preserves that behavior. A future change should distinguish useful partial localization from indiscriminate highlighting and should be evaluated separately.
- W&B `best_image` means argmax within the selected question's candidates. It does not certify a good edit. Inspect the image score, component evidence and other candidates together.
- A rising aggregate training accuracy can reflect task mix and skipped hard groups. Monitor per-task accuracy, skip rates and held-out benchmarks. Training accuracy is not an unbiased held-out metric.
- The optimized image-batch-2 pilot completed two verified updates, with three of four question groups contributing. This is a functionality check, not a long-run stability or benchmark improvement claim.
- Judge API latency and synchronized-rank stragglers can dominate throughput. The observed compute improvements do not establish an end-to-end 2× speedup. No H200 run has been measured.
- Reproducibility pins code, data, seeds, source/model revisions and judge identity. Hosted judge services and GPU kernels can still change; retain raw judgment audits and environment details with each run.

## What was packaged

The production and optimized training modules are exact source snapshots. `provenance/source_snapshots.json` stores their SHA256 values. `validation/*.txt` preserves earlier trial reports; reports describe the state at their own trial dates, and paths under `EXPERIMENT_ARCHIVE` refer to artifacts outside this Git repository. The later `two_update_pilot.txt` supersedes the earlier compute trial's incomplete-live-run status.

The new portable scripts replace site-specific Slurm/SSH/systemd controllers, absolute storage paths and deployment-plan migrations. They do not carry automatic production recovery permissions or silently migrate old checkpoints. The production filesystem retry helper is included, and required checkpoint corruption remains fatal. Existing production services were not modified by this release.

Release validation covers CPU numerical/judge/batching tests, pinned upstream downloads, data reconstruction, source hashes and wrapper preparation. The archived GPU evidence applies to the preserved source modules. It is not a claim that every new launch wrapper was rerun end-to-end on fresh GPUs during packaging.
