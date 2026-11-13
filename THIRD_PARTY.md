# Origins and licenses

- Native ThinkMorph is downloaded from https://github.com/ThinkMorph/ThinkMorph at the revision in `provenance/upstream.json`. Its source license is Apache-2.0 and remains in the downloaded checkout.
- Evaluation uses https://github.com/hychaochao/VLMEvalKit_Thinkmorph at the recorded revision, plus this repository's patch. Its source license is Apache-2.0. `patches/vlmeval.patch` records the local modifications.
- SDE/GRPO mathematics and the independent historical test oracle originate from the user's BAGEL-Canvas/MathCanvas implementation. The BAGEL-Canvas source carries Apache-2.0. The original numerical functions are preserved in `tests/fixtures/historical_math.py`; the training adaptations add native ThinkMorph trajectories, grounding, masking and replay.
- A copy of Apache-2.0 is included at `licenses/Apache-2.0.txt`. Existing notices in upstream and adapted files remain applicable. This private research snapshot does not assert a new blanket license for all original additions.
- Model and dataset repositories have their own terms and metadata. Their pinned repository identities are in `provenance/`; fetching them does not transfer their ownership or alter their terms. No model weights, raw dataset image/parquet files or trained adapters are distributed through Git here.
