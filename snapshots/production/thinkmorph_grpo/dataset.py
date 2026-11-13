"""Explicitly separate policy inputs, terminal labels, and unused SFT traces."""
from pathlib import Path
import json
import re

TASK_REPOS = {task: "ThinkMorph/" + task for task in
    ("Visual_Search", "Chart_Refocus", "Jigsaw_Assembly", "Spatial_Navigation")}


def export_examples(task, revision, output, limit=None):
    """Stream a pinned training split into the trainer's lightweight manifest.

    Reference text/images are intentionally not exported. This also handles the
    source dataset's misspelled resoning_thought_* columns by never consuming them.
    """
    from datasets import load_dataset
    if task not in TASK_REPOS or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Require a released training task and an exact dataset commit")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rows = load_dataset(TASK_REPOS[task], revision=revision, split="train", streaming=True)
    manifest = []
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        image = row["problem_image_0"].convert("RGB")
        image_path = root / f"problem_{index:06d}.png"
        image.save(image_path)
        manifest.append({"task":task,"repo":TASK_REPOS[task],"revision":revision,
            "split":"train","row_index":index,"pid":row["pid"],"question":row["question"],
            "problem_image":{"path":str(image_path.resolve())},"answer":row["answer"]})
    (root / "manifest.json").write_text(json.dumps(manifest,indent=2))
    return root / "manifest.json"
