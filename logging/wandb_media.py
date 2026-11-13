"""Bounded original/generated image and trace tables; no model/GPU imports."""
from pathlib import Path
import re


def final_answer(trace):
    text = "\n".join(e.get("text", "") for e in trace if e.get("kind") == "text")
    answers = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return answers[-1].strip() if answers else None


def build_media(wandb, wave, segment, examples, read_json, read_image, *,
                image_side=640, input_side=256, max_generated_images=64, text_limit=12000):
    """Every question gets a thumbnail/IDs; one rotating rank gets N8 images.

    All complete rollout traces receive summary rows. Detailed media preserves
    the distinction between input images and generated images; no reference/GT
    image is implied. Hard bounds and any omissions are explicitly logged.
    """
    counts = dict(input_images=0, generated_images=0, generated_images_available=0,
                  generated_images_omitted=0, rollout_rows=0, text_cells_truncated=0)
    def text(value, limit=text_limit):
        value = "" if value is None else str(value)
        if len(value) > limit:
            counts["text_cells_truncated"] += 1
            return value[:limit] + f"\n[truncated; original {len(value)} characters, full trace retained on storage]"
        return value
    def image(path, caption, side):
        return wandb.Image(read_image(path, side), caption=caption)
    questions = wandb.Table(columns=["global_wave", "global_step", "rank", "group", "sample_id", "task",
        "question", "expected_answer", "accepted", "terminal_accuracy", "input_image"])
    rollouts = wandb.Table(columns=["global_wave", "global_step", "rank", "group", "sample_id", "rollout",
        "completed", "terminal_correct", "answer", "image_rewards", "text_rewards", "generated_steps",
        "scored_text_tokens", "proposal_excerpt", "last_observation_excerpt", "trace_locator"])
    comparison = wandb.Table(columns=["global_wave", "global_step", "rank", "group", "sample_id", "rollout",
        "reasoning_step", "input_image", "generated_image", "proposal", "observation", "image_reward",
        "text_reward", "observation_reward", "terminal_answer", "terminal_correct", "judge_evidence", "trace_locator"])
    groups = []
    workers = sorted((w for w in wave["workers"] if w.get("assigned")), key=lambda w: w["rank"])
    for worker in workers:
        group_index = worker["group"]
        folder = segment / f"group_{group_index:06d}"
        info = read_json(folder/"group.json")
        example = info.get("example", examples.get(group_index, {}))
        sample_id = example.get("example_id", worker.get("example_id", str(group_index)))
        source = Path(example["problem_image"]["path"])
        thumb = image(source, f"Input | {sample_id} | {example.get('question', '')}", input_side)
        counts["input_images"] += 1
        records = []
        for rollout in range(8):
            rollout_folder = folder/f"rollout_{rollout:02d}"
            trace = read_json(rollout_folder/"trace.json")
            judgment = read_json(rollout_folder/"judge.json", optional=not worker.get("accepted", False)) or {}
            texts = [e.get("text", "") for e in trace if e.get("kind") == "text"]
            images = [e for e in trace if e.get("kind") == "image"]
            correct = judgment.get("terminal", {}).get("correct")
            image_rewards = [s.get("image") for s in judgment.get("steps", []) if s.get("process") is not None]
            text_rewards = [s.get("text") for s in judgment.get("steps", [])]
            locator = str((rollout_folder/"trace.json").relative_to(segment))
            rollouts.add_data(wave["wave"], wave["optimizer_updates"], worker["rank"], group_index,
                sample_id, rollout, judgment.get("completed"), correct, final_answer(trace),
                str(image_rewards), str(text_rewards), len(images),
                sum(len(e.get("target_ids", [])) for e in trace if e.get("kind") == "text"),
                text(texts[0] if texts else "", min(4000, text_limit)),
                text(texts[-1] if len(texts)>1 else "", min(4000, text_limit)), locator)
            counts["rollout_rows"] += 1
            records.append((rollout, rollout_folder, trace, judgment, locator))
        correct_values = [r[3].get("terminal", {}).get("correct") for r in records]
        correct_values = [float(v) for v in correct_values if isinstance(v, (int, float))]
        accuracy = sum(correct_values)/len(correct_values) if correct_values else None
        questions.add_data(wave["wave"], wave["optimizer_updates"], worker["rank"], group_index,
            sample_id, example.get("task"), text(example.get("question")), text(example.get("answer")),
            worker.get("accepted"), accuracy, thumb)
        groups.append((worker, example, records, source))
    payload = {"samples/questions": questions, "samples/rollouts": rollouts}
    if not groups:
        return payload, counts
    # Cycle through available ranks, including a partial final wave.
    selected = groups[wave["wave"] % len(groups)]
    worker, example, records, source = selected
    sample_id = example.get("example_id", worker.get("example_id"))
    original = image(source, f"Original input | {sample_id} | wave {wave['wave']}", image_side)
    payload["samples/input_image"] = original
    counts.update(selected_rank=worker["rank"], selected_group=worker["group"])
    generated, ranked = [], []
    for rollout, folder, trace, judgment, locator in records:
        text_step = -1
        for index, event in enumerate(trace):
            if event.get("kind") == "text":
                text_step += 1
                continue
            if event.get("kind") != "image":
                continue
            counts["generated_images_available"] += 1
            proposal = next((e.get("text", "") for e in reversed(trace[:index]) if e.get("kind") == "text"), "")
            observation = next((e.get("text", "") for e in trace[index+1:] if e.get("kind") == "text"), "")
            steps = judgment.get("steps", [])
            reward = steps[text_step] if 0 <= text_step < len(steps) else {}
            correct = judgment.get("terminal", {}).get("correct")
            image_reward = reward.get("image")
            caption = (f"{sample_id} | rollout {rollout} | reasoning step {text_step} | "
                       f"image reward={image_reward} | terminal correct={correct}\n"
                       f"Proposal: {proposal[:250]}\nObservation: {observation[:250]}")
            generated_image = None
            if counts["generated_images"] < max_generated_images:
                path = (folder/event["file"]).resolve()
                if folder.resolve() not in path.parents:
                    raise ValueError("Generated image path escapes its rollout folder")
                generated_image = image(path, caption, image_side)
                generated.append(generated_image)
                counts["generated_images"] += 1
                if isinstance(image_reward, (int, float)):
                    ranked.append((float(image_reward), generated_image))
            else:
                counts["generated_images_omitted"] += 1
            process = reward.get("process") or {}
            comparison.add_data(wave["wave"], wave["optimizer_updates"], worker["rank"], worker["group"],
                sample_id, rollout, text_step, original, generated_image, text(proposal), text(observation),
                image_reward, reward.get("text"), reward.get("post_text"), final_answer(trace), correct,
                text(process.get("evidence")), locator)
    payload["samples/comparison"] = comparison
    if generated:
        payload["samples/generated_images"] = generated
    if ranked:
        payload["samples/best_image"] = max(ranked, key=lambda item: item[0])[1]
        payload["samples/worst_image"] = min(ranked, key=lambda item: item[0])[1]
    payload.update({"media/"+key: value for key, value in counts.items()})
    return payload, counts
