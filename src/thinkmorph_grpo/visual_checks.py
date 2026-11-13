"""Original-image pixel/template hints, not automatic semantic rewards."""
import numpy as np
from PIL import Image


def diagnostic_crop_content(original, after, diagnostics, max_regions=2):
    """Supply visual detail for suspicious chart rows, never an automatic label.

    Coordinates use the same thumbnail/alignment as visual_diagnostics. Full
    images remain available to judge whether this correspondence is meaningful.
    """
    import base64
    from io import BytesIO
    import cv2
    regions = [r for r in diagnostics.get("original_bar_regions", [])
               if r.get("possible_missing_mark")][:max_regions]
    if not regions:
        return [], []
    source = Image.open(original).convert("RGB")
    source.thumbnail((1200, 1200))
    result = Image.open(after).convert("RGB")
    result.thumbnail((1200, 1200))
    result = Image.fromarray(cv2.resize(np.asarray(result), source.size))
    content, metadata = [], []
    for index, region in enumerate(regions):
        x1, y1, x2, y2 = region["original_bbox_xyxy_pixels"]
        # Nearby rows help distinguish a label background from a plotted bar.
        margin = max(30, 3*(y2-y1))
        box = [0, max(0, y1-margin), source.width, min(source.height, y2+margin)]
        metadata.append({"region_index": index, "crop_xyxy_pixels": box,
                         "original_mark_xyxy_pixels": [x1, y1, x2, y2],
                         "alignment": "after resized to original thumbnail; inspect full images for layout changes"})
        for label, image in (("ORIGINAL", source), ("AFTER", result)):
            stream = BytesIO()
            image.crop(box).save(stream, format="PNG")
            content.extend([{"type": "text", "text": f"DETAIL {index} {label}: same chart row coordinates {box}. Verify actual bar presence, independently of label/value. This crop is a diagnostic, not ground truth."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()}}])
    return content, metadata


def visual_diagnostics(original, before, after, task):
    import cv2
    def load(path):
        image = Image.open(path).convert("RGB")
        image.thumbnail((1200, 1200))
        return np.asarray(image)
    source, result = load(original), load(after)
    info = {"diagnostic_only": True, "limitations": "Crops, redraws and occlusions can confound matching; inspect the images."}
    if task == "Chart_Refocus" and abs(source.shape[1] / source.shape[0] - result.shape[1] / result.shape[0]) < .05:
        result = cv2.resize(result, (source.shape[1], source.shape[0]))
        def colored(a):
            return ((a.max(-1).astype(float) - a.min(-1)) > 65) & (a.min(-1) < 180)
        _, _, stats, _ = cv2.connectedComponentsWithStats(colored(source).astype(np.uint8))
        bars = []
        for x, y, w, h, area in stats[1:]:
            if w > source.shape[1] * .035 and h > 3 and w > 2 * h and area / (w * h) > .65:
                margin = max(1, h // 5)
                box = result[y + margin:y + h - margin, x + 2:x + w - 2]
                if not box.size:
                    continue
                bars.append({"original_bbox_xyxy_pixels": [int(x), int(y), int(x + w), int(y + h)],
                    "after_colored_fraction": round(float(colored(box).mean()), 3),
                    "possible_missing_mark": bool(colored(box).mean() < .10)})
        info["original_bar_regions"] = sorted(bars, key=lambda b: b["original_bbox_xyxy_pixels"][1])
        def red(a):
            a = a.astype(float)
            return (a[:, :, 0] > 140) & (a[:, :, 0] > a[:, :, 1] * 1.35) & (a[:, :, 0] > a[:, :, 2] * 1.35)
        previous = load(before)
        if abs(previous.shape[1] / previous.shape[0] - source.shape[1] / source.shape[0]) < .05:
            previous = cv2.resize(previous, (source.shape[1], source.shape[0]))
            old_red = cv2.dilate(red(previous).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            info["new_red_pixel_fraction"] = round(float((red(result) & ~old_red).mean()), 6)
    if task == "Jigsaw_Assembly":
        h, w = source.shape[:2]
        white = (source.min(-1) > 240).mean(0)
        candidates = np.flatnonzero((white > .85) & (np.arange(w) > w * .4) & (np.arange(w) < w * .6))
        if candidates.size:
            left, right = int(candidates.min()), int(candidates.max()) + 1
            matches = []
            target = cv2.cvtColor(cv2.resize(result, (w, h)), cv2.COLOR_RGB2GRAY)
            for name, part in (("input_left_piece", source[:, :left]), ("input_right_piece", source[:, right:])):
                ph, pw = part.shape[:2]
                patch = cv2.cvtColor(part[int(ph*.08):int(ph*.92), int(pw*.08):int(pw*.92)], cv2.COLOR_RGB2GRAY)
                best = (-1., None)
                for scale in (.85, .95, 1., 1.05, 1.15):
                    resized = cv2.resize(patch, None, fx=scale, fy=scale)
                    if resized.shape[0] > h or resized.shape[1] > w:
                        continue
                    _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(target, resized, cv2.TM_CCOEFF_NORMED))
                    if score > best[0]:
                        best = (score, loc)
                matches.append({"piece": name, "best_landmark_correlation": round(best[0], 3),
                    "after_top_left_pixels": best[1]})
            info["piece_landmark_matches"] = matches
    return info


def identical_image_pixels(before, after):
    """Exact duplicate image actions add no visual information; text is separate."""
    with Image.open(before) as left, Image.open(after) as right:
        return left.size == right.size and left.convert("RGB").tobytes() == right.convert("RGB").tobytes()
