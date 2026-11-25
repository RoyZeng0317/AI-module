"""Narrow pixel heuristic for clean 64px, single filled shapes on white."""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

PALETTE = {"red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 170, 65), "yellow": (235, 195, 30)}

def infer(path):
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    fail = lambda reason: {"scorable": False, "reason": reason, "color": None, "shape": None}
    if pixels.shape != (64, 64, 3): return fail("requires 64x64")
    distances = np.stack([np.linalg.norm(pixels - c, axis=2) for c in PALETTE.values()])
    foreground = np.min(pixels, axis=2) < 230
    ys, xs = np.where(foreground)
    if not 200 <= len(xs) <= 1900: return fail("foreground area")
    # Most foreground must have a single palette color; reject blur, gradients and noise.
    labels = distances.argmin(axis=0)
    counts = np.bincount(labels[foreground], minlength=4)
    color_index = int(counts.argmax())
    if counts[color_index] / len(xs) < .98: return fail("mixed color")
    if np.mean(distances[color_index][foreground] <= 45) < .98: return fail("off palette or blurred")
    if np.mean(np.max(abs(pixels[~foreground] - 255), axis=1) > 12) > .01: return fail("background")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1-x0+1, y1-y0+1
    if min(x0, y0) < 2 or max(x1,y1) > 61 or not .9 <= w/h <= 1.1: return fail("bounds or aspect")
    widths, centers = [], []
    for row in foreground[y0:y1+1]:
        columns = np.flatnonzero(row)
        if not len(columns) or len(columns) != columns[-1]-columns[0]+1: return fail("holes or fragmented mask")
        widths.append(len(columns)); centers.append((columns[0]+columns[-1])/2)
    widths = np.asarray(widths) / w
    if np.ptp(centers) > 1.5: return fail("asymmetric object")
    fill = len(xs)/(w*h)
    t = (np.arange(h)+.5)/h
    profiles = {"square": np.ones(h), "circle": np.sqrt(np.maximum(0, 1-(2*t-1)**2)), "triangle": t}
    errors = {s: float(np.mean(abs(widths-p))) for s,p in profiles.items()}
    shape = min(errors, key=errors.get)
    ranges = {"square": (.96,1.01), "circle": (.73,.85), "triangle": (.46,.57)}
    low, high = ranges[shape]
    if errors[shape] > .065 or not low <= fill <= high: return fail("ambiguous geometry")
    return {"scorable": True, "reason": None, "color": list(PALETTE)[color_index], "shape": shape}

def evaluate(manifest):
    manifest = Path(manifest)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if isinstance(entries, dict): entries = entries["samples"]
    if not isinstance(entries, list) or not entries: raise ValueError("Manifest must contain nonempty samples")
    details, groups, by_split, by_combo = [], {}, {}, {}
    for entry in entries:
        target = entry.get("target", entry.get("metadata", entry))
        if "color" not in target or "shape" not in target:
            colors = {"紅色": "red", "藍色": "blue", "綠色": "green", "黃色": "yellow"}
            shapes = {"圓形": "circle", "正方形": "square", "三角形": "triangle"}
            prompt = entry.get("prompt", entry.get("caption", ""))
            matches = [(c, s) for cz,c in colors.items() for sz,s in shapes.items() if prompt == cz+sz]
            if len(matches) != 1: raise ValueError("Use explicit target labels or one exact supported caption")
            color, shape = matches[0]
            target = {**target, "color": color, "shape": shape}
        color, shape = target["color"], target["shape"]
        if color not in PALETTE or shape not in ("circle", "square", "triangle"): raise ValueError("invalid target labels")
        combo = f"{color}_{shape}"
        canonical_split = "dev" if combo in {"red_triangle", "blue_square"} else "test" if combo in {"green_circle", "yellow_triangle"} else "train"
        if target.get("combo", entry.get("combo", combo)) != combo: raise ValueError("combo disagrees with target labels")
        split = target.get("split", entry.get("split", canonical_split))
        if split != canonical_split: raise ValueError("split disagrees with canonical v1 combination")
        try: prediction = infer(manifest.parent / entry["image"])
        except (OSError, ValueError) as exc: prediction = {"scorable": False, "reason": str(exc), "color": None, "shape": None}
        group = "seen" if split == "train" else "unseen" if split in ("dev","test") else "unknown"
        results = {"color_correct": prediction["color"] == color, "shape_correct": prediction["shape"] == shape}
        results["joint_correct"] = all(results.values())
        for table, key in ((groups, group), (by_split, split), (by_combo, combo)):
            counters = table.setdefault(key, {"count": 0, "unscorable": 0, "color_correct": 0, "shape_correct": 0, "joint_correct": 0})
            counters["count"] += 1; counters["unscorable"] += not prediction["scorable"]
            for metric, value in results.items(): counters[metric] += value
        details.append({"image": entry["image"], "expected": {"color": color, "shape": shape}, "split": split, "combo": combo, "prediction": prediction, **results})
    for table in (groups, by_split, by_combo):
        for counters in table.values(): counters["joint_accuracy"] = counters["joint_correct"] / counters["count"]
    return {"groups": groups, "by_split": by_split, "by_combo": by_combo, "details": details, "limitation": "Narrow clean-shape heuristic; human inspection still required. All unscorable outputs count as failures. Aggregate unseen scores do not establish test or per-combination success."}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(evaluate(args.manifest), ensure_ascii=False, indent=2)
    if args.output:
        root = Path(__file__).resolve().parents[2]
        if not args.output.resolve().is_relative_to(root): raise ValueError("Output must stay inside project")
        with args.output.open("x", encoding="utf-8") as stream: stream.write(result)
    else: print(result)
