"""Staged, inference-only evaluation of raw shape samples (no training)."""
import argparse, hashlib, json, math
from pathlib import Path
from PIL import Image, ImageDraw
from tranning.imagegen.shapes_score import evaluate

ROOT = Path(__file__).resolve().parents[2]
SEEDS = {"dev": [42, 43, 44], "confirm": [101, 102, 103], "test": [42, 43, 44]}
LOCKED_SOURCES = {
    "train.json": "1a68c789684c92280f60c0c6baee8025c702b6c76eeb8a8d1b756e5a24680944",
    "dev.json": "e0386de274ea83722d6d812597b54b3a22a24625ed0ea765649eaa4fecf9e14e",
    "scorer_sha256": "778dbb2e6e914f64a37ee5c44ecd2b2ae6709b624af99c2426c39c16418d3783",
}
DEV = {"red_triangle", "blue_square"}
TEST = {"green_circle", "yellow_triangle"}
TRAIN = {f"{c}_{s}" for c in ("red", "blue", "green", "yellow") for s in ("circle", "square", "triangle")} - DEV - TEST

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT): raise ValueError("Path must stay inside project")
    return path

def combinations(stage):
    return TRAIN | DEV if stage == "dev" else DEV if stage == "confirm" else TEST

def gate(samples, scores, stage):
    """Recompute exact per-combination proof; a saved boolean is never sufficient."""
    if stage not in SEEDS: raise ValueError("Invalid stage")
    expected = {(combo, seed) for combo in combinations(stage) for seed in SEEDS[stage]}
    actual = [(s["target"]["combo"], s["seed"]) for s in samples]
    if len(actual) != len(expected) or set(actual) != expected: raise ValueError("Incomplete or duplicate combination/seed coverage")
    details = scores["details"]
    if len(details) != len(samples): raise ValueError("Score count mismatch")
    passed = True
    for combo in combinations(stage):
        selected = [(s, d) for s, d in zip(samples, details) if s["target"]["combo"] == combo]
        correct = 0
        for sample, detail in selected:
            target = sample["target"]
            canonical = "train" if combo in TRAIN else "dev" if combo in DEV else "test"
            if target["split"] != canonical or combo != f'{target["color"]}_{target["shape"]}':
                raise ValueError("Invalid sample target")
            if (detail["image"] != sample["image"] or detail["combo"] != combo or detail["split"] != target["split"]
                    or detail["expected"] != {k: target[k] for k in ("color", "shape")}):
                raise ValueError("Score provenance mismatch")
            prediction = detail["prediction"]
            ok = prediction["scorable"] is True and all(prediction[k] == target[k] for k in ("color", "shape"))
            if detail["joint_correct"] is not ok: raise ValueError("Invalid joint score")
            correct += ok
        counter = scores["by_combo"][combo]
        accuracy = counter["joint_accuracy"]
        if (not isinstance(accuracy, (float, int)) or not math.isfinite(accuracy)
                or counter["count"] != 3 or counter["joint_correct"] != correct or accuracy != correct / 3):
            raise ValueError("Invalid finite per-combination counters")
        passed &= correct / 3 >= .8
    if set(scores["by_combo"]) != combinations(stage): raise ValueError("Unexpected score combination")
    splits = {s["target"]["split"] for s in samples}
    if set(scores["by_split"]) != splits: raise ValueError("Unexpected score split")
    for split in splits:
        rows = [d for d in details if d["split"] == split]
        counter = scores["by_split"][split]
        correct = sum(d["joint_correct"] for d in rows)
        if (counter["count"] != len(rows) or counter["joint_correct"] != correct
                or not math.isfinite(counter["joint_accuracy"]) or counter["joint_accuracy"] != correct / len(rows)):
            raise ValueError("Invalid split counters")
    return bool(passed)

def proof(path, stage, identity, sources):
    path = inside(path)
    result = read(path)
    if (result["stage"] != stage or result["seeds"] != SEEDS[stage]
            or result["identity"] != identity or result["sources"] != sources):
        raise ValueError("Previous stage/checkpoint/source mismatch")
    manifest = inside(path.parent / result["manifest"])
    if sha(manifest) != result["manifest_sha256"]: raise ValueError("Previous manifest changed")
    raw = read(manifest)
    if raw["identity"] != identity or raw["sources"] != sources: raise ValueError("Manifest provenance mismatch")
    samples = raw["samples"]
    for sample in samples:
        if sha(inside(manifest.parent / sample["image"])) != sample["image_sha256"]: raise ValueError("Previous raw sample changed")
    if not gate(samples, result["scores"], stage): raise ValueError("Previous stage gate failed")
    if stage == "confirm":
        prior = result["previous"]
        prior_path = inside(path.parent / prior["path"])
        if sha(prior_path) != prior["sha256"]: raise ValueError("Previous dev proof changed")
        proof(prior_path, "dev", identity, sources)
    return result

def montage(samples, directory):
    sheet = Image.new("RGB", (5 * 192, ((len(samples) + 4) // 5) * 96), "white")
    draw = ImageDraw.Draw(sheet)
    for i, sample in enumerate(samples):
        x, y = i % 5 * 192, i // 5 * 96
        with Image.open(directory / sample["image"]) as image: sheet.paste(image, (x, y))
        draw.text((x, y + 65), f'{sample["target"]["combo"]} seed {sample["seed"]}', fill="black")
    sheet.save(directory / "contact_sheet.png")

def run(run_dir, output_dir, stage="dev", previous=None, dataset=None, generate_fn=None, score_fn=evaluate):
    if stage not in SEEDS: raise ValueError("Invalid stage")
    run_dir, output_dir = inside(run_dir), inside(output_dir)
    dataset = inside(dataset or ROOT / "data/imagegen_shapes/v1")
    identity = {"checkpoint_sha256": sha(run_dir / "model.pt"), "config_sha256": sha(run_dir / "config.json"),
                "scorer_sha256": sha(Path(__file__).with_name("shapes_score.py"))}
    sources = {f"{split}.json": sha(dataset / f"{split}.json") for split in ("train", "dev")}
    if {**sources, "scorer_sha256": identity["scorer_sha256"]} != LOCKED_SOURCES:
        raise ValueError("Locked proposal source mismatch")
    config = read(run_dir / "config.json")
    if any(config.get(f"{split}_manifest_sha256") != sources[f"{split}.json"] for split in ("train", "dev")):
        raise ValueError("Checkpoint training source mismatch")
    # No test manifest or images are opened before both previous gates pass.
    if stage != "dev":
        if previous is None: raise ValueError("Previous stage summary required")
        proof(previous, "dev" if stage == "confirm" else "confirm", identity, sources)
    if output_dir.exists() and any(output_dir.iterdir()): raise ValueError("Refusing nonempty output directory")
    entries = []
    for split in (("train", "dev") if stage == "dev" else ("dev",) if stage == "confirm" else ("test",)):
        manifest = dataset / f"{split}.json"
        sources[manifest.name] = sha(manifest)
        entries.extend(read(manifest))
    captions = {}
    for entry in entries:
        target = {key: entry[key] for key in ("color", "shape", "combo", "split")}
        if target["combo"] != f'{target["color"]}_{target["shape"]}': raise ValueError("Invalid source target")
        key = entry["caption"]
        if key in captions and captions[key] != target: raise ValueError("Ambiguous caption")
        captions[key] = target
    if len(captions) != len(combinations(stage)) or {t["combo"] for t in captions.values()} != combinations(stage):
        raise ValueError("Unexpected caption coverage")
    if generate_fn is None:
        from lib.img import generate as generate_fn
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for caption, target in captions.items():
        for seed in SEEDS[stage]:
            paths = generate_fn([caption], out_dir=run_dir, save_to=output_dir, seed=seed)
            if len(paths) != 1: raise ValueError("Generator must return exactly one raw sample")
            image = inside(paths[0])
            if not image.is_relative_to(output_dir) or image.suffix.lower() != ".png": raise ValueError("Invalid generated image path")
            samples.append({"image": image.relative_to(output_dir).as_posix(), "image_sha256": sha(image),
                            "caption": caption, "seed": seed, "target": target})
    if identity["checkpoint_sha256"] != sha(run_dir / "model.pt") or identity["config_sha256"] != sha(run_dir / "config.json"):
        raise ValueError("Checkpoint changed during evaluation")
    manifest = output_dir / "samples.json"
    write(manifest, {"identity": identity, "sources": sources, "samples": samples})
    scores = score_fn(manifest)
    passed = gate(samples, scores, stage)
    result = {"stage": stage, "seeds": SEEDS[stage], "identity": identity, "sources": sources,
              "run_dir": run_dir.relative_to(ROOT).as_posix(), "dataset": dataset.relative_to(ROOT).as_posix(),
              "manifest": manifest.name, "manifest_sha256": sha(manifest), "scores": scores, "gate_passed": passed,
              "previous": None, "limitation": "Engineering execution is not model quality. Human inspection is required."}
    if previous:
        import os
        result["previous"] = {"path": Path(os.path.relpath(inside(previous), output_dir)).as_posix(), "sha256": sha(previous)}
    write(output_dir / "summary.json", result)
    montage(samples, output_dir)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=SEEDS, default="dev")
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({"stage": result["stage"], "gate_passed": result["gate_passed"]}))

if __name__ == "__main__": main()
