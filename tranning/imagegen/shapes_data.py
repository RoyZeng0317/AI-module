"""Deterministic teaching images, never model output. No training is performed."""
import argparse, hashlib, json, random
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 170, 65), "yellow": (235, 195, 30)}
ZH = {"red": "紅色", "blue": "藍色", "green": "綠色", "yellow": "黃色", "circle": "圓形", "square": "正方形", "triangle": "三角形"}
HELD = {"dev": {"red_triangle", "blue_square"}, "test": {"green_circle", "yellow_triangle"}}

def caption(color, shape):
    return f"{ZH[color]}{ZH[shape]}"

def build(output, variants=30, seed=42):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must stay inside this project")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty dataset")
    if variants < 1 or variants > 500:
        raise ValueError("variants must be between 1 and 500")
    rng, rows, hashes = random.Random(seed), {s: [] for s in ("train", "dev", "test")}, set()
    output.mkdir(parents=True, exist_ok=True)
    for color in COLORS:
        for shape in ("circle", "square", "triangle"):
            combo = f"{color}_{shape}"
            split = next((s for s, groups in HELD.items() if combo in groups), "train")
            for index in range(variants):
                while True:
                    size, x, y = rng.randint(24, 40), rng.randint(8, 15), rng.randint(8, 15)
                    im = Image.new("RGB", (64, 64), "white")
                    draw, box = ImageDraw.Draw(im), (x, y, x + size, y + size)
                    if shape == "circle": draw.ellipse(box, fill=COLORS[color])
                    elif shape == "square": draw.rectangle(box, fill=COLORS[color])
                    else: draw.polygon([(x + size // 2, y), (x, y + size), (x + size, y + size)], fill=COLORS[color])
                    digest = hashlib.sha256(im.tobytes()).hexdigest()
                    if digest not in hashes: break
                hashes.add(digest)
                relative = f"images/{split}/{combo}_{index:03d}.png"
                dest = output / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                im.save(dest)
                rows[split].append({"image": relative, "caption": caption(color, shape), "color": color, "shape": shape,
                                    "split": split, "combo": combo, "pixel_sha256": digest,
                                    "file_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                                    "source": "PIL synthetic teaching image", "box": list(box)})
    for split, entries in rows.items():
        (output / f"{split}.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    spec = {"version": 1, "seed": seed, "size": 64, "variants_per_combo": variants,
            "counts": {s: len(r) for s, r in rows.items()}, "unique_pixel_hashes": len(hashes),
            "combinations": {s: sorted({e['combo'] for e in r}) for s, r in rows.items()},
            "source": "Algorithmically synthesized teaching images, NOT model generations",
            "test_policy": "Do not train on test; first real model evaluation is sealed after reporting",
            "manifests_sha256": {s: hashlib.sha256((output / f'{s}.json').read_bytes()).hexdigest() for s in rows}}
    (output / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return spec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variants", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.variants, args.seed), ensure_ascii=False, indent=2))
