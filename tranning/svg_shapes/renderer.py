# -*- coding: utf-8 -*-
"""有限幾何 schema → 安全 SVG 的純程式 renderer（svg_shapes.v1）。

模型只負責把中文描述預測成 schema 欄位；SVG 一律由這支程式從白名單常數組出來。
模型或使用者的原始字串不會進入 SVG，所以不會出現 script、事件屬性、foreignObject
或外部 href。schema 不合法就拒絕輸出（不猜、不補值）。只用標準函式庫。
用法：python tranning/svg_shapes/renderer.py spec.json --out shape.svg
CLI 只能讀寫專案資料夾內的檔案；路徑（含解析 symlink 後）碰到 .env 一律拒絕。
"""
import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 專案根目錄（本檔在 tranning/svg_shapes/）
SCHEMA_VERSION = "svg_shapes.v1"
VIEWBOX = "0 0 300 100"
WIDTH, HEIGHT = 300, 100
SHAPES = ("circle", "square", "triangle")
COLORS = {"red": "#d62728", "blue": "#1f77b4", "green": "#2ca02c", "yellow": "#e6c200"}
POSITIONS = {"left": 50, "center": 150, "right": 250}  # 圖形中心 x；y 固定 50
SIZES = {"small": 20, "large": 40}  # 圓半徑／方形與三角形的半邊長
MAX_SHAPES = 1  # v1 只允許 1 個形狀
SPEC_KEYS = {"schema_version", "shapes"}
SHAPE_KEYS = {"shape", "color", "position", "size"}
SVG_NS = "http://www.w3.org/2000/svg"
# 稽核白名單：每種元素允許的屬性
ALLOWED = {
    "svg": {"xmlns", "viewBox", "width", "height"},
    "circle": {"cx", "cy", "r", "fill"},
    "rect": {"x", "y", "width", "height", "fill"},
    "polygon": {"points", "fill"},
}


class SpecError(ValueError):
    pass


def validate_spec(spec):
    """回傳錯誤清單；空清單才可以 render。"""
    if not isinstance(spec, dict):
        return ["$: spec must be an object"]
    errors = [f"$: missing key '{k}'" for k in sorted(SPEC_KEYS - spec.keys())]
    errors += [f"$: unknown key '{k}'" for k in sorted(spec.keys() - SPEC_KEYS)]
    if errors:
        return errors
    if spec["schema_version"] != SCHEMA_VERSION:
        errors.append(f"$.schema_version: must be '{SCHEMA_VERSION}'")
    shapes = spec["shapes"]
    if not isinstance(shapes, list) or not 1 <= len(shapes) <= MAX_SHAPES:
        return errors + [f"$.shapes: must be a list with 1 to {MAX_SHAPES} items"]
    for i, s in enumerate(shapes):
        where = f"$.shapes[{i}]"
        if not isinstance(s, dict):
            errors.append(f"{where}: must be an object")
            continue
        errors += [f"{where}: missing key '{k}'" for k in sorted(SHAPE_KEYS - s.keys())]
        errors += [f"{where}: unknown key '{k}'" for k in sorted(s.keys() - SHAPE_KEYS)]
        for key, allowed in (("shape", SHAPES), ("color", COLORS), ("position", POSITIONS), ("size", SIZES)):
            # 先確認是字串，避免 list/dict 等不可雜湊型別或 bool 混進來
            if key in s and not (isinstance(s[key], str) and s[key] in allowed):
                errors.append(f"{where}.{key}: must be one of {list(allowed)}")
    return errors


def _element(s):
    # 所有數值與顏色都來自白名單常數，不使用輸入字串本身
    cx, cy, r, fill = POSITIONS[s["position"]], 50, SIZES[s["size"]], COLORS[s["color"]]
    if s["shape"] == "circle":
        return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>'
    if s["shape"] == "square":
        return f'<rect x="{cx - r}" y="{cy - r}" width="{2 * r}" height="{2 * r}" fill="{fill}"/>'
    return f'<polygon points="{cx},{cy - r} {cx - r},{cy + r} {cx + r},{cy + r}" fill="{fill}"/>'


def render_svg(spec):
    """合法 spec → SVG 字串；不合法就丟 SpecError（不輸出半成品）。"""
    errors = validate_spec(spec)
    if errors:
        raise SpecError("; ".join(errors))
    body = "".join(_element(s) for s in spec["shapes"])
    svg = (f'<svg xmlns="{SVG_NS}" viewBox="{VIEWBOX}" width="{WIDTH}" height="{HEIGHT}">'
           f"{body}</svg>\n")
    problems = audit_svg(svg)
    if problems:  # 理論上不會發生；發生就是 renderer 本身有 bug，拒絕輸出
        raise SpecError("internal audit failed: " + "; ".join(problems))
    return svg


def audit_svg(text):
    """白名單稽核：只允許 svg/circle/rect/polygon 與固定屬性，座標必須在 viewBox 內。
    先拒絕 DOCTYPE/ENTITY 再解析，避免實體展開攻擊。"""
    if not isinstance(text, str):
        return ["svg must be a string"]
    low = text.lower()
    if "<!doctype" in low or "<!entity" in low or "<?xml-stylesheet" in low:
        return ["DOCTYPE, ENTITY and stylesheet instructions are not allowed"]
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        return [f"not well-formed XML: {exc}"]
    problems = []
    if root.tag != f"{{{SVG_NS}}}svg":
        problems.append("root element must be svg")
    if root.get("width") != str(WIDTH) or root.get("height") != str(HEIGHT):
        problems.append(f"svg width and height must be {WIDTH} and {HEIGHT}")
    # 根元素底下恰好 MAX_SHAPES 個直接子元素，且形狀元素不可再有子元素
    if len(root) != MAX_SHAPES:
        problems.append(f"svg must contain exactly {MAX_SHAPES} shape element(s)")
    for child in root:
        if len(child):
            problems.append("shape elements must not have child elements")
    for el in root.iter():
        ns, _, tag = el.tag.rpartition("}")
        if ns.lstrip("{") != SVG_NS or tag not in ALLOWED:
            problems.append(f"element '{el.tag}' is not allowed")
            continue
        for attr in el.attrib:
            if attr not in ALLOWED[tag]:
                problems.append(f"attribute '{attr}' on '{tag}' is not allowed")
        if (el.text or "").strip() or (el.tail or "").strip():
            problems.append(f"text content in '{tag}' is not allowed")
        if tag == "svg":
            if el is not root:
                problems.append("nested svg is not allowed")
            if el.get("viewBox") != VIEWBOX:
                problems.append(f"viewBox must be '{VIEWBOX}'")
            continue
        fill = el.get("fill")
        if fill not in COLORS.values():
            problems.append(f"fill on '{tag}' must be a palette color")
        problems += _bounds(tag, el)
    return problems


def _bounds(tag, el):
    try:
        if tag == "circle":
            cx, cy, r = (float(el.get(k)) for k in ("cx", "cy", "r"))
            box = (cx - r, cy - r, cx + r, cy + r)
        elif tag == "rect":
            x, y, w, h = (float(el.get(k)) for k in ("x", "y", "width", "height"))
            box = (x, y, x + w, y + h)
        else:
            pts = [tuple(float(v) for v in p.split(",")) for p in el.get("points", "").split()]
            if len(pts) != 3 or any(len(p) != 2 for p in pts):
                return ["polygon must have exactly 3 points"]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            box = (min(xs), min(ys), max(xs), max(ys))
    except (TypeError, ValueError):
        return [f"'{tag}' has missing or non-numeric geometry"]
    # 明確要求有限數值：min()/max() 遇到 nan 會略過它，不能只靠範圍比較擋下
    values = list(box) + ([v for p in pts for v in p] if tag == "polygon" else [])
    if not all(math.isfinite(v) for v in values):
        return [f"'{tag}' geometry must be finite numbers"]
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= WIDTH and 0 <= y1 < y2 <= HEIGHT):
        return [f"'{tag}' is outside the viewBox"]
    return []


def _no_dup_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise SpecError(f"duplicate key: {k}")
        out[k] = v
    return out


def _bad_constant(name):
    raise SpecError(f"non-finite number is not allowed: {name}")


def safe_project_path(raw, suffix):
    """回傳 (解析後路徑, 錯誤)。必須在 ROOT 內、副檔名正確（原始與解析後都要），
    原始路徑或解析後路徑的任何一層以 .env 開頭都拒絕（涵蓋 symlink 指向 .env）。"""
    p = Path(raw)
    try:
        resolved = p.resolve()
        root = Path(ROOT).resolve()
    except (OSError, RuntimeError) as exc:
        return None, f"cannot resolve path: {exc}"
    if not resolved.is_relative_to(root):
        return None, "must stay inside the project folder"
    parts = [x.lower() for x in p.parts] + [x.lower() for x in resolved.relative_to(root).parts]
    if any(x.startswith(".env") for x in parts):
        return None, "must not be or be inside a .env path"
    if p.suffix.lower() != suffix or resolved.suffix.lower() != suffix:
        return None, f"must be a {suffix} file"
    return resolved, None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render a svg_shapes.v1 spec JSON to a safe SVG file.")
    ap.add_argument("spec", help="Spec JSON file to read")
    ap.add_argument("--out", required=True, help="New .svg file to write (never overwrites)")
    args = ap.parse_args(argv)

    def report(ok, errors, code, out=None):
        print(json.dumps({"ok": ok, "errors": errors, "out": out}, indent=2))
        return code

    spec_path, problem = safe_project_path(args.spec, ".json")
    if problem:
        return report(False, [f"spec: {problem}"], 2)
    out_path, problem = safe_project_path(args.out, ".svg")
    if problem:
        return report(False, [f"out: {problem}"], 2)
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f, object_pairs_hook=_no_dup_keys, parse_constant=_bad_constant)
    except (OSError, ValueError) as exc:
        return report(False, [f"spec: cannot read: {exc}"], 2)
    errors = validate_spec(spec)
    if errors:
        return report(False, errors, 1)
    svg = render_svg(spec)
    try:
        with open(out_path, "x", encoding="utf-8") as f:  # "x"：檔案已存在就失敗，不覆寫
            f.write(svg)
    except FileExistsError:
        return report(False, ["out: file already exists; refusing to overwrite"], 2)
    except OSError as exc:
        return report(False, [f"out: cannot write: {exc}"], 2)
    return report(True, [], 0, str(out_path))


if __name__ == "__main__":
    sys.exit(main())
