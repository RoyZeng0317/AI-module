# -*- coding: utf-8 -*-
"""renderer.py 的工程測試：只測 schema 驗證、SVG 安全與 CLI，不涉及模型或生成品質。
執行：python -m unittest tranning/svg_shapes/test_renderer.py
"""
import contextlib
import io
import itertools
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import renderer as r  # noqa: E402


def spec(shape="circle", color="red", position="center", size="small"):
    return {"schema_version": r.SCHEMA_VERSION,
            "shapes": [{"shape": shape, "color": color, "position": position, "size": size}]}


class RendererTest(unittest.TestCase):
    def test_all_72_combinations_render_safely(self):
        outputs = set()
        for combo in itertools.product(r.SHAPES, r.COLORS, r.POSITIONS, r.SIZES):
            svg = r.render_svg(spec(*combo))
            self.assertEqual(r.audit_svg(svg), [], combo)
            self.assertEqual(svg, r.render_svg(spec(*combo)))  # 相同輸入得到相同輸出
            outputs.add(svg)
        self.assertEqual(len(outputs), 72)  # 每個組合畫出來都不同

    def test_shape_to_element(self):
        self.assertIn("<circle ", r.render_svg(spec("circle")))
        self.assertIn("<rect ", r.render_svg(spec("square")))
        self.assertIn("<polygon ", r.render_svg(spec("triangle")))
        svg = r.render_svg(spec("circle", "blue", "left", "large"))
        self.assertIn('cx="50" cy="50" r="40" fill="#1f77b4"', svg)
        self.assertIn('viewBox="0 0 300 100"', svg)

    def test_invalid_specs_are_rejected(self):
        bad = [
            None, [], {"shapes": []},
            dict(spec(), extra=1),
            dict(spec(), schema_version="svg_shapes.v0"),
            {"schema_version": r.SCHEMA_VERSION, "shapes": []},
            {"schema_version": r.SCHEMA_VERSION, "shapes": spec()["shapes"] * 2},  # v1 只允許 1 個
            {"schema_version": r.SCHEMA_VERSION, "shapes": ["circle"]},
        ]
        for field, value in (("shape", "star"), ("color", "purple"), ("position", "top"), ("size", "medium"),
                             ("color", True), ("size", ["small"]), ("shape", {"a": 1}),
                             ("color", '"/><script>alert(1)</script>'), ("color", "#000000")):
            s = spec(); s["shapes"][0][field] = value
            bad.append(s)
        s = spec(); s["shapes"][0]["href"] = "https://example.com"
        bad.append(s)
        s = spec(); del s["shapes"][0]["size"]
        bad.append(s)
        for b in bad:
            self.assertTrue(r.validate_spec(b), b)
            with self.assertRaises(r.SpecError):
                r.render_svg(b)

    def test_audit_rejects_unsafe_svg(self):
        head = f'<svg xmlns="{r.SVG_NS}" viewBox="{r.VIEWBOX}" width="300" height="100">'
        ok_circle = '<circle cx="150" cy="50" r="20" fill="#d62728"/>'
        bad = [
            head + "<script>alert(1)</script></svg>",
            head + '<circle cx="150" cy="50" r="20" fill="#d62728" onclick="x()"/></svg>',
            head + "<foreignObject><div/></foreignObject></svg>",
            head + '<image href="https://example.com/a.png"/></svg>',
            head + f'<circle xmlns:xl="http://www.w3.org/1999/xlink" xl:href="#a" cx="150" cy="50" r="20" fill="#d62728"/></svg>',
            head + '<circle cx="150" cy="50" r="20" fill="#d62728" style="fill:url(#x)"/></svg>',
            head + '<circle cx="150" cy="50" r="20" fill="url(https://e.com/x)"/></svg>',
            head + '<circle cx="290" cy="50" r="20" fill="#d62728"/></svg>',  # 超出 viewBox
            head + ok_circle + "hello</svg>",
            f'<svg xmlns="{r.SVG_NS}" viewBox="0 0 900 900">{ok_circle}</svg>',
            '<!DOCTYPE svg [<!ENTITY a "x">]>' + head + ok_circle + "</svg>",
            head + f"<svg>{ok_circle}</svg></svg>",
            f'<svg viewBox="{r.VIEWBOX}">{ok_circle}</svg>',  # 缺少 SVG namespace
            "<svg", 123,
            head.replace('width="300"', 'width="999"') + ok_circle + "</svg>",
            head.replace(' height="100"', "") + ok_circle + "</svg>",
            head + "</svg>",  # 沒有形狀
            head + ok_circle + ok_circle.replace('cx="150"', 'cx="50"') + "</svg>",  # v1 只能 1 個
            head + ok_circle.replace("/>", ">") + ok_circle + "</circle></svg>",  # 巢狀形狀
        ]
        # 非有限數值：舊版對 polygon 的 nan 會漏擋（min/max 會略過 nan），這裡逐一確認
        for el in ('<circle cx="nan" cy="50" r="20" fill="#d62728"/>',
                   '<circle cx="150" cy="50" r="inf" fill="#d62728"/>',
                   '<rect x="10" y="10" width="NaN" height="20" fill="#d62728"/>',
                   '<polygon points="150,10 nan,90 190,90" fill="#d62728"/>',
                   '<polygon points="150,10 110,90 inf,90" fill="#d62728"/>'):
            bad.append(head + el + "</svg>")
        for text in bad:
            self.assertTrue(r.audit_svg(text), text)
        self.assertEqual(r.audit_svg(head + ok_circle + "</svg>"), [])

    def test_output_contains_only_whitelisted_text(self):
        for combo in itertools.product(r.SHAPES, r.COLORS, r.POSITIONS, r.SIZES):
            svg = r.render_svg(spec(*combo)).lower()
            self.assertIsNone(re.search(r"\son\w+=|script|href|foreignobject|style|<!", svg), combo)

    def run_cli(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = r.main([str(a) for a in args])
        text = buf.getvalue()
        self.assertTrue(text.isascii())  # CLI 輸出固定英文
        return code, json.loads(text)

    def setUp(self):
        self._root = r.ROOT

    def tearDown(self):
        r.ROOT = self._root

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d).resolve()
            r.ROOT = d  # 測試時把專案根目錄指到暫存資料夾
            good, bad, nan = d / "good.json", d / "bad.json", d / "nan.json"
            good.write_text(json.dumps(spec()), encoding="utf-8")
            bad.write_text(json.dumps(spec(color="purple")), encoding="utf-8")
            nan.write_text('{"schema_version": NaN}', encoding="utf-8")
            code, out = self.run_cli(good, "--out", d / "a.svg")
            self.assertEqual((code, out["ok"]), (0, True))
            self.assertEqual(r.audit_svg((d / "a.svg").read_text(encoding="utf-8")), [])
            self.assertEqual(self.run_cli(good, "--out", d / "a.svg")[0], 2)  # 不覆寫
            self.assertEqual(self.run_cli(bad, "--out", d / "b.svg")[0], 1)
            self.assertFalse((d / "b.svg").exists())  # 不合法不輸出
            self.assertEqual(self.run_cli(nan, "--out", d / "c.svg")[0], 2)
            self.assertEqual(self.run_cli(good, "--out", d / "c.html")[0], 2)
            self.assertEqual(self.run_cli(d / ".env", "--out", d / "d.svg")[0], 2)
            self.assertEqual(self.run_cli(d / "missing.json", "--out", d / "e.svg")[0], 2)

    def test_cli_paths_stay_in_project(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as outside:
            d, outside = Path(d).resolve(), Path(outside).resolve()
            r.ROOT = d
            good = d / "good.json"
            good.write_text(json.dumps(spec()), encoding="utf-8")
            (outside / "o.json").write_text(json.dumps(spec()), encoding="utf-8")
            (d / ".env").mkdir()
            (d / ".env" / "s.json").write_text(json.dumps(spec()), encoding="utf-8")
            (d / "sub").mkdir()
            rejected = [
                (outside / "o.json", d / "a.svg"),            # 讀專案外
                (good, outside / "a.svg"),                     # 寫專案外
                (good, d / "sub" / ".." / ".." / "x.svg"),     # 用 .. 跳出
                (d / ".env" / "s.json", d / "b.svg"),          # .env 祖先資料夾
                (good, d / ".env" / "c.svg"),
            ]
            for spec_path, out_path in rejected:
                code, out = self.run_cli(spec_path, "--out", out_path)
                self.assertEqual(code, 2, (spec_path, out_path))
            self.assertFalse((outside / "a.svg").exists())
            self.assertFalse((d.parent / "x.svg").exists())
            try:
                (d / "link_out.json").symlink_to(outside / "o.json")
                (d / "link_env.json").symlink_to(d / ".env" / "s.json")
                (d / "outdir").symlink_to(outside, target_is_directory=True)
                (d / "dangling.svg").symlink_to(outside / "new.svg")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are not available here")
            for spec_path, out_path in ((d / "link_out.json", d / "d.svg"),   # symlink 指到專案外
                                        (d / "link_env.json", d / "e.svg"),   # symlink 指到 .env
                                        (good, d / "outdir" / "f.svg"),       # 經 symlink 資料夾寫到專案外
                                        (good, d / "dangling.svg")):          # 懸空 symlink 指到專案外
                self.assertEqual(self.run_cli(spec_path, "--out", out_path)[0], 2, (spec_path, out_path))
            self.assertEqual(sorted(p.name for p in outside.iterdir()), ["o.json"])  # 專案外沒有新檔


if __name__ == "__main__":
    unittest.main()
