"""Transformer 多輪上下文驗收（to-do #50）：一次跑完評估並依預先宣告的門檻判定。

跑的項目（全部是模型原始生成，greedy 解碼，不經過 smart_reply 或任何模板代答）：
  1. v2 test（context_dialogues_v2.py 產生，字形分層、實體與句型都沒在訓練出現）
     ＋反事實檢查：拿掉歷史要回缺資訊、換成別段歷史答案要跟著變
  2. v1 test（舊凍結測試，只供跨版本對照，不列入門檻）
  3. 單輪 regression（pairs 前 20 筆）：候選模型 vs 基準模型，檢查有沒有洗掉單輪能力

門檻寫在 CRITERIA，是本檔提出的建議值，需要使用者確認；門檻要在看結果「之前」定好，
不能看完分數再調。報告寫到 --report-dir：validation_report.json＋validation_summary.md。

用法（需要已訓練的 model.pt，建議在 RTX 4060 上跑）：
    python context_validate.py --model-dir gpt_context_runs/v3 --baseline-dir gpt_chat_runs
    python context_validate.py --model-dir gpt_context_runs/v3 --limit 40   # 快速冒煙，不算正式驗收
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from context_evaluate import evaluate_file

ROOT = Path(__file__).resolve().parents[1]
DATA_V2 = ROOT / "data/context_dialogues_v2"
DATA_V1 = ROOT / "data/context_dialogues"

# 建議門檻（2026-11-17 提出，待使用者確認後才算正式）
CRITERIA = {
    "v2_macro_task_success": 0.80,      # 四類任務平均（去重後輸入）
    "v2_min_task_success": 0.60,        # 任一任務不得低於此值（v1 曾出現偏好／格式 0%）
    "v2_min_form_success": 0.50,        # 任一字形不得低於此值（v1 的切分偏差就是某字形全錯）
    "v2_missing_success": 0.80,         # 沒提過就要說缺資訊，不能亂猜
    "v2_contrastive_success": 0.70,     # 換歷史答案要跟著換，證明真的有看前文
    "v2_missing_history_success": 0.80, # 拿掉歷史要回缺資訊
    "regression_max_drop": 0.05,        # 單輪 exact 比基準最多掉 5 個百分點
}


def breakdown(rows, key):
    """依某欄位（task/kind/entity_form）統計 task_success。缺欄位的列略過。"""
    groups = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None:
            groups[value].append(row["task_success"])
    return {value: {"count": len(items), "task_success": sum(items) / len(items)} for value, items in sorted(groups.items())}


def judge(v2_result, regression=None, baseline_regression=None, criteria=CRITERIA):
    """把評估結果套上門檻；回傳每項的 value/threshold/passed 與總結。可用假資料測試。"""
    summary = v2_result["unique_input_summary"]
    rows = v2_result["rows"]
    by_form = breakdown([r for r in rows if r.get("kind") in ("recall", "corrected")], "entity_form")
    missing_rows = [r for r in rows if r.get("kind", "").startswith("missing")]
    checks = {
        "v2_macro_task_success": (summary["macro_task_success"], ">="),
        "v2_min_task_success": (min(t["task_success"] for t in summary["by_task"].values()), ">="),
        "v2_min_form_success": (min((f["task_success"] for f in by_form.values()), default=0.0), ">="),
        "v2_missing_success": (sum(r["task_success"] for r in missing_rows) / max(1, len(missing_rows)), ">="),
        "v2_contrastive_success": (v2_result["summary"]["contrastive_success"], ">="),
        "v2_missing_history_success": (v2_result["summary"]["missing_history_success"], ">="),
    }
    result = {}
    for name, (value, op) in checks.items():
        result[name] = {"value": value, "threshold": criteria[name], "passed": value >= criteria[name]}
    if regression is not None and baseline_regression is not None:
        cand = regression["summary"]["macro_exact"]
        base = baseline_regression["summary"]["macro_exact"]
        result["regression_max_drop"] = {"value": base - cand, "candidate": cand, "baseline": base,
                                         "threshold": criteria["regression_max_drop"],
                                         "passed": base - cand <= criteria["regression_max_drop"]}
    else:
        result["regression_max_drop"] = {"value": None, "threshold": criteria["regression_max_drop"], "passed": None,
                                         "note": "沒有提供 --baseline-dir，無法判定單輪是否退步"}
    decided = [c["passed"] for c in result.values() if c["passed"] is not None]
    return {"checks": result, "by_form": by_form, "by_kind": breakdown(rows, "kind"),
            "passed": all(decided) and result["regression_max_drop"]["passed"] is not None}


def _pct(value):
    return "—" if value is None else f"{value:.1%}"


def format_summary(verdict, model_dir, v1_summary=None, limit=None):
    lines = [f"# 多輪上下文驗收：{Path(model_dir).name}", ""]
    if limit:
        lines += [f"⚠️ 只跑前 {limit} 筆，是冒煙測試，不算正式驗收。", ""]
    lines += ["| 項目 | 結果 | 門檻 | 通過 |", "|---|---|---|---|"]
    for name, check in verdict["checks"].items():
        mark = {True: "✅", False: "❌", None: "未判定"}[check["passed"]]
        op = "≤" if name == "regression_max_drop" else "≥"
        lines.append(f"| {name} | {_pct(check['value'])} | {op} {check['threshold']:.0%} | {mark} |")
    lines += ["", "## 各字形（recall＋corrected）", "| 字形 | 筆數 | 正確率 |", "|---|---|---|"]
    lines += [f"| {k} | {v['count']} | {v['task_success']:.1%} |" for k, v in verdict["by_form"].items()]
    lines += ["", "## 各類型", "| 類型 | 筆數 | 正確率 |", "|---|---|---|"]
    lines += [f"| {k} | {v['count']} | {v['task_success']:.1%} |" for k, v in verdict["by_kind"].items()]
    if v1_summary:
        lines += ["", f"v1 舊測試（只供對照）：macro task_success {v1_summary['macro_task_success']:.1%}"]
    lines += ["", f"**總判定：{'通過' if verdict['passed'] else '未通過'}**",
              "", "門檻為建議值，只代表模板式多輪記憶任務，不代表一般對話理解。"]
    return "\n".join(lines)


def run_validation(model_dir, report_dir, baseline_dir=None, data_v2=DATA_V2, data_v1=DATA_V1,
                   device=None, limit=None, seed=42):
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    v2 = evaluate_file(Path(data_v2) / "test.json", model_dir, report_dir / "v2_test.json", device, seed, limit)
    v1 = None
    if (Path(data_v1) / "test.json").exists():
        v1 = evaluate_file(Path(data_v1) / "test.json", model_dir, report_dir / "v1_test.json", device, seed, limit)
    regression = evaluate_file(Path(data_v2) / "regression.json", model_dir, report_dir / "regression.json",
                               device, seed, None, counterfactual=False, legacy=True)
    baseline = None
    if baseline_dir:
        baseline = evaluate_file(Path(data_v2) / "regression.json", Path(baseline_dir),
                                 report_dir / "regression_baseline.json", device, seed, None,
                                 counterfactual=False, legacy=True)
    verdict = judge(v2, regression, baseline)
    verdict.update(model_dir=str(Path(model_dir).resolve()), baseline_dir=str(baseline_dir) if baseline_dir else None,
                   limit=limit, criteria=CRITERIA, v1_summary=v1["unique_input_summary"] if v1 else None)
    if limit:
        verdict["passed"] = False
        verdict["note"] = "limit 模式只跑部分資料，不算正式驗收"
    (report_dir / "validation_report.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = format_summary(verdict, model_dir, verdict["v1_summary"], limit)
    (report_dir / "validation_summary.md").write_text(summary, encoding="utf-8")
    return verdict, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True, help="要驗收的多輪 checkpoint")
    parser.add_argument("--baseline-dir", type=Path, help="單輪對照用的原正式權重（例如 gpt_chat_runs）")
    parser.add_argument("--report-dir", type=Path, help="預設 output/context_validation/<模型資料夾名>")
    parser.add_argument("--data-v2", type=Path, default=DATA_V2)
    parser.add_argument("--data-v1", type=Path, default=DATA_V1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, help="只跑前 N 筆（冒煙用，判定一律未通過）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report_dir = args.report_dir or ROOT / "output/context_validation" / args.model_dir.resolve().name
    _, summary = run_validation(args.model_dir, report_dir, args.baseline_dir, args.data_v2, args.data_v1,
                                args.device, args.limit, args.seed)
    print(summary)
    print(f"\n報告：{report_dir}")


if __name__ == "__main__":
    main()
