"""Record chat outputs for human review; never infer a semantic score.

Inference requires --run. Offline mode disables the tool router and skips
network cases. No training, checkpoint writes, or external model selection.
"""
import argparse, json, time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch


def evaluate_cases(cases: list[dict], engine, allow_network: bool = False):
    """Yield reproducible review records; engine is injectable for cheap tests."""
    for case in cases:
        record = dict(case, semantic_review="pending_human_review", semantic_score=None)
        if case.get("requires_network") and not allow_network:
            yield dict(record, status="skipped_network", latency_ms=None, route=None, reply=None)
            continue
        started = time.perf_counter()
        # Even unmarked prompts may otherwise trigger the router; disable it
        # entirely in offline mode, without changing product code or .env.
        context = nullcontext() if allow_network else patch.object(engine, "route_reply", return_value=None)
        try:
            with context:
                route, reply = engine.smart_reply_traced(case["prompt"], force_mode="auto", history=case.get("history"))
            result = dict(status="recorded", route=route, reply=reply)
        except Exception as exc:
            result = dict(status="error", route=None, reply=None, error=f"{type(exc).__name__}: {exc}")
        yield dict(record, **result, latency_ms=round((time.perf_counter() - started) * 1000, 2),
                   tools_enabled=allow_network)


def main(argv=None, engine=None):
    parser = argparse.ArgumentParser(description="Record outputs for manual semantic review; no automatic quality score")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--cases", type=Path, default=root / "data/chat_eval_cases.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="explicitly start local model inference")
    parser.add_argument("--allow-network", action="store_true", help="enable tools and network cases; default is offline")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("Add --run to explicitly start inference. No evaluation was run.")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    required = {"id", "prompt", "language", "expected_intent", "review_criteria"}
    if not isinstance(cases, list) or any(not isinstance(case, dict) or not required <= case.keys() for case in cases):
        parser.error("Cases must be a list with id, prompt, language, expected_intent and review_criteria.")
    if len({case["id"] for case in cases}) != len(cases):
        parser.error("Case ids must be unique.")
    if engine is None:
        import chats as engine
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in evaluate_cases(cases, engine, args.allow_network):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
    print(f"Saved human-review records to {args.output}. No semantic score was calculated.")


if __name__ == "__main__":
    main()
