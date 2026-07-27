"""反事实套件的命令行入口。

    python eval/run_counterfactual.py                    # 全部
    python eval/run_counterfactual.py --tag core         # 只跑核心用例
    python eval/run_counterfactual.py --case want_you__partner_vs_stranger -v
    python eval/run_counterfactual.py --json out.json    # 给 CI / UI 用

调参时的典型用法：改 config 或 AppraisalParams → 跑这个 → 看方向正确率有没有掉。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from affect.appraisal import RelationalAppraisal  # noqa: E402
from affect.counterfactual import (  # noqa: E402
    CASES_DIR,
    load_cases,
    run_case,
    summarize,
)

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="反事实成对测试")
    ap.add_argument("--cases", default=str(CASES_DIR), help="YAML 文件或目录")
    ap.add_argument("--tag", action="append", default=None, help="只跑带该标签的用例")
    ap.add_argument("--case", action="append", default=None, help="只跑指定 id")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印每条断言")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    cases = load_cases(args.cases)
    if args.tag:
        cases = [c for c in cases if set(c.tags) & set(args.tag)]
    if args.case:
        cases = [c for c in cases if c.id in set(args.case)]
    if not cases:
        print("没有匹配的用例", file=sys.stderr)
        return 2

    engine = RelationalAppraisal()
    results = [run_case(c, engine) for c in cases]

    for r in results:
        mark = f"{GREEN}✓{RESET}" if r.passed else f"{RED}✗{RESET}"
        title = f"{r.id}" + (f"  「{r.utterance}」" if r.utterance else "")
        print(f"{mark} {title}")
        for a in r.assertions:
            if a.ok and not args.verbose:
                continue
            m = f"{GREEN}✓{RESET}" if a.ok else f"{RED}✗{RESET}"
            print(f"    {m} {a.text}\n      {DIM}{a.detail}{RESET}")
        if not r.passed or args.verbose:
            for side in ("a", "b"):
                vals = " ".join(f"{k[:4]}={v:+.2f}" for k, v in r.states[side].items())
                print(f"    {DIM}{side}: {vals}{RESET}")

    s = summarize(results)
    ok = s["cases_passed"] == s["cases"]
    colour = GREEN if ok else RED
    print(
        f"\n{colour}用例 {s['cases_passed']}/{s['cases']}  "
        f"断言 {s['assertions'] - s['assertions_failed']}/{s['assertions']}  "
        f"方向正确率 {s['direction_accuracy']}{RESET}"
    )
    if not ok:
        print("\n失败的用例：")
        for f in s["failures"]:
            print(f"  {f['id']}: {', '.join(f['failed'])}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {"summary": s, "results": [r.to_dict() for r in results]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"报告 → {args.json_out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
