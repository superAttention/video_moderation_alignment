import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Print the base → SFT → DPO comparison table from saved eval JSON files.

Usage:
    python scripts/compare_evals.py
    python scripts/compare_evals.py --checkpoints base sft dpo
    python scripts/compare_evals.py --checkpoints base sft dpo --out outputs/summary.md

Reads:  outputs/eval_{checkpoint}.json  for each checkpoint
Prints: one table per split (harmful / benign) showing how each metric
        evolves across the training pipeline.

The two splits tell different parts of the story:
    harmful split  →  safety   (does it correctly refuse harmful requests?)
    benign  split  →  utility  (does it escalate instead of over-refusing?)
"""
import json
import argparse
from pathlib import Path

DEFAULT_ORDER = ["base_freeform", "base", "sft", "dpo"]
METRICS = [
    ("valid_json_rate",     "valid_json",   "{:.1%}"),
    ("action_accuracy",     "action_acc",   "{:.1%}"),
    ("category_accuracy",   "category_acc", "{:.1%}"),
    ("avg_rationale_score", "rationale",    "{:.2f}"),
]

# Derived metric: only meaningful on the benign split
# over_refusal_rate = 1 - benign_action_accuracy
BENIGN_DERIVED = [
    ("over_refusal_rate", "over_refusal", "{:.1%}"),
]


def load_results(checkpoints: list[str]) -> dict[str, dict]:
    loaded = {}
    for ck in checkpoints:
        path = Path("outputs") / f"eval_{ck}.json"
        if not path.exists():
            print(f"  [warn] {path} not found — skipping {ck}")
            continue
        data = json.loads(path.read_text())
        loaded[ck] = data["results"]
    return loaded


def fmt(value, fmt_str: str) -> str:
    if value is None:
        return "N/A"
    try:
        return fmt_str.format(value)
    except (ValueError, TypeError):
        return "N/A"


def print_table(
    split: str,
    checkpoints: list[str],
    results: dict[str, dict],
    out_lines: list[str],
) -> None:
    def emit(line: str = "") -> None:
        print(line)
        out_lines.append(line)

    emit(f"\n{'═' * 64}")
    emit(f"  {split.upper()} SPLIT  "
         f"({'safety: should REFUSE' if split == 'harmful' else 'utility: should ESCALATE'})")
    emit(f"{'═' * 64}")

    # Header
    col_w = 16
    header = f"  {'metric':<22}" + "".join(f"{ck:>{col_w}}" for ck in checkpoints)
    emit(header)
    emit("  " + "─" * (22 + col_w * len(checkpoints)))

    for key, label, fmt_str in METRICS:
        row = f"  {label:<22}"
        for ck in checkpoints:
            if ck not in results:
                row += f"{'—':>{col_w}}"
                continue
            split_data = results[ck].get(split, {})
            value = split_data.get("overall", {}).get(key)
            row += f"{fmt(value, fmt_str):>{col_w}}"
        emit(row)

    if split == "benign":
        for key, label, fmt_str in BENIGN_DERIVED:
            row = f"  {label:<22}"
            for ck in checkpoints:
                if ck not in results:
                    row += f"{'—':>{col_w}}"
                    continue
                action_acc = results[ck].get(split, {}).get("overall", {}).get("action_accuracy")
                value = 1.0 - action_acc if action_acc is not None else None
                row += f"{fmt(value, fmt_str):>{col_w}}"
            emit(row)

    # Per-category breakdown for action_accuracy
    emit("")
    emit("  Action accuracy by category:")
    all_cats: set[str] = set()
    for ck in checkpoints:
        if ck in results:
            all_cats |= set(results[ck].get(split, {}).get("by_category", {}).keys())

    for cat in sorted(all_cats):
        row = f"    {cat:<20}"
        for ck in checkpoints:
            if ck not in results:
                row += f"{'—':>{col_w}}"
                continue
            cat_data = results[ck].get(split, {}).get("by_category", {}).get(cat, {})
            value = cat_data.get("action_accuracy")
            row += f"{fmt(value, '{:.1%}'):>{col_w}}"
        emit(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_ORDER,
                        help="Checkpoint labels to compare, in order")
    parser.add_argument("--out", default="",
                        help="Optional path to write a markdown-compatible summary")
    args = parser.parse_args()

    results = load_results(args.checkpoints)
    if not results:
        print("No eval files found. Run scripts/evaluate.py for each checkpoint first.")
        return

    present = [ck for ck in args.checkpoints if ck in results]
    out_lines: list[str] = []

    for split in ["harmful", "benign"]:
        print_table(split, present, results, out_lines)

    # Sample counts
    print()
    out_lines.append("")
    for ck in present:
        for split in ["harmful", "benign"]:
            n = results[ck].get(split, {}).get("overall", {}).get("n", "?")
            print(f"  {ck} / {split}: n={n}")
            out_lines.append(f"  {ck} / {split}: n={n}")

    if args.out:
        Path(args.out).write_text("\n".join(out_lines))
        print(f"\nSummary written → {args.out}")


if __name__ == "__main__":
    main()
