import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Stratified 80/10/10 train/val/test split of generated structured data.

Stratification key: (category, split)
Ensures all 6 strata (3 categories × 2 splits: harmful/benign) are
represented in every fold — prevents any category from being unseen at val/test.

Input:  data/generated_structured.jsonl
Output: data/train.jsonl, data/val.jsonl, data/test.jsonl

The test set is locked — never used during training or DPO pair generation.
Run once after generate_structured_responses.py completes.

    python scripts/split_data.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path


TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
# TEST_FRAC  = 0.10  (remainder)
SEED = 42


def split_stratum(examples: list, train_f: float, val_f: float, rng: random.Random):
    """Split a single stratum into train/val/test."""
    rng.shuffle(examples)
    n = len(examples)
    n_train = max(1, round(n * train_f))
    n_val   = max(1, round(n * val_f))
    # Ensure test gets at least 1 if stratum is large enough
    if n >= 3:
        n_val = min(n_val, n - n_train - 1)
    else:
        n_val = max(0, n - n_train)
    return (
        examples[:n_train],
        examples[n_train:n_train + n_val],
        examples[n_train + n_val:],
    )


def main():
    in_path = Path("data/generated_structured.jsonl")
    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found. Run scripts/generate_structured_responses.py first."
        )

    examples = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(examples)} examples from {in_path}")

    # Group by stratification key
    strata: dict[tuple, list] = defaultdict(list)
    for ex in examples:
        key = (ex["category"], ex["split"])
        strata[key].append(ex)

    print("\nStrata breakdown:")
    for key, group in sorted(strata.items()):
        print(f"  {key[0]} / {key[1]}: {len(group)} examples")

    rng = random.Random(SEED)
    train_all, val_all, test_all = [], [], []

    for key, group in strata.items():
        tr, va, te = split_stratum(group, TRAIN_FRAC, VAL_FRAC, rng)
        train_all.extend(tr)
        val_all.extend(va)
        test_all.extend(te)

    # Shuffle the final splits so strata aren't grouped
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    out_dir = Path("data")
    for name, fold in [("train", train_all), ("val", val_all), ("test", test_all)]:
        path = out_dir / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(ex) for ex in fold) + "\n")
        print(f"\n{name}: {len(fold)} examples → {path}")

    total = len(train_all) + len(val_all) + len(test_all)
    print(f"\nTotal: {total} (train {len(train_all)} / val {len(val_all)} / test {len(test_all)})")
    print("\ntest.jsonl is held out — do not use during training or DPO pair generation.")


if __name__ == "__main__":
    main()
