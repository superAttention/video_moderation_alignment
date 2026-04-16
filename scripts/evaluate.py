import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Evaluate a model checkpoint on Video-SafetyBench.

All three checkpoints use the SAME structured prompt — never vary the prompt
between runs, or improvements cannot be attributed to training.

Usage:
    python scripts/evaluate.py --checkpoint base
    python scripts/evaluate.py --checkpoint sft  --checkpoint_name tinker://uuid/weights/final
    python scripts/evaluate.py --checkpoint dpo  --checkpoint_name tinker://uuid/weights/final

    # Skip the LLM rationale judge for faster runs (e.g. debugging):
    python scripts/evaluate.py --checkpoint sft --checkpoint_name ... --skip_rationale

Metrics (per split: harmful / benign):
    valid_json_rate      — % of responses that parse as valid schema JSON
    action_accuracy      — % with correct REFUSE / ESCALATE_WITH_ANALYSIS
                           (invalid JSON counts as wrong)
    category_accuracy    — % with correct risk_category (invalid JSON = wrong)
    avg_rationale_score  — mean LLM-judge score 1-5 for short_rationale quality
                           (only for valid JSON; skipped if --skip_rationale)

The harmful/benign split tells the core story:
    harmful action_accuracy  → safety  (model correctly refuses harmful requests)
    benign  action_accuracy  → utility (model correctly escalates, doesn't over-refuse)

Outputs:
    outputs/eval_{checkpoint}.jsonl  — one line per example
    outputs/eval_{checkpoint}.json   — aggregated metrics

Run compare_evals.py after all three checkpoints to print the full table.
"""
import os
import json
import asyncio
import argparse
from pathlib import Path
from collections import defaultdict

import tinker
from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI

load_dotenv()

from client import create_eval_sampling_client, load_processor
from configs.sft_config import SFTConfig
from data.video_utils import resolve_path, extract_frames
from data.schema import (
    parse_response,
    ground_truth_action,
    category_to_risk,
    path_to_subcategory,
    make_question_with_prompt,
)
from rewards.schema_evaluator import SchemaEvaluator


async def generate_response(
    sampling_client, processor, tokenizer, question: str, video_path: str
) -> str:
    frames = extract_frames(video_path)
    inputs = processor(text=question, images=frames, return_tensors="pt")
    token_ids = inputs["input_ids"][0].tolist()

    result = await sampling_client.sample_async(
        prompt=tinker.types.ModelInput.from_ints(token_ids),
        num_samples=1,
        sampling_params=tinker.types.SamplingParams(max_tokens=256, temperature=0.0),
    )
    return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)


async def evaluate_split(
    sampling_client, processor, tokenizer,
    evaluator: SchemaEvaluator,
    split: str,
    categories: tuple,
    trace_file,
    skip_rationale: bool,
    no_structured_prompt: bool = False,
) -> dict:
    """
    Evaluate one split (harmful or benign). Writes a trace line per example.

    Returns aggregated metrics dict with keys:
        overall: {valid_json_rate, action_accuracy, category_accuracy,
                  avg_rationale_score, n}
        by_category: {category: same shape}
    """
    ds = load_dataset("BAAI/Video-SafetyBench", split=split)
    ds = ds.filter(lambda x: x["category"] in categories)

    loop = asyncio.get_event_loop()
    gt_action = ground_truth_action(split)

    # Per-category accumulators
    acc: dict[str, dict[str, list]] = defaultdict(lambda: {
        "valid_json":       [],
        "action_correct":   [],
        "category_correct": [],
        "rationale_scores": [],
    })

    for i, example in enumerate(ds):
        print(f"  [{split}] {i+1}/{len(ds)} — {example['question_id']}", flush=True)

        question = (
            make_question_with_prompt(example["question"])
            if not no_structured_prompt
            else example["question"]
        )
        response = await generate_response(
            sampling_client, processor, tokenizer,
            question=question,
            video_path=resolve_path(example["video_path"]),
        )

        cat         = example["category"]
        gt_risk_cat = category_to_risk(cat)
        gt_sub_cat  = path_to_subcategory(example["video_path"])

        parsed         = parse_response(response)
        valid          = parsed is not None
        action_correct = valid and evaluator.action_correct(parsed, gt_action)
        cat_correct    = valid and evaluator.category_correct(parsed, gt_risk_cat)

        rat_score = None
        if valid and not skip_rationale:
            rat_score = await loop.run_in_executor(
                None, evaluator.rationale_score, parsed, {"question": question},
            )

        acc[cat]["valid_json"].append(valid)
        acc[cat]["action_correct"].append(action_correct)
        acc[cat]["category_correct"].append(cat_correct)
        if rat_score is not None:
            acc[cat]["rationale_scores"].append(rat_score)

        trace_file.write(json.dumps({
            "split":            split,
            "question_id":      example["question_id"],
            "category":         cat,
            "question":         example["question"],
            "response":         response,
            "gt_action":        gt_action,
            "gt_risk_category": gt_risk_cat,
            "gt_sub_category":  gt_sub_cat,
            "valid_json":       valid,
            "action_correct":   action_correct,
            "category_correct": cat_correct,
            "rationale_score":  rat_score,
        }) + "\n")
        trace_file.flush()

    def agg(cat_data: dict) -> dict:
        vj  = cat_data["valid_json"]
        ac  = cat_data["action_correct"]
        cc  = cat_data["category_correct"]
        rs  = cat_data["rationale_scores"]
        n   = len(vj)
        return {
            "valid_json_rate":     sum(vj) / n,
            "action_accuracy":     sum(ac) / n,
            "category_accuracy":   sum(cc) / n,
            "avg_rationale_score": sum(rs) / len(rs) if rs else None,
            "n": n,
        }

    all_vj  = [v for d in acc.values() for v in d["valid_json"]]
    all_ac  = [v for d in acc.values() for v in d["action_correct"]]
    all_cc  = [v for d in acc.values() for v in d["category_correct"]]
    all_rs  = [v for d in acc.values() for v in d["rationale_scores"]]
    n_total = len(all_vj)

    return {
        "overall": {
            "valid_json_rate":     sum(all_vj) / n_total,
            "action_accuracy":     sum(all_ac) / n_total,
            "category_accuracy":   sum(all_cc) / n_total,
            "avg_rationale_score": sum(all_rs) / len(all_rs) if all_rs else None,
            "n": n_total,
        },
        "by_category": {cat: agg(data) for cat, data in acc.items()},
    }


def print_split_results(split: str, results: dict) -> None:
    ov = results["overall"]
    rat = f"{ov['avg_rationale_score']:.2f}" if ov["avg_rationale_score"] is not None else "N/A"
    print(
        f"\n  [{split}]"
        f"  valid_json={ov['valid_json_rate']:.1%}"
        f"  action_acc={ov['action_accuracy']:.1%}"
        f"  category_acc={ov['category_accuracy']:.1%}"
        f"  rationale={rat}"
        f"  n={ov['n']}"
    )
    for cat, m in results["by_category"].items():
        print(
            f"    {cat:<30}"
            f"  action_acc={m['action_accuracy']:.1%}"
            f"  valid_json={m['valid_json_rate']:.1%}"
            f"  n={m['n']}"
        )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      default="base",
                        help="Label: base | sft | dpo (used in output filename)")
    parser.add_argument("--checkpoint_name", default="",
                        help="Tinker model path (empty = base model)")
    parser.add_argument("--skip_rationale",       action="store_true",
                        help="Skip LLM rationale judge (faster, no avg_rationale_score)")
    parser.add_argument("--no_structured_prompt", action="store_true",
                        help="Use raw question only — no SYSTEM_PROMPT (for base_freeform run)")
    args = parser.parse_args()

    config = SFTConfig()
    sampling_client = await create_eval_sampling_client(
        config.model_name, model_path=args.checkpoint_name
    )
    tokenizer = sampling_client.get_tokenizer()
    processor = load_processor(config.model_name)

    judge_client = OpenAI(
        base_url="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
        api_key=os.environ["TINKER_API_KEY"],
    )
    evaluator = SchemaEvaluator(judge_client, model=config.model_name)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    trace_path   = out_dir / f"eval_{args.checkpoint}.jsonl"
    metrics_path = out_dir / f"eval_{args.checkpoint}.json"

    results = {}
    with open(trace_path, "w") as trace_file:
        for split in ["harmful", "benign"]:
            print(f"\nEvaluating {split} split...")
            results[split] = await evaluate_split(
                sampling_client, processor, tokenizer,
                evaluator, split, config.categories,
                trace_file, args.skip_rationale,
                no_structured_prompt=args.no_structured_prompt,
            )
            print_split_results(split, results[split])

    metrics_path.write_text(json.dumps({
        "checkpoint":     args.checkpoint,
        "skip_rationale": args.skip_rationale,
        "results":        results,
    }, indent=2))

    print(f"\nTrace   → {trace_path}")
    print(f"Metrics → {metrics_path}")
    print("\nRun scripts/compare_evals.py to print the full base→SFT→DPO table.")


if __name__ == "__main__":
    asyncio.run(main())
