import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Evaluate a model checkpoint on Video-SafetyBench.

Run at three points for comparison:
    python scripts/evaluate.py --checkpoint base   # before any training
    python scripts/evaluate.py --checkpoint sft    # after SFT
    python scripts/evaluate.py --checkpoint dpo    # after DPO

Metrics per split, per category:
  Legacy (backward-compatible with baseline):
    - refusal_rate:    % of responses classified as refusals by RJScore
    - attack_success:  % of responses that comply (1 - refusal_rate)

  Structured output (base model outputs free-form, so these are 0 for base):
    - valid_json_rate:      % of responses that parse as valid JSON schema
    - action_accuracy:      % with correct REFUSE/ESCALATE_WITH_ANALYSIS
    - category_accuracy:    % with correct risk_category
    - subcategory_accuracy: % with correct sub_category
    - over_refusal_rate:    % of BENIGN examples that were incorrectly REFUSED
    - avg_rationale_score:  mean LLM-judge score (1-5) for short_rationale

Outputs:
    outputs/eval_{checkpoint}.jsonl   — one line per example (response + all scores)
    outputs/eval_{checkpoint}.json    — aggregated metrics
"""
import os
import json
import asyncio
import argparse
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from datasets import load_dataset
import tinker

load_dotenv()

from openai import OpenAI
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
from rewards.rjscore import RJScoreEvaluator, THRESHOLD
from rewards.schema_evaluator import SchemaEvaluator


async def generate_response(sampling_client, processor, tokenizer, question: str, video_path: str) -> str:
    """Run inference on one (question, video) pair. Returns decoded response string."""
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
    rj_evaluator: RJScoreEvaluator,
    schema_evaluator: SchemaEvaluator,
    split: str,
    categories: tuple,
    trace_file,
    use_structured_prompt: bool,
) -> dict:
    """
    Run inference on every example in the split (filtered to categories).
    Computes both legacy RJScore metrics and structured output metrics.
    Writes one JSON line per example to trace_file as it goes.
    """
    ds = load_dataset("BAAI/Video-SafetyBench", split=split)
    ds = ds.filter(lambda x: x["category"] in categories)

    loop = asyncio.get_event_loop()

    # Per-category accumulators
    per_cat_refused:             dict[str, list[bool]]  = defaultdict(list)
    per_cat_valid_json:          dict[str, list[bool]]  = defaultdict(list)
    per_cat_action_correct:      dict[str, list[bool]]  = defaultdict(list)
    per_cat_category_correct:    dict[str, list[bool]]  = defaultdict(list)
    per_cat_subcategory_correct: dict[str, list[bool]]  = defaultdict(list)
    per_cat_rationale_scores:    dict[str, list[float]] = defaultdict(list)

    gt_action      = ground_truth_action(split)
    is_benign      = split == "benign"

    for i, example in enumerate(ds):
        print(f"  [{split}] {i+1}/{len(ds)} — {example['question_id']}", flush=True)

        question = (
            make_question_with_prompt(example["question"])
            if use_structured_prompt
            else example["question"]
        )

        response = await generate_response(
            sampling_client, processor, tokenizer,
            question=question,
            video_path=resolve_path(example["video_path"]),
        )

        cat  = example["category"]
        gt_risk_cat = category_to_risk(cat)
        gt_sub_cat  = path_to_subcategory(example["video_path"])

        # ── Legacy RJScore ────────────────────────────────────────────────
        risk_score = await loop.run_in_executor(
            None, rj_evaluator.score,
            example["harmful_intention"], response, cat,
        )
        refused = risk_score <= THRESHOLD
        per_cat_refused[cat].append(refused)

        # ── Structured metrics ────────────────────────────────────────────
        parsed = parse_response(response)
        valid  = parsed is not None

        per_cat_valid_json[cat].append(valid)
        per_cat_action_correct[cat].append(
            valid and schema_evaluator.action_correct(parsed, gt_action)
        )
        per_cat_category_correct[cat].append(
            valid and schema_evaluator.category_correct(parsed, gt_risk_cat)
        )
        per_cat_subcategory_correct[cat].append(
            valid and schema_evaluator.subcategory_correct(parsed, gt_sub_cat)
        )

        rat_score = 0.0
        if valid:
            rat_score = await loop.run_in_executor(
                None, schema_evaluator.rationale_score,
                parsed, {"question": question},
            )
        per_cat_rationale_scores[cat].append(rat_score)

        trace_file.write(json.dumps({
            "split":               split,
            "question_id":         example["question_id"],
            "category":            cat,
            "question":            example["question"],
            "harmful_intention":   example["harmful_intention"],
            "response":            response,
            # Legacy
            "risk_score":          risk_score,
            "refused":             refused,
            # Structured
            "valid_json":          valid,
            "action_correct":      bool(valid and schema_evaluator.action_correct(parsed, gt_action)),
            "category_correct":    bool(valid and schema_evaluator.category_correct(parsed, gt_risk_cat)),
            "subcategory_correct": bool(valid and schema_evaluator.subcategory_correct(parsed, gt_sub_cat)),
            "rationale_score":     rat_score,
            "gt_action":           gt_action,
            "gt_risk_category":    gt_risk_cat,
            "gt_sub_category":     gt_sub_cat,
        }) + "\n")
        trace_file.flush()

    # ── Aggregate metrics ─────────────────────────────────────────────────
    def agg(results: list, scores: list[float] | None = None) -> dict:
        n = len(results)
        rate = sum(results) / n if n else 0.0
        d = {"rate": rate, "n": n}
        if scores is not None:
            d["avg_score"] = sum(scores) / n if n else 0.0
        return d

    all_cats = list(per_cat_refused.keys())

    def overall(per_cat: dict) -> list:
        return [v for vals in per_cat.values() for v in vals]

    def overall_scores(per_cat: dict) -> list:
        return [v for vals in per_cat.values() for v in vals]

    all_refused       = overall(per_cat_refused)
    all_valid         = overall(per_cat_valid_json)
    all_action        = overall(per_cat_action_correct)
    all_category      = overall(per_cat_category_correct)
    all_subcategory   = overall(per_cat_subcategory_correct)
    all_rat_scores    = overall_scores(per_cat_rationale_scores)
    n_total           = len(all_refused)

    over_refusal = None
    if is_benign and n_total > 0:
        # Over-refusal: benign examples incorrectly refused
        # Using action_correct as proxy: action_correct=False on benign means model said REFUSE
        over_refusal = sum(not v for v in all_action) / n_total

    def build_overall() -> dict:
        d = {
            "refusal_rate":         sum(all_refused)     / n_total,
            "attack_success":       1 - sum(all_refused) / n_total,
            "valid_json_rate":      sum(all_valid)        / n_total,
            "action_accuracy":      sum(all_action)       / n_total,
            "category_accuracy":    sum(all_category)     / n_total,
            "subcategory_accuracy": sum(all_subcategory)  / n_total,
            "avg_rationale_score":  sum(all_rat_scores)   / n_total,
            "n":                    n_total,
        }
        if over_refusal is not None:
            d["over_refusal_rate"] = over_refusal
        return d

    def build_by_category() -> dict:
        result = {}
        for c in all_cats:
            n = len(per_cat_refused[c])
            ref_rate = sum(per_cat_refused[c]) / n
            result[c] = {
                "refusal_rate":         ref_rate,
                "attack_success":       1 - ref_rate,
                "valid_json_rate":      sum(per_cat_valid_json[c])          / n,
                "action_accuracy":      sum(per_cat_action_correct[c])      / n,
                "category_accuracy":    sum(per_cat_category_correct[c])    / n,
                "subcategory_accuracy": sum(per_cat_subcategory_correct[c]) / n,
                "avg_rationale_score":  sum(per_cat_rationale_scores[c])    / n,
                "n":                    n,
            }
        return result

    return {"overall": build_overall(), "by_category": build_by_category()}


def print_results(split: str, results: dict) -> None:
    ov = results["overall"]
    print(f"\n  [{split}] overall "
          f"refusal={ov['refusal_rate']:.1%}  "
          f"action_acc={ov['action_accuracy']:.1%}  "
          f"cat_acc={ov['category_accuracy']:.1%}  "
          f"valid_json={ov['valid_json_rate']:.1%}  "
          f"rationale={ov['avg_rationale_score']:.2f}  "
          + (f"over_refusal={ov.get('over_refusal_rate', 0):.1%}  " if split == "benign" else "")
          + f"n={ov['n']}")
    for cat, m in results["by_category"].items():
        print(f"    {cat:<25} action_acc={m['action_accuracy']:.1%}  "
              f"refusal={m['refusal_rate']:.1%}  n={m['n']}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      default="base",
                        help="Label for this eval run (base | sft | dpo)")
    parser.add_argument("--checkpoint_name", default="",
                        help="Tinker model_path for trained checkpoint (empty = base model)")
    parser.add_argument("--no_structured_prompt", action="store_true",
                        help="Skip SYSTEM_PROMPT prefix (use for base model eval)")
    args = parser.parse_args()

    # Base model evals use the raw question; trained checkpoints use the structured prompt
    use_structured_prompt = not args.no_structured_prompt and args.checkpoint != "base"

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
    rj_evaluator     = RJScoreEvaluator(judge_client, model=config.model_name)
    schema_evaluator = SchemaEvaluator(judge_client,  model=config.model_name)

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
                rj_evaluator, schema_evaluator,
                split, config.categories, trace_file,
                use_structured_prompt=use_structured_prompt,
            )
            print_results(split, results[split])

    metrics_path.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "structured_prompt": use_structured_prompt,
        "results": results,
    }, indent=2))
    print(f"\nTrace   → {trace_path}")
    print(f"Metrics → {metrics_path}")


if __name__ == "__main__":
    asyncio.run(main())
