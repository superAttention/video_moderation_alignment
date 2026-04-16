import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Best-of-N DPO pair generation from the SFT model.

Runs AFTER SFT. For each example in data/train.jsonl:
  1. Sample N=5 responses from the SFT model at temperature=0.7
  2. Parse each response as a StructuredResponse
  3. Select chosen/rejected pair using a 3-level priority hierarchy:
       Level 1 (priority = wrong/N): wrong action  — deterministic ground truth
       Level 2 (priority = 0.5):     wrong category — deterministic ground truth
       Level 3 (priority = 0.2):     all correct, rank rationale by LLM judge
  4. Chosen selection: for Level 1/2, pick the BEST-scored correct-action sample
     (not just the first), so the chosen side is as strong as possible.
  5. Quality filters:
       --min_chosen_score: skip pair if best correct response scores below threshold
       --min_score_margin: skip Level 3 pair if score(chosen)-score(rejected) < margin

Examples sorted by priority descending in output — highest disagreement first.

Usage:
    python scripts/sample_dpo_pairs.py --sft_checkpoint tinker://uuid/weights/final

Output: data/dpo_pairs.jsonl (sorted by priority descending)
        data/dpo_pairs_skipped.jsonl (filtered-out examples)
"""
import json
import asyncio
import argparse
import os
from collections import Counter, defaultdict

import tinker
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from client import create_eval_sampling_client, load_processor
from configs.sft_config import SFTConfig
from data.video_utils import resolve_path, extract_frames
from data.schema import (
    StructuredResponse,
    parse_response,
    to_json_string,
    ground_truth_action,
    decode_model_output,
)
from rewards.schema_evaluator import SchemaEvaluator

N_SAMPLES = 5
TEMPERATURE = 0.7


async def sample_n(
    sampling_client, processor, tokenizer,
    question: str, video_path: str,
    n: int, temperature: float,
) -> list[str]:
    """Sample N responses from the model for a single (question, video) pair."""
    frames = extract_frames(resolve_path(video_path))
    inputs = processor(text=question, images=frames, return_tensors="pt")
    token_ids = inputs["input_ids"][0].tolist()

    result = await sampling_client.sample_async(
        prompt=tinker.types.ModelInput.from_ints(token_ids),
        num_samples=n,
        sampling_params=tinker.types.SamplingParams(max_tokens=256, temperature=temperature),
    )
    return [
        decode_model_output(tokenizer, seq.tokens)
        for seq in result.sequences
    ]


def score_responses(
    responses: list[StructuredResponse],
    evaluator: SchemaEvaluator,
    example: dict,
) -> list[float]:
    """Score a list of parsed responses with the LLM rationale judge."""
    return [evaluator.rationale_score(r, example) for r in responses]


def select_pair(
    samples: list[str],
    gt_action: str,
    gt_risk_category: str,
    gt_sub_category: str,
    sft_target: str,
    evaluator: SchemaEvaluator,
    example: dict,
    min_chosen_score: float,
    min_score_margin: float,
) -> dict | None:
    """
    Select chosen/rejected pair from N samples using the priority hierarchy.

    Level 1/2: chosen = best-scored correct-action sample (not just first).
    Level 3:   chosen/rejected by rationale score with margin filter.

    Returns None if no pair passes quality filters.
    """
    parsed = [parse_response(s) for s in samples]

    correct_action   = [p for p in parsed if p and p.action == gt_action]
    incorrect_action = [p for p in parsed if p and p.action != gt_action]

    # Level 1: action disagreement — highest signal, fully deterministic
    if incorrect_action:
        if not correct_action:
            # Fall back to SFT target as chosen
            chosen_resp = parse_response(sft_target)
        else:
            # Pick best-scoring correct-action sample as chosen
            scores = score_responses(correct_action, evaluator, example)
            chosen_resp = correct_action[max(range(len(scores)), key=lambda i: scores[i])]
            chosen_score = scores[max(range(len(scores)), key=lambda i: scores[i])]
            if chosen_score < min_chosen_score:
                return {"skip_reason": f"chosen_score {chosen_score:.1f} < {min_chosen_score}"}

        if chosen_resp is None:
            return None

        rejected_resp = incorrect_action[0]
        chosen_score  = evaluator.rationale_score(chosen_resp, example) if not correct_action else chosen_score
        priority  = len(incorrect_action) / len(parsed)
        pair_type = "action"

        return {
            "chosen":        to_json_string(chosen_resp),
            "rejected":      to_json_string(rejected_resp),
            "priority":      priority,
            "pair_type":     pair_type,
            "chosen_score":  chosen_score,
            "rejected_score": None,   # wrong action — score not meaningful
            "score_margin":  None,
        }

    # Level 2: category disagreement among correct-action samples
    elif any(
        p and p.risk_category != gt_risk_category
        for p in parsed if p and p.action == gt_action
    ):
        correct_cat   = [p for p in correct_action if p.risk_category == gt_risk_category]
        incorrect_cat = [p for p in correct_action if p.risk_category != gt_risk_category]
        if not correct_cat or not incorrect_cat:
            return None

        # Pick best-scoring correct-category sample as chosen
        scores = score_responses(correct_cat, evaluator, example)
        best_idx     = max(range(len(scores)), key=lambda i: scores[i])
        chosen_resp  = correct_cat[best_idx]
        chosen_score = scores[best_idx]

        if chosen_score < min_chosen_score:
            return {"skip_reason": f"chosen_score {chosen_score:.1f} < {min_chosen_score}"}

        rejected_resp = incorrect_cat[0]
        return {
            "chosen":        to_json_string(chosen_resp),
            "rejected":      to_json_string(rejected_resp),
            "priority":      0.5,
            "pair_type":     "category",
            "chosen_score":  chosen_score,
            "rejected_score": None,
            "score_margin":  None,
        }

    # Level 3: all samples have correct action + category; rank by rationale quality
    else:
        valid = [p for p in parsed if p and p.action == gt_action]
        if len(valid) < 2:
            return None

        scores    = score_responses(valid, evaluator, example)
        best_idx  = max(range(len(scores)), key=lambda i: scores[i])
        worst_idx = min(range(len(scores)), key=lambda i: scores[i])

        if best_idx == worst_idx:
            return None

        chosen_score   = scores[best_idx]
        rejected_score = scores[worst_idx]
        margin         = chosen_score - rejected_score

        if chosen_score < min_chosen_score:
            return {"skip_reason": f"chosen_score {chosen_score:.1f} < {min_chosen_score}"}
        if margin < min_score_margin:
            return {"skip_reason": f"score_margin {margin:.1f} < {min_score_margin}"}

        return {
            "chosen":        to_json_string(valid[best_idx]),
            "rejected":      to_json_string(valid[worst_idx]),
            "priority":      0.2,
            "pair_type":     "rationale",
            "chosen_score":  chosen_score,
            "rejected_score": rejected_score,
            "score_margin":  margin,
        }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_checkpoint", required=True,
                        help="Tinker checkpoint path for SFT model")
    parser.add_argument("--input",  default="data/train.jsonl")
    parser.add_argument("--output", default="data/dpo_pairs.jsonl")
    parser.add_argument("--n_samples",        type=int,   default=N_SAMPLES)
    parser.add_argument("--min_chosen_score", type=float, default=3.0,
                        help="Skip pair if best correct response scores below this (1-5)")
    parser.add_argument("--min_score_margin", type=float, default=1.0,
                        help="For Level 3 pairs: skip if chosen-rejected score margin < this")
    args = parser.parse_args()

    config = SFTConfig()
    sampling_client = await create_eval_sampling_client(
        config.model_name, model_path=args.sft_checkpoint
    )
    tokenizer = sampling_client.get_tokenizer()
    processor = load_processor(config.model_name)

    judge_client = OpenAI(
        base_url="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
        api_key=os.environ["TINKER_API_KEY"],
    )
    evaluator = SchemaEvaluator(judge_client, model=config.model_name)

    examples = [
        json.loads(line)
        for line in Path(args.input).read_text().splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(examples)} examples from {args.input}")

    pairs: list[dict] = []
    skipped: list[dict] = []
    pair_type_counts: Counter = Counter()
    skip_reason_counts: Counter = Counter()

    for i, example in enumerate(examples):
        print(f"  [{i+1}/{len(examples)}] {example['question_id']}", flush=True)

        gt_action   = example["gt_action"]
        gt_risk_cat = example["gt_risk_category"]
        gt_sub_cat  = example["gt_sub_category"]

        raw_samples = await sample_n(
            sampling_client, processor, tokenizer,
            question=example["question"],
            video_path=example["video_path"],
            n=args.n_samples,
            temperature=TEMPERATURE,
        )

        result = select_pair(
            samples=raw_samples,
            gt_action=gt_action,
            gt_risk_category=gt_risk_cat,
            gt_sub_category=gt_sub_cat,
            sft_target=example["chosen"],
            evaluator=evaluator,
            example={"question": example["question"]},
            min_chosen_score=args.min_chosen_score,
            min_score_margin=args.min_score_margin,
        )

        if result is None:
            skip_reason_counts["no_valid_pair"] += 1
            skipped.append({"question_id": example["question_id"], "reason": "no_valid_pair"})
            print(f"    → skipped (no valid pair)")
            continue

        if "skip_reason" in result:
            reason = result["skip_reason"]
            skip_reason_counts[reason.split()[0]] += 1
            skipped.append({"question_id": example["question_id"], "reason": reason})
            print(f"    → skipped ({reason})")
            continue

        pair_type_counts[result["pair_type"]] += 1
        pairs.append({
            "question_id":      example["question_id"],
            "question":         example["question"],
            "video_path":       example["video_path"],
            "category":         example["category"],
            "split":            example["split"],
            "gt_action":        gt_action,
            "gt_risk_category": gt_risk_cat,
            "gt_sub_category":  gt_sub_cat,
            "chosen":           result["chosen"],
            "rejected":         result["rejected"],
            "priority":         result["priority"],
            "pair_type":        result["pair_type"],
            "chosen_score":     result["chosen_score"],
            "rejected_score":   result["rejected_score"],
            "score_margin":     result["score_margin"],
        })

    # Sort by priority descending — highest disagreement examples first
    pairs.sort(key=lambda x: x["priority"], reverse=True)

    out_path = Path(args.output)
    out_path.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")

    skipped_path = Path(args.output.replace(".jsonl", "_skipped.jsonl"))
    if skipped:
        skipped_path.write_text("\n".join(json.dumps(s) for s in skipped) + "\n")

    print(f"\nGenerated {len(pairs)}/{len(examples)} DPO pairs → {out_path}")
    print(f"Skipped:    {len(skipped)} ({len(examples) - len(pairs) - len(skipped)} no-pair)")
    print("\nPair type breakdown:")
    for pt, count in sorted(pair_type_counts.items()):
        print(f"  {pt}: {count}")
    if skip_reason_counts:
        print("\nSkip reason breakdown:")
        for reason, count in sorted(skip_reason_counts.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
