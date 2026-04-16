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
  4. Examples with highest disagreement (most wrong-action samples) are
     sorted first in the output — they carry the strongest learning signal.

Examples where all N samples agree AND the action is correct are still
included as Level 3 pairs (rationale quality improvement).

Usage:
    python scripts/sample_dpo_pairs.py --sft_checkpoint tinker://uuid/weights/final

Output: data/dpo_pairs.jsonl (sorted by priority descending)
"""
import json
import asyncio
import argparse
import os
from collections import defaultdict

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
        tokenizer.decode(seq.tokens, skip_special_tokens=True)
        for seq in result.sequences
    ]


def select_pair(
    samples: list[str],
    gt_action: str,
    gt_risk_category: str,
    gt_sub_category: str,
    sft_target: str,
    evaluator: SchemaEvaluator,
    example: dict,
) -> dict | None:
    """
    Select chosen/rejected pair from N samples using the priority hierarchy.
    Returns None if no meaningful pair can be constructed.
    """
    parsed = [parse_response(s) for s in samples]

    correct_action   = [p for p in parsed if p and p.action == gt_action]
    incorrect_action = [p for p in parsed if p and p.action != gt_action]

    # Level 1: action disagreement — highest signal, fully deterministic
    if incorrect_action:
        chosen_resp = correct_action[0] if correct_action else parse_response(sft_target)
        rejected_resp = incorrect_action[0]
        if chosen_resp is None:
            return None
        priority = len(incorrect_action) / len(parsed)
        pair_type = "action"

    # Level 2: category disagreement among correct-action samples
    elif any(
        p and p.risk_category != gt_risk_category
        for p in parsed if p and p.action == gt_action
    ):
        correct_cat   = [p for p in correct_action if p.risk_category == gt_risk_category]
        incorrect_cat = [p for p in correct_action if p.risk_category != gt_risk_category]
        if not correct_cat or not incorrect_cat:
            return None
        chosen_resp  = correct_cat[0]
        rejected_resp = incorrect_cat[0]
        priority = 0.5
        pair_type = "category"

    # Level 3: all samples have correct action + category; rank by rationale quality
    else:
        valid = [p for p in parsed if p and p.action == gt_action]
        if len(valid) < 2:
            return None
        scores = [evaluator.rationale_score(p, example) for p in valid]
        best_idx  = max(range(len(scores)), key=lambda i: scores[i])
        worst_idx = min(range(len(scores)), key=lambda i: scores[i])
        if best_idx == worst_idx or scores[best_idx] == scores[worst_idx]:
            return None
        chosen_resp  = valid[best_idx]
        rejected_resp = valid[worst_idx]
        priority = 0.2
        pair_type = "rationale"

    return {
        "chosen":   to_json_string(chosen_resp),
        "rejected": to_json_string(rejected_resp),
        "priority": priority,
        "pair_type": pair_type,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_checkpoint", required=True,
                        help="Tinker checkpoint path for SFT model")
    parser.add_argument("--input",  default="data/train.jsonl")
    parser.add_argument("--output", default="data/dpo_pairs.jsonl")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
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

    pairs = []
    pair_type_counts: dict[str, int] = defaultdict(int)

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

        pair = select_pair(
            samples=raw_samples,
            gt_action=gt_action,
            gt_risk_category=gt_risk_cat,
            gt_sub_category=gt_sub_cat,
            sft_target=example["chosen"],
            evaluator=evaluator,
            example=example,
        )

        if pair is None:
            print(f"    → skipped (no meaningful pair)")
            continue

        pair_type_counts[pair["pair_type"]] += 1
        pairs.append({
            "question_id":      example["question_id"],
            "question":         example["question"],
            "video_path":       example["video_path"],
            "category":         example["category"],
            "split":            example["split"],
            "gt_action":        gt_action,
            "gt_risk_category": gt_risk_cat,
            "gt_sub_category":  gt_sub_cat,
            "chosen":           pair["chosen"],
            "rejected":         pair["rejected"],
            "priority":         pair["priority"],
            "pair_type":        pair["pair_type"],
        })

    # Sort by priority descending — highest disagreement examples first
    pairs.sort(key=lambda x: x["priority"], reverse=True)

    out_path = Path(args.output)
    out_path.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")

    print(f"\nGenerated {len(pairs)} DPO pairs → {out_path}")
    print("Pair type breakdown:")
    for pt, count in sorted(pair_type_counts.items()):
        print(f"  {pt}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
