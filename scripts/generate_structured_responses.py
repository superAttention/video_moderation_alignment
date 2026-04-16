import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Generate structured JSON training data from Video-SafetyBench.

For each (video, question) pair across both harmful and benign splits:
  - action, risk_category, sub_category: deterministic from benchmark metadata
  - short_rationale: Best-of-K selection — sample K candidates from the base model
                     conditioned on the correct action, score all K with the LLM
                     judge concurrently, keep the highest-scoring one

Examples where best candidate scores below --min_score are skipped.

Run before training:
    python scripts/generate_structured_responses.py

Output: data/generated_structured.jsonl
        data/generated_structured_skipped.jsonl
"""
import json
import asyncio
import argparse
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI
import tinker

load_dotenv()

from client import create_eval_sampling_client, load_processor
from configs.sft_config import SFTConfig
from data.video_utils import resolve_path, extract_frames
from data.schema import (
    RATIONALE_GENERATION_PROMPT,
    StructuredResponse,
    ground_truth_action,
    category_to_risk,
    path_to_subcategory,
    to_json_string,
    make_question_with_prompt,
    decode_model_output,
)
from rewards.schema_evaluator import SchemaEvaluator



async def generate_best_rationale(
    sampling_client, processor, tokenizer,
    evaluator: SchemaEvaluator,
    video_path: str,
    risk_category: str,
    sub_category: str,
    action: str,
    n_samples: int,
    question: str,
    loop: asyncio.AbstractEventLoop,
) -> tuple[str, float]:
    """
    Sample n_samples rationale candidates and return (best_rationale, best_score).
    The K judge calls are made concurrently via run_in_executor.
    """
    prompt = RATIONALE_GENERATION_PROMPT.format(
        category=risk_category,
        sub_category=sub_category,
        action=action,
    )
    frames = extract_frames(resolve_path(video_path))
    inputs = processor(text=prompt, images=frames, return_tensors="pt")
    token_ids = inputs["input_ids"][0].tolist()

    result = await sampling_client.sample_async(
        prompt=tinker.types.ModelInput.from_ints(token_ids),
        num_samples=n_samples,
        sampling_params=tinker.types.SamplingParams(max_tokens=128, temperature=0.7),
    )
    candidates = [
        decode_model_output(tokenizer, seq.tokens)
        for seq in result.sequences
    ]

    # Score all K candidates concurrently (each is a blocking HTTP call)
    def score_one(rationale: str) -> float:
        tmp = StructuredResponse(
            action=action,
            risk_category=risk_category,
            sub_category=sub_category,
            short_rationale=rationale,
        )
        return evaluator.rationale_score(tmp, {"question": question})

    scores = await asyncio.gather(*[
        loop.run_in_executor(None, score_one, c) for c in candidates
    ])

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return candidates[best_idx], scores[best_idx]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rationale_samples", type=int, default=5,
                        help="Number of rationale candidates per example (Best-of-K)")
    parser.add_argument("--min_score", type=float, default=3.0,
                        help="Min judge score (1-5) to include an example")
    args = parser.parse_args()

    config = SFTConfig()
    sampling_client = await create_eval_sampling_client(config.model_name)
    tokenizer = sampling_client.get_tokenizer()
    processor = load_processor(config.model_name)

    judge_client = OpenAI(
        base_url="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
        api_key=os.environ["TINKER_API_KEY"],
    )
    evaluator = SchemaEvaluator(judge_client, model=config.model_name)

    out_path     = Path("data/generated_structured.jsonl")
    skipped_path = Path("data/generated_structured_skipped.jsonl")
    out_path.parent.mkdir(exist_ok=True)

    score_counts: Counter = Counter()
    n_total = n_skipped = 0
    score_sum = 0.0
    skipped_examples = []
    loop = asyncio.get_event_loop()

    with out_path.open("w") as f:
        for split in ["harmful", "benign"]:
            ds = load_dataset("BAAI/Video-SafetyBench", split=split)
            ds = ds.filter(lambda x: x["category"] in config.categories)
            total = len(ds)

            for i, example in enumerate(ds):
                n_total += 1
                print(f"[{split}] {i+1}/{total} — {example['question_id']}", flush=True)

                action   = ground_truth_action(split)
                risk_cat = category_to_risk(example["category"])
                sub_cat  = path_to_subcategory(example["video_path"])
                question = make_question_with_prompt(example["question"])

                rationale, score = await generate_best_rationale(
                    sampling_client, processor, tokenizer, evaluator,
                    video_path=example["video_path"],
                    risk_category=risk_cat,
                    sub_category=sub_cat,
                    action=action,
                    n_samples=args.n_rationale_samples,
                    question=question,
                    loop=loop,
                )

                score_counts[int(score)] += 1

                if score < args.min_score:
                    n_skipped += 1
                    skipped_examples.append({
                        "question_id": example["question_id"],
                        "split": split,
                        "category": example["category"],
                        "best_score": score,
                        "best_rationale": rationale,
                    })
                    print(f"    → skipped (score {score:.1f} < {args.min_score})")
                    continue

                score_sum += score

                chosen_resp = StructuredResponse(
                    action=action, risk_category=risk_cat,
                    sub_category=sub_cat, short_rationale=rationale,
                )
                wrong_action = "ESCALATE_WITH_ANALYSIS" if action == "REFUSE" else "REFUSE"
                rejected_resp = StructuredResponse(
                    action=wrong_action, risk_category=risk_cat, sub_category=sub_cat,
                    short_rationale=(
                        "This content may warrant human review rather than outright refusal."
                        if wrong_action == "ESCALATE_WITH_ANALYSIS"
                        else "This content appears harmful and should be refused."
                    ),
                )
                chosen_str   = to_json_string(chosen_resp)
                rejected_str = to_json_string(rejected_resp)

                f.write(json.dumps({
                    "question_id":      example["question_id"],
                    "question":         question,
                    "video_path":       example["video_path"],
                    "category":         example["category"],
                    "split":            split,
                    "chosen":           chosen_str,
                    "rejected":         rejected_str,
                    "refusal":          chosen_str,
                    "gt_action":        action,
                    "gt_risk_category": risk_cat,
                    "gt_sub_category":  sub_cat,
                    "rationale_score":  score,
                }) + "\n")
                f.flush()

    if skipped_examples:
        skipped_path.write_text(
            "\n".join(json.dumps(e) for e in skipped_examples) + "\n"
        )

    n_kept = n_total - n_skipped
    avg_score = score_sum / n_kept if n_kept > 0 else 0.0
    print(f"\nGenerated {n_kept}/{n_total} examples  "
          f"({n_skipped} skipped, avg score: {avg_score:.2f})")
    print("Score distribution: " +
          "  ".join(f"{k}={v}" for k, v in sorted(score_counts.items())))
    if skipped_examples:
        print(f"Skipped → {skipped_path}")
    print(f"\nOutput → {out_path}")
    print("Run scripts/split_data.py next.")


if __name__ == "__main__":
    asyncio.run(main())
