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
    - refusal_rate:   % of responses classified as refusals
    - attack_success: % of responses that comply (1 - refusal_rate)

Results saved to: outputs/eval_{checkpoint}.json
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
from rewards.rjscore import RJScoreEvaluator


async def generate_response(sampling_client, processor, tokenizer, question: str, video_path: str) -> str:
    """Run inference on one (question, video) pair. Returns decoded response string."""
    frames = extract_frames(video_path)

    inputs = processor(
        text=question,
        images=frames,
        return_tensors="pt",
    )
    token_ids = inputs["input_ids"][0].tolist()

    result = await sampling_client.sample_async(
        prompt=tinker.types.ModelInput.from_ints(token_ids),
        num_samples=1,
        sampling_params=tinker.types.SamplingParams(max_tokens=256, temperature=0.0),
    )
    return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)


async def evaluate_split(sampling_client, processor, tokenizer, evaluator: RJScoreEvaluator, split: str, categories: tuple) -> dict:
    """
    Run inference on every example in the split (filtered to categories).

    Returns:
    {
        "overall": {"refusal_rate": float, "attack_success": float, "n": int},
        "by_category": {
            "1_Violent_Crimes": {"refusal_rate": float, "attack_success": float, "n": int},
            "10_Hate":          {...},
            "12_Sexual_Content": {...},
        }
    }
    """
    ds = load_dataset("BAAI/Video-SafetyBench", split=split)
    ds = ds.filter(lambda x: x["category"] in categories)

    per_category = defaultdict(list)  # category -> list of bool (True = refused)
    loop = asyncio.get_event_loop()

    for i, example in enumerate(ds):
        print(f"  [{split}] {i+1}/{len(ds)} — {example['question_id']}", flush=True)
        response = await generate_response(
            sampling_client, processor, tokenizer,
            question=example["question"],
            video_path=resolve_path(example["video_path"]),
        )
        # run sync judge call in thread so it doesn't block the event loop
        refused = await loop.run_in_executor(
            None, evaluator.is_safe, example["question"], response
        )
        per_category[example["category"]].append(refused)

    def metrics(results: list[bool]) -> dict:
        rate = sum(results) / len(results)
        return {"refusal_rate": rate, "attack_success": 1 - rate, "n": len(results)}

    all_results = [r for results in per_category.values() for r in results]
    return {
        "overall": metrics(all_results),
        "by_category": {cat: metrics(res) for cat, res in per_category.items()},
    }


def print_results(split: str, results: dict) -> None:
    print(f"\n  [{split}] overall — refusal: {results['overall']['refusal_rate']:.1%}  "
          f"attack_success: {results['overall']['attack_success']:.1%}  "
          f"n={results['overall']['n']}")
    for cat, m in results["by_category"].items():
        print(f"    {cat:<25} refusal: {m['refusal_rate']:.1%}  n={m['n']}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="base", help="Label for this eval run")
    parser.add_argument("--checkpoint_name", default="", help="Tinker model_path for trained checkpoint (empty = base model)")
    args = parser.parse_args()

    config = SFTConfig()
    sampling_client = await create_eval_sampling_client(config.model_name, model_path=args.checkpoint_name)
    tokenizer = sampling_client.get_tokenizer()
    processor = load_processor(config.model_name)

    judge_client = OpenAI(
        base_url="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
        api_key=os.environ["TINKER_API_KEY"],
    )
    evaluator = RJScoreEvaluator(judge_client, model="Qwen/Qwen3-VL-30B-A3B-Instruct")

    results = {}
    for split in ["harmful", "benign"]:
        print(f"\nEvaluating {split} split...")
        results[split] = await evaluate_split(sampling_client, processor, tokenizer, evaluator, split, config.categories)
        print_results(split, results[split])

    out_path = Path("outputs") / f"eval_{args.checkpoint}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"checkpoint": args.checkpoint, "results": results}, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
