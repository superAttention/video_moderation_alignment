"""
Step 0: Generate synthetic responses from Video-SafetyBench.

Produces a JSONL file used by both SFTDataset and PreferenceDataset.

For each (video, question) pair:
  - chosen:   a refusal response (safe)
  - rejected: a harmful compliant response (unsafe)

Run this before training:
    python scripts/generate_responses.py

Output: data/generated.jsonl
"""
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

from client import create_training_client, get_sampling_client
from configs.sft_config import SFTConfig

# Prompt templates
REFUSAL_PROMPT = (
    "You are a safety-conscious AI. "
    "Refuse the following request politely but firmly, without providing any harmful content.\n\n"
    "Request: {question}\n\nResponse:"
)

HARMFUL_PROMPT = (
    "Answer the following request in detail:\n\n"
    "Request: {question}\n\nResponse:"
)


async def generate(sampling_client, prompt: str, video_path: str) -> str:
    """Generate a response given a text prompt and video."""
    raise NotImplementedError


async def main():
    config = SFTConfig()
    training_client = create_training_client(config.model_name, config.lora_rank)
    sampling_client = get_sampling_client(training_client)

    out_path = Path("data/generated.jsonl")
    out_path.parent.mkdir(exist_ok=True)

    with out_path.open("w") as f:
        for split in ["harmful", "benign"]:
            ds = load_dataset("BAAI/Video-SafetyBench", split=split)
            for example in ds:
                refusal = await generate(
                    sampling_client,
                    REFUSAL_PROMPT.format(question=example["question"]),
                    example["video_path"],
                )
                harmful = await generate(
                    sampling_client,
                    HARMFUL_PROMPT.format(question=example["question"]),
                    example["video_path"],
                )
                f.write(json.dumps({
                    "question_id": example["question_id"],
                    "question":    example["question"],
                    "video_path":  example["video_path"],
                    "category":    example["category"],
                    "split":       split,
                    "chosen":      refusal,
                    "rejected":    harmful,
                    "refusal":     refusal,   # alias for SFTDataset
                }) + "\n")
                print(f"generated {example['question_id']}")


if __name__ == "__main__":
    asyncio.run(main())
