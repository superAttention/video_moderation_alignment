import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Step 1: Generate synthetic responses from Video-SafetyBench.

Produces data/generated.jsonl used by SFTDataset and PreferenceDataset.

For each (video, question) pair:
  - chosen:   refusal response (safe)
  - rejected: harmful compliant response (unsafe)

Run before training:
    python scripts/generate_responses.py
"""
import sys
import json
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from datasets import load_dataset
import tinker

load_dotenv()

from client import create_training_client, get_sampling_client, load_processor
from configs.sft_config import SFTConfig
from data.video_utils import resolve_path, extract_frames

REFUSAL_PROMPT = (
    "You are a safety-conscious AI. "
    "Refuse the following request politely but firmly, without providing any harmful content.\n\n"
    "Request: {question}\n\nResponse:"
)

HARMFUL_PROMPT = (
    "Answer the following request in detail:\n\n"
    "Request: {question}\n\nResponse:"
)


async def generate(sampling_client, processor, tokenizer, prompt: str, video_path: str) -> str:
    """Generate a response given a text prompt and video."""
    frames = extract_frames(resolve_path(video_path))

    inputs = processor(
        text=prompt,
        images=frames,
        return_tensors="pt",
    )
    token_ids = inputs["input_ids"][0].tolist()

    result = await sampling_client.sample_async(
        prompt=tinker.types.ModelInput.from_ints(token_ids),
        num_samples=1,
        sampling_params=tinker.types.SamplingParams(max_tokens=512, temperature=0.7),
    )
    return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)


async def main():
    config = SFTConfig()
    training_client = await create_training_client(config.model_name, config.lora_rank)
    sampling_client = await get_sampling_client(training_client)
    tokenizer = training_client.get_tokenizer()
    processor = load_processor(config.model_name)

    out_path = Path("data/generated.jsonl")
    out_path.parent.mkdir(exist_ok=True)

    with out_path.open("w") as f:
        for split in ["harmful", "benign"]:
            ds = load_dataset("BAAI/Video-SafetyBench", split=split)
            ds = ds.filter(lambda x: x["category"] in config.categories)
            total = len(ds)

            for i, example in enumerate(ds):
                print(f"[{split}] {i+1}/{total} — {example['question_id']}")

                refusal = await generate(
                    sampling_client, processor, tokenizer,
                    prompt=REFUSAL_PROMPT.format(question=example["question"]),
                    video_path=example["video_path"],
                )
                harmful = await generate(
                    sampling_client, processor, tokenizer,
                    prompt=HARMFUL_PROMPT.format(question=example["question"]),
                    video_path=example["video_path"],
                )

                f.write(json.dumps({
                    "question_id": example["question_id"],
                    "question":    example["question"],
                    "video_path":  example["video_path"],
                    "category":    example["category"],
                    "split":       split,
                    "chosen":      refusal,
                    "rejected":    harmful,
                    "refusal":     refusal,
                }) + "\n")
                f.flush()


if __name__ == "__main__":
    asyncio.run(main())
