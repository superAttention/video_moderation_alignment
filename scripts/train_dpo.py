import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""Entry point for DPO training.

Typical usage (after SFT):
    python scripts/train_dpo.py \\
        --sft_checkpoint tinker://uuid/weights/final

The SFT checkpoint is used as the frozen reference model.
The policy starts from the same checkpoint and is updated.
"""
import asyncio
import argparse
from dotenv import load_dotenv

load_dotenv()

from configs.dpo_config import DPOConfig
from client import create_training_client, load_processor
from data.preference_dataset import PreferenceDataset
from trainers.dpo_trainer import DPOTrainer


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_checkpoint", default="", help="Tinker checkpoint path for SFT model (used as reference)")
    args = parser.parse_args()

    config = DPOConfig()

    # Policy client — initialized from SFT, weights will be updated
    policy_client = await create_training_client(config.model_name, config.lora_rank)

    # Reference client — frozen SFT model, never call optim_step on this
    ref_client = await create_training_client(config.model_name, config.lora_rank)

    if args.sft_checkpoint:
        await (await policy_client.load_state_async(args.sft_checkpoint))
        await (await ref_client.load_state_async(args.sft_checkpoint))

    tokenizer = policy_client.get_tokenizer()
    processor = load_processor(config.model_name)

    train_dataset = PreferenceDataset(config.train_data_path, tokenizer, processor, config.max_seq_len)
    val_dataset   = PreferenceDataset(config.val_data_path,   tokenizer, processor, config.max_seq_len) if config.val_data_path else None

    trainer = DPOTrainer(
        training_client=policy_client,
        ref_client=ref_client,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
