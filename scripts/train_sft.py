"""Entry point for SFT training."""
import asyncio
from dotenv import load_dotenv

load_dotenv()

from configs.sft_config import SFTConfig
from client import create_training_client
from data.sft_dataset import SFTDataset
from trainers.sft_trainer import SFTTrainer


async def main():
    config = SFTConfig()

    training_client = create_training_client(config.model_name, config.lora_rank)
    tokenizer = training_client.get_tokenizer()

    train_dataset = SFTDataset(config.train_data_path, tokenizer, config.max_seq_len)
    val_dataset = SFTDataset(config.val_data_path, tokenizer, config.max_seq_len) if config.val_data_path else None

    trainer = SFTTrainer(
        training_client=training_client,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
