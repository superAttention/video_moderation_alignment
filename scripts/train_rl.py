import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""Entry point for RL training (GRPO / PPO)."""
import asyncio
from dotenv import load_dotenv

load_dotenv()

from configs.rl_config import RLConfig
from client import create_training_client, get_sampling_client
from data.prompt_dataset import PromptDataset
from trainers.rl_trainer import RLTrainer

# Compose your reward functions here — they are summed at runtime
# from rewards.base import ExactMatchReward, LLMJudgeReward
reward_fns = []  # e.g. [ExactMatchReward(references), LLMJudgeReward(...)]


async def main():
    config = RLConfig()

    training_client = await create_training_client(config.model_name, config.lora_rank)
    sampling_client = await get_sampling_client(training_client)

    tokenizer = training_client.get_tokenizer()
    prompt_dataset = PromptDataset(config.prompt_data_path, tokenizer, config.max_prompt_len)

    trainer = RLTrainer(
        training_client=training_client,
        sampling_client=sampling_client,
        config=config,
        prompt_dataset=prompt_dataset,
        reward_fns=reward_fns,
    )
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
