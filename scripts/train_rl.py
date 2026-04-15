"""Entry point for RL training (GRPO / PPO)."""
import asyncio
from dotenv import load_dotenv

load_dotenv()

from configs.rl_config import RLConfig
from client import create_training_client, get_sampling_client
from data.rl_dataset import RLDataset
from trainers.rl_trainer import RLTrainer

# TODO: import your concrete Env subclass here
# from envs.my_task_env import MyTaskEnv


async def main():
    config = RLConfig()

    training_client = create_training_client(config.model_name, config.lora_rank)
    sampling_client = get_sampling_client(training_client)

    # TODO: load prompts and plug in your Env subclass
    prompts: list[str] = []
    env_cls = None  # replace with MyTaskEnv

    rl_dataset = RLDataset(prompts, env_cls, num_generations=config.num_generations)

    trainer = RLTrainer(
        training_client=training_client,
        sampling_client=sampling_client,
        config=config,
        rl_dataset=rl_dataset,
    )
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
