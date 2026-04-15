"""
RL dataset: produces groups of environments for rollout.

Tinker RL pattern:
  - Env         — one episode (prompt → completions → reward)
  - EnvGroup    — G envs sharing a prompt, used for GRPO group normalization
  - RLDataset   — iterable of EnvGroups for the training loop
"""
from envs.base_env import BaseEnv


class EnvGroup:
    """
    A group of G environments sharing the same prompt.
    Rollout produces G completions; rewards are normalized within the group.

    Args:
        prompt: the shared prompt string
        num_envs: number of parallel rollouts (G in GRPO)
        env_cls: BaseEnv subclass to instantiate per slot
    """

    def __init__(self, prompt: str, num_envs: int, env_cls: type[BaseEnv]):
        raise NotImplementedError

    def rollout(self, sampling_client) -> list[dict]:
        """
        Generate completions for each env and collect rewards.
        Returns list of {"completion": str, "reward": float} dicts.
        """
        raise NotImplementedError


class RLDataset:
    """
    Iterable that yields EnvGroups for the RL training loop.

    Usage:
        dataset = RLDataset(prompts, env_cls, num_generations=8)
        for env_group in dataset:
            results = env_group.rollout(sampling_client)
            ...
    """

    def __init__(self, prompts: list[str], env_cls: type[BaseEnv], num_generations: int):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError
