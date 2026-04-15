"""
RL trainer using Tinker's TrainingClient + SamplingClient.

Tinker handles: weight storage, forward/backward, AdamW, text generation.
You implement: rollout orchestration, advantage computation, loss assembly, reward integration.

Per-step loop:
  1. rollout    — sample G completions per prompt via SamplingClient
  2. reward     — score completions via EnvGroup.step()
  3. advantages — normalize rewards within each group (GRPO) or compute GAE (PPO)
  4. assemble   — convert trajectories to types.Datum with log-prob weights
  5. update     — forward_backward_async → optim_step_async
"""
from tinker import TrainingClient, SamplingClient, types
from configs.rl_config import RLConfig
from data.rl_dataset import RLDataset
from .base_trainer import BaseTrainer


class RLTrainer(BaseTrainer):

    def __init__(
        self,
        training_client: TrainingClient,
        sampling_client: SamplingClient,
        config: RLConfig,
        rl_dataset: RLDataset,
    ):
        super().__init__(training_client, config)
        self.sc = sampling_client
        self.rl_dataset = rl_dataset

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    async def rollout(self, env_group) -> list[dict]:
        """
        Sample G completions for the group's prompt.
        Returns list of {"completion": str, "reward": float, "log_probs": ...}.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    def compute_advantages(self, rewards: list[float]) -> list[float]:
        """
        Convert raw rewards to advantages.
        GRPO: normalize within group (subtract mean, divide by std).
        PPO:  GAE over the trajectory.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Training data assembly
    # ------------------------------------------------------------------

    def assemble_training_data(
        self,
        rollout_results: list[dict],
        advantages: list[float],
    ) -> list[types.Datum]:
        """
        Convert rollout results + advantages into types.Datum objects
        for forward_backward_async.
        Uses importance_sampling or PPO loss depending on config.algorithm.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    async def train(self) -> None:
        """
        Main RL loop over rl_dataset:
          for each env_group:
            results   = await self.rollout(env_group)
            advantages = self.compute_advantages([r["reward"] for r in results])
            data      = self.assemble_training_data(results, advantages)
            await tc.forward_backward_async(data=data, loss_fn=...)
            await tc.optim_step_async(types.AdamParams(learning_rate=config.learning_rate))
        """
        raise NotImplementedError
