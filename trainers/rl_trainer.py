"""
RL trainer using Tinker's TrainingClient + SamplingClient.

Follows TRL/veRL convention: reward_fns is a list of callables passed at
construction time. The trainer calls each fn on every batch of completions
and sums the scores, making rewards composable without coupling them to
data or environment classes.

Per-step loop:
  1. rollout    — generate num_generations completions per prompt (SamplingClient)
  2. reward     — call each reward_fn, sum scores across fns
  3. advantages — normalize within group (GRPO) or compute GAE (PPO)
  4. assemble   — convert to types.Datum with importance-sampled weights
  5. update     — forward_backward_async → optim_step_async
"""
from tinker import TrainingClient, SamplingClient, types
from configs.rl_config import RLConfig
from data.prompt_dataset import PromptDataset
from rewards import RewardFn
from .base_trainer import BaseTrainer


class RLTrainer(BaseTrainer):

    def __init__(
        self,
        training_client: TrainingClient,
        sampling_client: SamplingClient,
        config: RLConfig,
        prompt_dataset: PromptDataset,
        reward_fns: list[RewardFn],     # composable, summed at runtime
    ):
        super().__init__(training_client, config)
        self.sc = sampling_client
        self.prompt_dataset = prompt_dataset
        self.reward_fns = reward_fns

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    async def rollout(self, prompts: list[str]) -> dict:
        """
        Generate num_generations completions per prompt via SamplingClient.

        Returns:
            {
                "prompts":      list[str]       shape (B,)
                "completions":  list[list[str]] shape (B, G)
                "log_probs":    ...             shape (B, G, T)
            }
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Reward scoring
    # ------------------------------------------------------------------

    def score(self, prompts: list[str], completions: list[list[str]]) -> list[list[float]]:
        """
        Call each reward_fn on every (prompt, completion) pair and sum scores.

        Args:
            prompts:     (B,)
            completions: (B, G)
        Returns:
            rewards: (B, G) — sum of all reward_fn outputs
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    def compute_advantages(self, rewards: list[list[float]]) -> list[list[float]]:
        """
        GRPO: subtract group mean, divide by group std (per prompt).
        PPO:  GAE over the trajectory.

        Args:
            rewards: (B, G)
        Returns:
            advantages: (B, G)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Training data assembly
    # ------------------------------------------------------------------

    def assemble_training_data(
        self,
        rollout: dict,
        advantages: list[list[float]],
    ) -> list[types.Datum]:
        """
        Convert rollout + advantages into types.Datum objects for
        forward_backward_async. Uses importance_sampling or PPO loss
        depending on config.algorithm.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    async def train(self) -> None:
        for step, batch in enumerate(self.prompt_dataset.batches(1)):
            rollout = await self.rollout(batch["prompts"])
            rewards = self.score(rollout["prompts"], rollout["completions"])
            advantages = self.compute_advantages(rewards)
            data = self.assemble_training_data(rollout, advantages)

            await self.tc.forward_backward_async(data=data, loss_fn=self.config.algorithm)
            await self.tc.optim_step_async(types.AdamParams(learning_rate=self.config.learning_rate))

            self.global_step += 1
            if self.global_step % self.config.log_every == 0:
                flat_rewards = [r for group in rewards for r in group]
                self.log({"reward_mean": sum(flat_rewards) / len(flat_rewards)})
            if self.global_step % self.config.save_every == 0:
                await self.save_checkpoint()
            if self.global_step >= self.config.num_steps:
                break
