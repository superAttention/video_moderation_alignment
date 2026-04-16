"""
DPO trainer using Tinker's forward_backward_custom().

Unlike SFT (cross-entropy on one sequence) or RL (reward signal from rollouts),
DPO computes loss from log prob ratios between policy and reference model
on preference pairs — no sampling needed.

DPO loss (Rafailov et al. 2023):
    L = -E[ log σ( β * (log π(y_w|x) - log π_ref(y_w|x))
                     - β * (log π(y_l|x) - log π_ref(y_l|x)) ) ]

Where:
    y_w = chosen completion
    y_l = rejected completion
    π   = policy (model being trained)
    π_ref = reference model (frozen)
    β   = KL penalty coefficient (config.beta)
"""

"""
Basically for a single prompt, in the dataset we have the good 
good response and the bad response, we use the reference model and
the current model to calculate the probability of getting the good response and the 
bad response, by actively choosing the token that match the response
and calculate the joint probability of getting that reponse.
We try to maximise the difference between the probabilty of good response - the bad resopnse
"""

import torch
import torch.nn.functional as F
from tinker import TrainingClient, types
from configs.dpo_config import DPOConfig
from data.preference_dataset import PreferenceDataset
from .base_trainer import BaseTrainer


class DPOTrainer(BaseTrainer):

    def __init__(
        self,
        training_client: TrainingClient,
        ref_client: TrainingClient,     # frozen reference model
        config: DPOConfig,
        train_dataset: PreferenceDataset,
        val_dataset: PreferenceDataset | None = None,
    ):
        super().__init__(training_client, config)
        self.ref = ref_client
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    async def get_log_probs(self, client: TrainingClient, data: list[types.Datum]) -> torch.Tensor:
        """
        Run a forward pass (no grad) and return per-sequence log probs.
        Uses client.forward() — not forward_backward().
        Returns shape (B,).
        """


    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,      # (B,)
        policy_rejected_logps: torch.Tensor,    # (B,)
        ref_chosen_logps: torch.Tensor,         # (B,)
        ref_rejected_logps: torch.Tensor,       # (B,)
    ) -> torch.Tensor:
        """
        Compute scalar DPO loss.
        Implement the formula from the module docstring.
        """
        raise NotImplementedError

    async def train(self) -> None:
        for batch in self.train_dataset.batches(batch_size=8):
            # Get log probs from policy and reference for both chosen and rejected
            policy_chosen_logps   = await self.get_log_probs(self.tc,  batch["chosen"])
            policy_rejected_logps = await self.get_log_probs(self.tc,  batch["rejected"])
            ref_chosen_logps      = await self.get_log_probs(self.ref, batch["chosen"])
            ref_rejected_logps    = await self.get_log_probs(self.ref, batch["rejected"])

            loss = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            )

            await self.tc.forward_backward_custom_async(loss=loss)
            await self.tc.optim_step_async(types.AdamParams(learning_rate=self.config.learning_rate))

            self.global_step += 1
            if self.global_step % self.config.log_every == 0:
                self.log({"loss": loss.item()})
            if self.global_step % self.config.save_every == 0:
                await self.save_checkpoint()
