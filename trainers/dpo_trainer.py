"""
DPO trainer using Tinker's forward_backward_custom_async().

Unlike SFT (cross-entropy on one sequence), DPO computes loss from log prob
ratios between the policy and a frozen reference model on preference pairs.

DPO loss (Rafailov et al. 2023):
    L = -E[ log σ( β * (log π(y_w|x) - log π_ref(y_w|x))
                     - β * (log π(y_l|x) - log π_ref(y_l|x)) ) ]

How Tinker's forward_backward_custom_async works:
    1. Does a forward pass → produces per-token logprobs for each Datum
    2. Calls your loss_fn(data, logprobs_list) synchronously
    3. Calls loss.backward() to get ∂loss/∂logprobs
    4. Sends those gradients back to the server for the actual backward pass

For the reference model we use the same API with a zero-gradient dummy loss
to extract logprobs without updating weights.
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
        ref_client: TrainingClient,     # frozen reference model — never call optim_step on this
        config: DPOConfig,
        train_dataset: PreferenceDataset,
        val_dataset: PreferenceDataset | None = None,
    ):
        super().__init__(training_client, config)
        self.ref = ref_client
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    async def _extract_ref_logps(
        self, chosen: list[types.Datum], rejected: list[types.Datum]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Run a forward pass on the frozen reference model and return per-sequence
        log probs for chosen and rejected.

        We use forward_backward_custom_async with a dummy loss (gradient = 0) so
        that no weights are updated on the reference model.
        """
        combined = chosen + rejected
        B = len(chosen)
        captured: list[torch.Tensor] = []

        def extract_fn(data, logprobs_list):
            for datum, logprob in zip(data, logprobs_list):
                weights = torch.tensor(datum.loss_fn_inputs["weights"].data)
                # sum weighted log probs → scalar log p(response | prompt)
                captured.append((logprob * weights).sum().detach())
            # Zero-gradient dummy loss: depends on all logprobs so grad is 0, not None
            dummy_loss = sum(lp.sum() * 0.0 for lp in logprobs_list)
            return dummy_loss, {}

        # extract_fn is called synchronously inside the coroutine, so
        # `captured` is populated by the time this line returns.
        await (await self.ref.forward_backward_custom_async(combined, extract_fn))
        return captured[:B], captured[B:]

    async def train(self) -> None:
        adam_params = types.AdamParams(learning_rate=self.config.learning_rate)

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            for batch in self.train_dataset.batches(self.config.batch_size):
                chosen   = batch["chosen"]    # list[Datum], length B
                rejected = batch["rejected"]  # list[Datum], length B
                combined = chosen + rejected  # length 2B — policy processes all at once

                # Step 1: reference model log probs (no gradient, no weight update)
                ref_chosen_logps, ref_rejected_logps = await self._extract_ref_logps(chosen, rejected)

                ref_chosen   = torch.stack(ref_chosen_logps)    # (B,)
                ref_rejected = torch.stack(ref_rejected_logps)  # (B,)

                # Step 2: policy forward + backward with DPO loss
                def dpo_loss_fn(data, logprobs_list):
                    B = len(data) // 2

                    # Aggregate token-level logprobs → per-sequence log probs
                    def seq_logp(datum, logprob):
                        weights = torch.tensor(datum.loss_fn_inputs["weights"].data)
                        return (logprob * weights).sum()

                    policy_chosen   = torch.stack([seq_logp(data[i],   logprobs_list[i])   for i in range(B)])
                    policy_rejected = torch.stack([seq_logp(data[i+B], logprobs_list[i+B]) for i in range(B)])

                    # Log-ratio: how much did the policy change relative to reference?
                    chosen_ratio   = policy_chosen   - ref_chosen    # (B,)
                    rejected_ratio = policy_rejected - ref_rejected   # (B,)

                    loss = -F.logsigmoid(self.config.beta * (chosen_ratio - rejected_ratio)).mean()

                    return loss, {"dpo_loss": loss.item()}

                output = await (await self.tc.forward_backward_custom_async(combined, dpo_loss_fn))
                await (await self.tc.optim_step_async(adam_params))

                self.global_step += 1
                if self.global_step % self.config.log_every == 0:
                    self.log(output.metrics)
                if self.global_step % self.config.save_every == 0:
                    await self.save_checkpoint()

        await self.save_checkpoint(tag="final")
