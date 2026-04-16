"""
SFT trainer using Tinker's TrainingClient.

Tinker handles: weight storage, forward/backward, AdamW optimizer.
You implement: dataset batching, epoch loop, evaluation.
"""
from tinker import TrainingClient, types
from configs.sft_config import SFTConfig
from data.sft_dataset import SFTDataset
from .base_trainer import BaseTrainer


class SFTTrainer(BaseTrainer):

    def __init__(
        self,
        training_client: TrainingClient,
        config: SFTConfig,
        train_dataset: SFTDataset,
        val_dataset: SFTDataset | None = None,
    ):
        super().__init__(training_client, config)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    async def train(self) -> None:
        adam_params = types.AdamParams(learning_rate=self.config.learning_rate)

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            for batch in self.train_dataset.batches(self.config.batch_size):
                output = await (await self.tc.forward_backward_async(batch, self.config.loss_fn))
                await (await self.tc.optim_step_async(adam_params))
                self.global_step += 1

                if self.global_step % self.config.log_every == 0:
                    self.log(output.metrics)

                if self.val_dataset and self.global_step % self.config.eval_every == 0:
                    val_metrics = await self.evaluate()
                    print(f"  val: {val_metrics}")

                if self.global_step % self.config.save_every == 0:
                    await self.save_checkpoint()

        await self.save_checkpoint(tag="final")

    async def evaluate(self) -> dict:
        """
        Run forward (no grad) over val_dataset.
        Returns {"val_loss": float, "val_ppl": float}.
        """
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_dataset.batches(self.config.batch_size):
            output = await (await self.tc.forward_async(batch, self.config.loss_fn))
            # Tinker reports loss as "loss:sum" — divide by batch size for per-example loss
            batch_loss = output.metrics.get("loss:sum", 0.0) / len(batch)
            total_loss += batch_loss
            n_batches += 1

        if n_batches == 0:
            return {}

        avg_loss = total_loss / n_batches
        return {"val_loss": avg_loss, "val_ppl": float(2 ** avg_loss)}
