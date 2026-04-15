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
        """
        Epoch loop over train_dataset.
        Each step:
          1. Get a batch of types.Datum from dataset
          2. Call forward_backward_async with loss_fn from config
          3. Call optim_step_async with AdamParams
          4. Log, eval, and checkpoint on schedule
        """
        raise NotImplementedError

    async def evaluate(self) -> dict:
        """
        Run forward (no grad) over val_dataset.
        Returns {"val_loss": float, "val_ppl": float}.
        """
        raise NotImplementedError
