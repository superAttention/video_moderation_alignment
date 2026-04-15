"""
Base trainer skeleton for Tinker-backed training.

Tinker replaces: model weights, optimizer, forward/backward, sampling.
You implement: data pipeline, loss selection, training loop, evaluation.
"""
from abc import ABC, abstractmethod
from tinker import TrainingClient, SamplingClient


class BaseTrainer(ABC):

    def __init__(self, training_client: TrainingClient, config):
        self.tc = training_client
        self.config = config
        self.tokenizer = training_client.get_tokenizer()
        self.global_step = 0

    @abstractmethod
    async def train(self) -> None:
        """Run the full training loop."""
        ...

    def evaluate(self) -> dict:
        """Return eval metrics. Override in subclass."""
        return {}

    async def save_checkpoint(self, tag: str = "") -> None:
        name = tag or f"step_{self.global_step}"
        await self.tc.save_state(name=name)

    def log(self, metrics: dict) -> None:
        parts = [f"step={self.global_step}"] + [f"{k}={v:.4f}" for k, v in metrics.items()]
        print(" | ".join(parts))
