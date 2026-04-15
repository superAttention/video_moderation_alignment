"""
Tinker client setup.

Reads TINKER_API_KEY from the environment (loaded from .env).
Returns (training_client, sampling_client, tokenizer) ready to use.
"""
import os
import tinker
from tinker import TrainingClient, SamplingClient


def create_training_client(model_name: str, lora_rank: int) -> TrainingClient:
    service_client = tinker.ServiceClient()
    return service_client.create_lora_training_client(
        base_model=model_name,
        rank=lora_rank,
    )


def get_sampling_client(training_client: TrainingClient, checkpoint_name: str = "") -> SamplingClient:
    """
    Either save current weights and spin up a fresh sampler,
    or attach to an existing checkpoint by name.
    """
    if checkpoint_name:
        return training_client.create_sampling_client(checkpoint_name)
    return training_client.save_weights_and_get_sampling_client(name="latest")
