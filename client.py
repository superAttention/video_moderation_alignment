"""
Tinker client setup.

Reads TINKER_API_KEY from the environment (loaded from .env).
Returns (training_client, sampling_client, tokenizer) ready to use.
"""
import tinker
from tinker import TrainingClient, SamplingClient
from transformers import AutoProcessor


async def create_training_client(model_name: str, lora_rank: int) -> TrainingClient:
    service_client = tinker.ServiceClient()
    return await service_client.create_lora_training_client_async(
        base_model=model_name,
        rank=lora_rank,
    )


async def get_sampling_client(training_client: TrainingClient, checkpoint_name: str = "") -> SamplingClient:
    """
    Either save current weights and spin up a fresh sampler,
    or attach to an existing checkpoint by name.
    """
    if checkpoint_name:
        return await training_client.create_sampling_client_async(checkpoint_name)
    return await training_client.save_weights_and_get_sampling_client_async(name="latest")


async def create_eval_sampling_client(model_name: str, model_path: str = "") -> SamplingClient:
    """
    Create a sampling client for evaluation — no training run created.

    Args:
        model_name: base model identifier (used when model_path is empty)
        model_path: Tinker checkpoint path (e.g. "tinker://uuid/weights/step_500")
    """
    service_client = tinker.ServiceClient()
    if model_path:
        return await service_client.create_sampling_client_async(model_path=model_path)
    return await service_client.create_sampling_client_async(base_model=model_name)


def load_processor(model_name: str) -> AutoProcessor:
    """
    Load the Qwen3-VL processor locally from HuggingFace.
    Handles vision + text token combination — Tinker only provides the tokenizer.
    """
    return AutoProcessor.from_pretrained(model_name)
