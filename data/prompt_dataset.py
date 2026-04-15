"""
Prompt dataset for RL rollouts.

Yields batches of raw prompt strings. The trainer handles tokenization,
generation, and reward scoring — the dataset has no knowledge of rewards.
"""


class PromptDataset:
    """
    Iterable dataset of prompts for RL training.

    Usage:
        dataset = PromptDataset(data_path, tokenizer, max_prompt_len)
        for batch in dataset.batches(batch_size=8):
            # batch: {"prompts": list[str], "input_ids": Tensor}
            rollouts = await trainer.rollout(batch)
            ...
    """

    def __init__(self, data_path: str, tokenizer, max_prompt_len: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def batches(self, batch_size: int):
        """Yield dicts with keys: prompts (list[str]), input_ids (Tensor)."""
        raise NotImplementedError
