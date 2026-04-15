"""
SFT dataset: yields tinker.types.Datum objects ready for forward_backward().

Each Datum contains:
  - model_input: tokenized prompt + completion
  - loss_fn_inputs: weights (0=prompt, 1=completion) and target_tokens
"""
from tinker import types


class SFTDataset:
    """
    Iterable dataset that yields batches of types.Datum for SFT training.

    Usage:
        dataset = SFTDataset(data_path, tokenizer, max_seq_len)
        for batch in dataset.batches(batch_size=8):
            await training_client.forward_backward_async(data=batch, loss_fn="cross_entropy")
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def make_datum(self, prompt: str, completion: str) -> types.Datum:
        """
        Tokenize one (prompt, completion) pair and return a types.Datum.

        Weights: 0 on prompt tokens (no loss), 1 on completion tokens (loss).
        """
        raise NotImplementedError

    def batches(self, batch_size: int) -> list[list[types.Datum]]:
        """Yield batches of Datum objects."""
        raise NotImplementedError
