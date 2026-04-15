"""
Reward functions for RL training.

Industry standard (TRL/veRL): reward functions are standalone callables passed
to the trainer as a list. The trainer calls each one and sums (or averages) the
scores. This makes rewards composable and independently testable.

Signature matches TRL's convention:
    fn(prompts: list[str], completions: list[str]) -> list[float]
"""
from typing import Protocol


class RewardFn(Protocol):
    def __call__(self, prompts: list[str], completions: list[str]) -> list[float]:
        """
        Score a batch of (prompt, completion) pairs.

        Args:
            prompts:     batch of prompt strings, shape (B,)
            completions: batch of completion strings, shape (B,)
        Returns:
            list of scalar rewards, shape (B,)
        """
        ...


# ------------------------------------------------------------------
# Concrete stubs — implement __call__ for your task
# ------------------------------------------------------------------

class ExactMatchReward:
    """1.0 if completion contains the reference answer, else 0.0."""

    def __init__(self, references: list[str]):
        self.references = references

    def __call__(self, prompts: list[str], completions: list[str]) -> list[float]:
        raise NotImplementedError


class LengthReward:
    """Toy: reward proportional to completion length. Useful for smoke-testing."""

    def __call__(self, prompts: list[str], completions: list[str]) -> list[float]:
        raise NotImplementedError


class LLMJudgeReward:
    """
    Uses an external LLM (via OpenAI-compatible API) to score completions.
    Works with any OpenAI-compatible endpoint, including Tinker-hosted models.
    """

    def __init__(self, client, model: str, prompt_template: str):
        self.client = client    # openai.OpenAI instance
        self.model = model
        self.prompt_template = prompt_template  # must contain {prompt} and {completion}

    def __call__(self, prompts: list[str], completions: list[str]) -> list[float]:
        raise NotImplementedError
