"""
Reward functions for RL training.

In Tinker's pattern, rewards are computed inside Env.step(action).
These are standalone scoring functions you can call from within step().
They are intentionally decoupled from the Env so they can be composed
or reused across different environments.
"""
from abc import ABC, abstractmethod


class RewardFn(ABC):
    """
    Stateless scoring function: takes a prompt + completion, returns a scalar.
    Compose these inside your Env.step() implementation.
    """

    @abstractmethod
    def score(self, prompt: str, completion: str) -> float:
        ...

    def __call__(self, prompt: str, completion: str) -> float:
        return self.score(prompt, completion)


# ------------------------------------------------------------------
# Concrete stubs — implement score() for your task
# ------------------------------------------------------------------

class ExactMatchReward(RewardFn):
    """1.0 if completion contains the reference answer, else 0.0."""

    def __init__(self, reference: str):
        self.reference = reference

    def score(self, prompt: str, completion: str) -> float:
        raise NotImplementedError


class LengthReward(RewardFn):
    """Toy: reward proportional to completion length. Useful for smoke-testing."""

    def score(self, prompt: str, completion: str) -> float:
        raise NotImplementedError
