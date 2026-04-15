"""
Tinker RL environment interface.

Each Env represents one episode: a single prompt that receives one completion
and returns a scalar reward. Subclass this for your task.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StepResult:
    reward: float
    done: bool
    info: dict


class BaseEnv(ABC):
    """
    Minimal stateful environment for one RL episode.

    Tinker RL loop:
      obs = env.initial_observation()   # the prompt
      result = env.step(completion)     # model's response → reward
    """

    @abstractmethod
    def initial_observation(self) -> str:
        """Return the prompt shown to the model."""
        ...

    @abstractmethod
    def step(self, action: str) -> StepResult:
        """
        Receive the model's completion and return a reward.

        Args:
            action: decoded completion string from the model
        Returns:
            StepResult with reward scalar and done flag
        """
        ...
