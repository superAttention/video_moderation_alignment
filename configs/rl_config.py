from dataclasses import dataclass


@dataclass
class RLConfig:
    # Tinker model identifier (see https://tinker-docs.thinkingmachines.ai/tinker/models/)
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32

    # RL algorithm — "grpo" | "ppo" | "cispo"
    # Maps to Tinker loss: "cross_entropy" (importance-sampled) or built-in PPO/CISPO loss
    algorithm: str = "grpo"

    # Rollout
    num_generations: int = 8      # G in GRPO — completions sampled per prompt
    max_prompt_len: int = 256
    max_gen_len: int = 256
    temperature: float = 1.0

    # PPO / CISPO
    clip_eps: float = 0.2

    # GRPO
    group_norm: bool = True       # normalize rewards within group

    # KL penalty (applied in loss)
    kl_coef: float = 0.05

    # Optimization (Tinker uses AdamW internally via types.AdamParams)
    learning_rate: float = 1e-5
    num_steps: int = 1000

    # Logging / checkpointing
    log_path: str = "~/logs/rl"
    save_every: int = 50
    eval_every: int = 25
