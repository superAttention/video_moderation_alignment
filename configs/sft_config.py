from dataclasses import dataclass


@dataclass
class SFTConfig:
    # Tinker model identifier (see https://tinker-docs.thinkingmachines.ai/tinker/models/)
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32

    # Loss function — "cross_entropy" | "importance_sampling" | "custom"
    loss_fn: str = "cross_entropy"

    # Data
    train_data_path: str = ""
    val_data_path: str = ""
    max_seq_len: int = 512

    # Optimization (Tinker uses AdamW internally via types.AdamParams)
    learning_rate: float = 1e-4
    num_epochs: int = 1

    # Logging / checkpointing
    log_path: str = "~/logs/sft"
    save_every: int = 20     # steps between checkpoint saves
    eval_every: int = 10     # steps between evaluations
