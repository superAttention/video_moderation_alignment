from dataclasses import dataclass


@dataclass
class SFTConfig:
    # Tinker model identifier (see https://tinker-docs.thinkingmachines.ai/tinker/models/)
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    lora_rank: int = 32

    # Loss function — "cross_entropy" | "importance_sampling" | "custom"
    loss_fn: str = "cross_entropy"

    # Data
    categories: tuple = ("1_Violent_Crimes", "10_Hate", "12_Sexual_Content")
    train_data_path: str = "data/train.jsonl"
    val_data_path: str = "data/val.jsonl"
    max_seq_len: int = 512

    # Optimization (Tinker uses AdamW internally via types.AdamParams)
    learning_rate: float = 1e-4
    num_epochs: int = 1

    # Data loading
    batch_size: int = 4

    # Logging / checkpointing
    log_path: str = "~/logs/sft"
    log_every: int = 5       # steps between loss prints
    save_every: int = 20     # steps between checkpoint saves
    eval_every: int = 10     # steps between evaluations
