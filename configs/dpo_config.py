from dataclasses import dataclass


@dataclass
class DPOConfig:
    # Tinker model identifier
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    lora_rank: int = 32

    # DPO
    beta: float = 0.1       # KL penalty coefficient — higher = stay closer to reference

    # Data
    train_data_path: str = "data/dpo_pairs.jsonl"
    val_data_path: str = "data/val.jsonl"
    max_seq_len: int = 512
    batch_size: int = 4

    # Optimization
    learning_rate: float = 5e-5
    num_epochs: int = 1

    # Logging / checkpointing
    log_path: str = "~/logs/dpo"
    log_every: int = 10
    save_every: int = 50
    eval_every: int = 25
    output_dir: str = "outputs/dpo"
