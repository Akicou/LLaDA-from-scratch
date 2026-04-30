from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch

from configs_llada import (
    ActivationType,
    BlockType,
    InitFnType,
    LLaDAConfig,
    LayerNormType,
    ModelConfig,
)


DEFAULT_DATASET_PATHS = (
    "/teamspace/studios/this_studio/data_train_en/datasets--Fredtt3--LLaDA-Sample-10BT/snapshots/ee6dbc7d4bf1e4b2d0974e48f5fcb8b62b1f27f4",
    "/teamspace/studios/this_studio/data_train_es/datasets--Fredtt3--LLaDA-Sample-ES/snapshots/1f3128a94b4ff7e8d96892052704529e720f6b58",
)


@dataclass
class TrainingSettings:
    dataset_paths: Sequence[str] = field(default_factory=lambda: DEFAULT_DATASET_PATHS)
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    learning_rate: float = 4e-4
    weight_decay: float = 0.1
    total_steps: int = 50_000
    warmup_steps: int = 2_000
    log_every: int = 100
    save_every: int = 500
    output_dir: str = "checkpoints"
    tokenizer_name: str = "GSAI-ML/LLaDA-8B-Instruct"

    def checkpoint_path(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


MODEL_100M_KWARGS = dict(
    d_model=768,
    n_heads=12,
    n_layers=14,
    n_kv_heads=12,
    mlp_ratio=4,
    mlp_hidden_size=3072,
    max_sequence_length=4096,
    vocab_size=126464,
    mask_token_id=126336,
    eos_token_id=126081,
    pad_token_id=126081,
    layer_norm_type=LayerNormType.rms,
    rms_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    embedding_dropout=0.0,
    embedding_size=126464,
    block_type=BlockType.llama,
    block_group_size=1,
    attention_layer_norm=False,
    attention_layer_norm_with_affine=True,
    rope=True,
    rope_full_precision=True,
    rope_theta=500000.0,
    precision="bf16",
    weight_tying=False,
    init_fn=InitFnType.mitchell,
    init_std=0.02,
    activation_type=ActivationType.swiglu,
    alibi=False,
    alibi_bias_max=8.0,
)


def detect_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def build_model_config(device: str) -> ModelConfig:
    return ModelConfig(**MODEL_100M_KWARGS, init_device=device)


def build_hf_config(device: str) -> LLaDAConfig:
    return LLaDAConfig(**MODEL_100M_KWARGS, init_device=device)
