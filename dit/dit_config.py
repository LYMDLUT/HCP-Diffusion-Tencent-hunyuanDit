from dataclasses import dataclass, field

@dataclass
class DitConfig:
    model_type: str = 'DiT-g/2'
    image_size: list = field(default_factory=lambda: [720, 720])
    learn_sigma: bool = True
    text_states_dim: int = 1024
    text_states_dim_t5: int = 2048
    text_len: int = 77
    text_len_t5: int = 256
    norm: str = "layer"
    infer_mode: str = "torch"  # "fa", "torch", "trt"

    