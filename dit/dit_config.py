from dataclasses import dataclass
@dataclass
class DitConfig:
    model_type = 'DiT-g/2'
    image_size = [1024, 1024]
    learn_sigma = None
    text_states_dim = 1024
    text_states_dim_t5 = 2048
    text_len = 77
    text_len_t5 = 256
    norm = "layer"
    infer_mode = "torch" # "fa", "torch", "trt"
    use_fp16 = True
    
    