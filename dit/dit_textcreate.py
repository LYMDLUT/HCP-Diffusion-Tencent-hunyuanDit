from .dit_textencoder import DiTComposeTextEncoder
from transformers import BertModel
from .text_encoder import MT5Embedder
import torch
class DiTTextEncoder(DiTComposeTextEncoder):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, subfolder1=None, subfolder2=None, revision:str=None, **kwargs):
        clip = BertModel.from_pretrained(pretrained_model_name_or_path, False, subfolder=subfolder1, revision=None)
        mt5 = MT5Embedder.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16, subfolder=subfolder2, max_length=256)
        return cls([('clip', clip), ('mt5', mt5)])
    @classmethod
    def from_config(cls, pretrained_model_name_or_path: str, *args, subfolder1=None, subfolder2=None, revision:str=None, **kwargs):
        clip = BertModel.from_config(pretrained_model_name_or_path, False,subfolder=subfolder1, revision=None)
        mt5 = MT5Embedder.from_config(pretrained_model_name_or_path, torch_dtype=torch.float16, subfolder=subfolder2, max_length=256)
        return cls([('clip', clip), ('mt5', mt5)])