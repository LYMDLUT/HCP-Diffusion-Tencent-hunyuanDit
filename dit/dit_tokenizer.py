from hcpdiff.models.compose import ComposeTokenizer
from transformers import AutoTokenizer

class DiTTokenizer(ComposeTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, subfolder1=None, subfolder2=None, revision:str=None, **kwargs):
        clip = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder1, **kwargs)
        mt5 = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder2, **kwargs)
        return cls([('clip', clip), ('mt5', mt5)])