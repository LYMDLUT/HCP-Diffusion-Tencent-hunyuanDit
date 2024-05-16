import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPooling

class MT5Embedder(nn.Module):
    available_models = ["t5-v1_1-xxl"]

    def __init__(
        self,
        tokenizer, model, generation_model,
        model_kwargs=None,
        torch_dtype=None,
        use_tokenizer_only=False,
        conditional_generation=False,
        max_length=128,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.max_length = max_length
        if model_kwargs is None:
            model_kwargs = {
                # "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
            }
        model_kwargs["device_map"] = {"shared": self.device, "encoder": self.device}
        self.tokenizer = tokenizer
        self.model = model
        self.generation_model = generation_model
        self.config = model.config

    
    @classmethod
    def from_pretrained(cls, model_dir="t5-v1_1-xxl", conditional_generation=False, torch_dtype=torch.float16, subfolder=None, max_length=256, **model_kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_dir,subfolder=subfolder)
        generation_model = None
        if conditional_generation:
            model = None
            generation_model = T5ForConditionalGeneration.from_pretrained(
                model_dir,subfolder=subfolder
            )
            return
        model = T5EncoderModel.from_pretrained(model_dir,subfolder=subfolder, **model_kwargs).eval().to(torch_dtype)
        return MT5Embedder(tokenizer, model, generation_model, model_kwargs, torch_dtype=torch_dtype, max_length=max_length)
    
    @classmethod
    def from_config(cls, model_dir="t5-v1_1-xxl", conditional_generation=False, torch_dtype=torch.float16, subfolder=None, max_length=256, **model_kwargs):
        tokenizer = AutoTokenizer.from_config(model_dir,subfolder=subfolder)
        if conditional_generation:
            model = None
            generation_model = T5ForConditionalGeneration.from_config(
                model_dir,subfolder=subfolder
            )
            return
        model = T5EncoderModel.from_config(model_dir,subfolder=subfolder, **model_kwargs).eval().to(torch_dtype)
        return MT5Embedder(tokenizer, model, generation_model, model_kwargs, torch_dtype=torch_dtype, max_length=max_length)

    def get_tokens_and_mask(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        tokens = text_tokens_and_mask["input_ids"][0]
        mask = text_tokens_and_mask["attention_mask"][0]
        # tokens = torch.tensor(tokens).clone().detach()
        # mask = torch.tensor(mask, dtype=torch.bool).clone().detach()
        return tokens, mask

    def get_text_embeddings(self, texts, attention_mask=True, layer_index=-1):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=text_tokens_and_mask["input_ids"].to(self.device),
                attention_mask=text_tokens_and_mask["attention_mask"].to(self.device)
                if attention_mask
                else None,
                output_hidden_states=True,
            )
            text_encoder_embs = outputs["hidden_states"][layer_index].detach()

        return text_encoder_embs, text_tokens_and_mask["attention_mask"].to(self.device)

    @torch.no_grad()
    def forward(self, tokens, attention_mask, position_ids=None, output_attentions=None, 
                 output_hidden_states=None, return_dict=None,layer_index=-1):
        with torch.cuda.amp.autocast():
            outputs = self.model(
                input_ids=tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=return_dict
            )
        
        return BaseModelOutputWithPooling(
                    last_hidden_state=outputs.last_hidden_state,
                    pooler_output=None,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
    
    def get_input_embeddings(self):
        return self.model.shared
    
    def general(self, text: str):
        # input_ids = input_ids = torch.tensor([list(text.encode("utf-8"))]) + num_special_tokens
        input_ids = self.tokenizer(text, max_length=128).input_ids
        print(input_ids)
        outputs = self.generation_model(input_ids)
        return outputs