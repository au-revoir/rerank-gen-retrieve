import torch
import torch.nn as nn

from transformers import AutoModelForSeq2SeqLM

class T0(nn.Module):

    def __init__(self, model_name, tokenizer, max_length=768, sync_gpus=True):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, 
                                                           torch_dtype=torch.bfloat16, 
                                                           cache_dir='/data/local/gg676/')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sync_gpus = sync_gpus

    def forward(self, input_ids, attention_mask):
        output = self.model.generate(input_ids=input_ids, 
                                     attention_mask=attention_mask,
                                     max_length=self.max_length)
        decoded_output = self.tokenizer.decode(output, skip_special_tokens=True)
        return decoded_output
