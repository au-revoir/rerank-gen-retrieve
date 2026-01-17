import torch
import torch.nn as nn
import torch.distributed as dist

from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

class T0(nn.Module):
    
    def __init__(self, model_name, sync_gpus):
        super().__init__()
        #config = T5Config.from_pretrained(t5_name)
        #self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name, config=config, cache_dir='/data/gg676/')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, 
                                                           torch_dtype=torch.bfloat16, 
                                                           cache_dir='/data/local/gg676/KILT/')
        self.sync_gpus = sync_gpus

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask,
                            labels=labels)
        current_gpu_finished = False
        while True:
            if self.sync_gpus:
                current_gpu_job_finished_flag = torch.tensor(0.0 if current_gpu_finished else 1.0).cuda()
                dist.all_reduce(current_gpu_job_finished_flag, op=dist.ReduceOp.SUM)
                
                
                """                
                _ = self.model(input_ids, attention_mask=attention_mask, \
                            decoder_input_ids=decoder_input_ids, \
                            decoder_attention_mask=decoder_attention_mask, \
                             labels=labels)
                """
                if current_gpu_job_finished_flag.item() == 0.0:
                    break
                if output.logits.shape[0] == input_ids.shape[0]:
                    current_gpu_finished = True

            else:
                break
        return output

