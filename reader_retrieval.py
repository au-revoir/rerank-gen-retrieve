
import argparse
import os
import deepspeed
from tqdm import tqdm
import torch
import torch.nn as nn

from data import FeverDataset
from pathlib import Path
from transformers import T5Tokenizer, set_seed
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers.deepspeed import HfDeepSpeedConfig
from torch.utils.data import DataLoader
from model import T0
from util import load_jsonl, load_json, save_jsonl, save_json, get_ds_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def read(args):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "2"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    #print("World size: ", world_size)
    config = AutoConfig.from_pretrained(args.model_name)
    model_hidden_size = config.d_model
    train_batch_size = 1 * world_size
    ds_config = get_ds_config(model_hidden_size, train_batch_size)
    dschf = HfDeepSpeedConfig(ds_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="/data/local/gg676/")
    #model = T0(args.model_name, tokenizer)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                                              torch_dtype=torch.bfloat16,
                                                              cache_dir='/data/local/gg676/')

    #print("max token length: ", model.config.max_position_embeddings)
    #quit()
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config, 
                                      model_parameters=None, optimizer=None, 
                                      lr_scheduler=None)[0]
    ds_engine.eval()
     #tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="/data/local/gg676/")
    
    #path = os.path.join(args.output_dir, args.data_split, 'generate', 'merged_results.json')
    extension = Path(args.data_path).suffix
    print("Extension: ", extension)
    if extension == '.json':
        data = load_json(args.data_path)
    elif extension == '.jsonl':
        data = load_jsonl(args.data_path)
    #print(fever_data[0])
    if args.document_type == 'gpt':
        data = data[1:]
    prompts = load_json(args.prompts_path)

    dataset = FeverDataset(data, args.data_split, prompts, version=args.prompt_version, document_type=args.document_type, mode="read", concat_top_k=args.concat_top_k)
    #print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=world_size, shuffle=False, 
                             num_workers=args.num_workers, collate_fn=lambda x: x)
     
    rank = torch.distributed.get_rank()

    generated_text_list = []

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        inconsistent_result = None
        current_batch_size = len(batch)
        if current_batch_size < world_size:
            batch = batch * world_size
        #claim_input_ids, claim_attn_ids = tokenizer.batch_encode_plus(batch[rank]['text'], max_length=512, 
        #                                                              truncation=True, return_tensors='pt')  
        
        #print("reader text: ", batch[rank]['text'])
        #quit()
        inputs_encoded = tokenizer.encode(batch[rank]['text'], max_length=1024, padding='longest', truncation=True, return_tensors="pt").to(rank)
        #print("inputs_encoded: ", inputs_encoded)
        with torch.no_grad():    
            output = ds_engine.module.generate(inputs_encoded, max_length=10, synced_gpus=True)
            #print("output: ", output)
        output_decoded = tokenizer.decode(output[0], skip_special_tokens=True).lower()
        #print("output decoded: ", output_decoded)
        #quit()
        """
        if output_decoded == "true":
            decoded_label = "SUPPORTS"
        elif output_decoded == "false":
            decoded_label = "REFUTES"
        else:
            inconsistent_result = {"claim": batch[rank]["claim"], 
                                   "gold_context": batch[rank]["generated_context"], 
                                   "predicted_label": decoded_label}
        """
        result = {"input": batch[rank]["input"], 
                  "context_type": args.document_type,
                  "context": batch[rank]["context"], 
                  "input_with_prompt": batch[rank]["text"],
                  "output": output_decoded,
                  "answer": batch[rank]["answer"]}
        
        generated_text_list.append(result)
        #print(result)
    if args.document_type == 'rerankt0pp_concat':
        output_path = f'{args.output_dir}/{args.dataset}/{args.data_split}/reader/doctype_{args.document_type}_concatenatedtop{args.concat_top_k}/'
    else:
        output_path = f'{args.output_dir}/{args.dataset}/{args.data_split}/reader/doctype_{args.document_type}/'
    save_json(generated_text_list, output_path, rank) 
                                  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bigscience/T0pp')
    #parser.add_argument('--base_path', type=str, default='/data/local/gg676/KILT/artifacts')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--document_type', type=str, help='gpt, oracle, genread_bm25, bm25, t0pp, t5-11b', required=True )
    parser.add_argument('--concat_top_k', type=int, help='1,2,3,5 etc')
    parser.add_argument('--data_split', type=str, help='dev, test', required=True)
    parser.add_argument('--dataset', type=str, help='fever, fm2', required=True)
    parser.add_argument('--data_path', type=str, required=True) 
    parser.add_argument('--prompts_path', type=str, default='prompts.json')
    parser.add_argument('--prompt_version', type=str, required=True, help="fact_verification, question_answering")
    parser.add_argument('--output_dir', type=str, default='/data/local/gg676/ACL/outputs')#/data/local/gg676/KILT/oracle/fever/results/retrieval/t0pp')#/data/local/gg676/KILT/oracle/fever/results/retrieval/bm25')
    #parser.add_argument('--dev_data', type=str, default='/common/home/gg676/NLP/KILT/data/dev.pkl')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    read(args)
