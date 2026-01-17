import argparse
import os
import pandas as pd
import deepspeed
from tqdm import tqdm
import torch
import torch.nn as nn


from data import load_data, save_data, FeverDataset

from operator import itemgetter
from transformers import T5Tokenizer, set_seed
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers.deepspeed import HfDeepSpeedConfig
from torch.utils.data import DataLoader
from model import T0 
from tqdm import tqdm
from util import load_json, save_jsonl, save_json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def inference(local_rank, model, encoder_input_ids, encoder_attn_mask, decoder_input_ids):

    #print("local rank: ", local_rank,"encoder input: ", encoder_input_ids[0])
    with torch.no_grad():
        logits = model.module(input_ids=encoder_input_ids, attention_mask=encoder_attn_mask, \
                        labels=decoder_input_ids).logits
       #break
    #print("logits: ", logits.shape)
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    #print("log softmax shape: ", log_softmax)
    #torchprint("decoder input ids: ", decoder_input_ids.shape, "--> ", decoder_input_ids.unsqueeze(2).shape)
    nll = -log_softmax.gather(2, decoder_input_ids.unsqueeze(2)).squeeze(2)

    avg_nll = torch.sum(nll, dim=1)
    #sharded_nll_list.append(avg_nll)
    #print("local rank: ", local_rank, " avg nll length --> ", len(avg_nll))
    return avg_nll

def preprocess(rank, model, tokenizer, batch):

    #print("batch[0]: ", batch_paragraphs[0])
    #print("batch[1]: ", batch_paragraphs[1])
    #print("TYPE batch[1]: ", type(batch_paragraphs))
    #print("batch CLAIM: ", batch_claim)
    #paragraphs = [i[0] for i in batch_paragraphs]
    #print("local rank: ", local_rank,"paragraphs: ", batch_paragraphs[0], len(batch_paragraphs))
    paragraph_encoding = tokenizer.batch_encode_plus(batch[rank]['paragraphs'], padding='longest', max_length=768, \
                                                    truncation=True, return_tensors='pt')
    paragraph_input_ids, paragraph_attn_mask = paragraph_encoding.input_ids, paragraph_encoding.attention_mask

    #claim_repeated = torch.repeat_interleave(batch['claim'], len(batch['paragraphs']), dim=0)

    #paragraph_input_ids, paragraph_attn_mask = paragraph_input_ids.to(device=local_rank), paragraph_attn_mask.to(device=local_rank)
    #print("query_text: ", batch[rank]['query_text'])
    #print("paragraph text: ", batch[rank]['paragraphs'][0])
    #quit()

    query_encoded = tokenizer.batch_encode_plus([batch[rank]['query_text']], padding='longest',  max_length=768, \
                                                    truncation=True, return_tensors='pt')
    query_input_ids = query_encoded.input_ids

    query_input_ids_repeated = torch.repeat_interleave(query_input_ids, len(batch[rank]['paragraphs']), dim=0)
    #claim_input_ids_repeated = claim_input_ids_repeated.to(device=local_rank)

    sharded_nll_list = []
    #print("length paragraph input ids: ", len(paragraph_input_ids))
    for i in range(0, len(paragraph_input_ids), args.shard_size):
        encoder_input_ids = paragraph_input_ids[i: i + args.shard_size].to(device=rank)
        encoder_attn_mask = paragraph_attn_mask[i: i + args.shard_size].to(device=rank)
        decoder_input_ids = query_input_ids_repeated[i: i + args.shard_size].to(device=rank)

        sharded_nll_list.append(inference(rank, model, encoder_input_ids, encoder_attn_mask, decoder_input_ids))
    #print("length: ", len(sharded_nll_list))
    topk_scores, indexes = torch.topk(-torch.cat(sharded_nll_list), k=len(paragraph_input_ids))
    #print("local rank: ", local_rank," top k scores length: ", len(topk_scores))
    #return topk_scores, indexes
    #print("top 1: ", indexes[0], "--> top2: ", indexes[1], "--> top3: ", indexes[2])
    reordered_candidates = [batch[rank]['candidates'][i] for i in indexes]
    #reordered_candidates_desc = sorted(reordered_candidates, key=itemgetter('score'), reverse=True)
    #reordered_paragraphs_meta = [batch[rank]['paragraphs_meta'] for i in indexes]

    #reordered_candidates = []
    #for text, meta in zip(reordered_paragraphs, reordered_paragraphs_meta):
    #    reordered_candidates.append({''})

    row = {'question': batch[rank]['query'], 'context_type': 'UPR_background', 'query_text': batch[rank]['query_text'], 'answer': batch[rank]['answer'], 'candidates': reordered_candidates}
    return row

def main(args):

    # distributed setup
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "2"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    print("World size: ", world_size)

    #device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    config = AutoConfig.from_pretrained(args.model_name)
    model_hidden_size = config.d_model

    # batch size has to be divisible by world_size, but can be bigger than world_size
    train_batch_size = 1 * world_size

    reranked_data_list = []
    original_answers_list = []
    reranked_answers_list = []

    ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "none",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * 1,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * 1,
        "stage3_param_persistence_threshold": 1 * model_hidden_size
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
    }


    dschf = HfDeepSpeedConfig(ds_config)

    #model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir="/data/local/gg676/")
    model = T0(args.model_name, sync_gpus=True)

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config, model_parameters=None, optimizer=None, lr_scheduler=None)[0]
    #print("ds_engine: ", ds_engine)
    ds_engine.eval() 

    #prompt = "Explain?"
    provenance_data = load_json(args.data_path)
    #tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir='/data/gg676/')
    #print("\n test data: ", len(test_data))
    #print("\n dev data: ", dev_data)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="/data/local/gg676/")
    #print(tokenizer.encode("Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy", return_tensors="pt"))

    dataset = FeverDataset(provenance_data, args.verbalizer, args.verbalizer_head, args.rerank_type, tokenizer, max_len=768)
    print(len(provenance_data))
    #print("dataset test: ", provenance_data[0]['candidates'][0])
    #print("dataset: ", type(dataset[0]['paragraphs']))
    #for batxh_idx, batch in enumerate
    #x = tokenizer.batch_encode_plus(dataset[0]['paragraphs'], padding='longest', max_length=512, truncation=True, return_tensors='pt')
    #print(x['input_ids'])
    #quit()
    
    dataloader = DataLoader(dataset, batch_size=world_size, #args.batch_size, 
                              shuffle=False, num_workers=args.num_workers, collate_fn=lambda x: x)

    
    #tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="/data/local/gg676/")
    #model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir="/data/local/gg676/", torch_dtype=torch.bfloat16)
    #model.parallelize()
    #print(loader_test)    
    #model = T5Classifier(args.model_name)
    #model = torch.nn.DataParallel(model, device_ids=[0, 1,])
    #model.to(device)
    #model.load_state_dict(torch.load("/data/gg676/checkpoints_KILT/checkpoints_KILT/t5-base_best.pt".format(args.model_name)))
    #model.eval()

    #model.train()
    #dataloader_iter = iter(dataloader)
    rank = torch.distributed.get_rank()
    reranked_candidates_list = []

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        current_batch_size = len(batch)
        #print(batch[rank]['claim'], "--> ", batch[rank]['answer'], "--> ", batch[rank]['paragraph_meta'][:2])
        #quit()
        #print(type(list(zip(*batch['paragraphs']))[2]))
    #while True:
        #print("b claim: ", batch["claim"]) 
        #print("len batch b: ", len(batch))
        #print("len batch b[0]", len(batch['claim'][0]))
        #print("len batch b[1]", len(batch['claim'][1]))
        
        #print("zip list 0th: ", len(list(zip(*batch['paragraphs']))[0]))
        #print("zip list 1st: ", len(list(zip(*batch['paragraphs']))[1]))
        #print("zip list 2nd: ", len(list(zip(*batch['paragraphs']))[2]))
        
        #print("len batch paragraph b[0]", len(batch['paragraphs']))
        #print("len batch paragraph b[0]", batch['paragraphs'][0][0])
        #print("len batch paragraph b[1]", batch['paragraphs'][0][1])
        #print("len batch paragraph b[1]", batch['paragraphs'][0][2])
        

        #print("len batch paragraph b[1]", len(batch['paragraphs']))
        #print("len batch paragraph b[1]", batch['paragraphs'][1][0])
        #print("len batch paragraph b[1]", batch['paragraphs'][1][1])
        #quit()
        #print(batch.keys())
        if current_batch_size < world_size:
            remaining_examples = world_size - current_batch_size
            batch = batch * world_size

        result = preprocess(rank, ds_engine, tokenizer, batch)
        reranked_candidates_list.append(result)
        output_path = f'{args.output_dir}/{args.dataset}/{args.data_split}/rerankinput_{args.rerank_type}/'#top2rerankt0ppgpt/' 
        save_json(reranked_candidates_list, output_path, rank)
        """
        sharded_nll_list = []

        if batch_idx == 0:
            total_no_passages = len(list(list(zip(*batch['paragraphs']))[0]))
        if rank == 0:
            #batch = dataloader_iter.next()
            #print("batch ids: ", batch["paragraphs"][0], batch["claim"][0])
            sharded_nll_list.append(preprocess(rank, ds_engine, tokenizer, list(list(zip(*batch['paragraphs']))[0]), [batch["claim"][0]]))
        elif rank == 1:
            #batch = dataloader_iter.next()
            #print("batch ids: ", batch["id"])
            sharded_nll_list.append(preprocess(rank, ds_engine, tokenizer, list(list(zip(*batch['paragraphs']))[1]), [batch["claim"][1]]))
        elif rank == 2:
            #batch = dataloader_iter.next()
            #print("batch ids: ", batch["id"])
            sharded_nll_list.append(preprocess(rank, ds_engine, tokenizer, list(list(zip(*batch['paragraphs']))[2]), [batch["claim"][2]]))
        elif rank == 3:
            #batch = dataloader_iter.next()
            sharded_nll_list.append(preprocess(rank, ds_engine, tokenizer, list(list(zip(*batch['paragraphs']))[3]), [batch["claim"][3]]))

        elif rank == 4:
            #batch = dataloader_iter.next()
            #preprocess(rank, ds_engine, tokenizer, batch["paragraphs"], batch["claim"])
            #preprocess(rank, ds_engine, tokenizer, list(list(zip(*batch['paragraphs']))[4]), [batch["claim"][4]])
            sharded_nll_list.append(preprocess(rank, ds_engine, tokenizer, list(list(zip(*batch['paragraphs']))[4]), [batch["claim"][4]]))
            
        elif rank == 5:
            #batch = next(dataloader)
            #preprocess(rank, ds_engine, tokenizer, batch["paragraphs"], batch["claim"])
            #preprocess(rank, ds_engine, tokenizer, list(list(zip(*batch['paragraphs']))[5]), [batch["claim"][5]])
            sharded_nll_list.append(preprocess(rank, ds_engine, tokenizer, list(list(zip(*batch['paragraphs']))[5]), [batch["claim"][5]]))

        elif rank == 6:
            #batch = next(dataloader)
            #preprocess(rank, ds_engine, tokenizer, batch["paragraphs"], batch["claim"])
            #preprocess(rank, ds_engine, tokenizer, list(list(zip(*batch['paragraphs']))[6]), [batch["claim"][6]])
            sharded_nll_list.append(preprocess(rank, ds_engine, tokenizer, list(list(zip(*batch['paragraphs']))[6]), [batch["claim"][6]]))

        elif rank == 7:
            #batch = next(dataloader)
            #preprocess(rank, ds_engine, tokenizer, batch["paragraphs"], batch["claim"])
            preprocess(rank, ds_engine, tokenizer, list(list(zip(*batch['paragraphs']))[7]), [batch["claim"][7]])
        #print("batch type: ", type(batch['paragraphs']), '--> ', len(batch['paragraphs']))
            #nll = -log_softmax.gather(2, decoder_input_ids.unsqueeze(2)).squeeze(2)

            #avg_nll = torch.sum(nll, dim=1)
            #sharded_nll_list.append(avg_nll)

            #print("nll shape: ", nll.shape)
            #print("nll: ", nll)
            #quit()
        #topk_scores, indexes = torch.topk(-torch.cat(sharded_nll_list), k=total_no_passages)
        """
    #ids, claims, inputs, generated, gold = evaluate(loader_test, model, tokenizer, device, mode='score')
    #print(len(ids), len(predicted))
    #print(ids[:4])
    #print(predicted[:5])
    """
    if mode == 'score':delete
        data = {'ids': ids, 'prompt': [prompt]*len(ids), 'claim': claims, 'inputs': inputs, 'generated': generated, 'gold': gold}
        df = pd.DataFrame(data, columns=['ids', 'prompt', 'claim', 'inputs', 'generated', 'gold'])
        df.to_excel('p1_generated_response.xlsx')
        print("Excel file saved")
    """
    #result = post_process(test_data, ids, predicted)
    #save_artifact(result, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bigscience/T0pp')
    #parser.add_argument('--base_path', type=str, default='/data/local/gg676/KILT/artifacts')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--rerank_type', type=str, help='bm25, t0_gpt')
    parser.add_argument('--data_split', type=str, help="dev, test", required=True)
    parser.add_argument('--dataset', type=str, help="fever, fm2", required=True)
    parser.add_argument('--data_path', type=str, required=True, help="path where bm25 ranked passage with test data are located") 
    parser.add_argument('--shard_size', type=int, default=20)
    parser.add_argument('--verbalizer_head', type=str, default='Passage: ')
    parser.add_argument('--verbalizer', type=str, default='Please write a statement based on this passage.')
    parser.add_argument('--output_dir', type=str, default='/data/local/gg676/ACL/retrieved_docs')
    #parser.add_argument('--dev_data', type=str, default='/common/home/gg676/NLP/KILT/data/dev.pkl')

    #parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=18)
    args = parser.parse_args()
    main(args)

