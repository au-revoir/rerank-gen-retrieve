import os
import sys
import json
import pickle
import torch
import torch.nn as nn
 
#from collections import OrderedDict
from sklearn import metrics

def get_ds_config(model_hidden_size, train_batch_size): 
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
    return ds_config

def include_rank(rank, data):
    return {rank: data}
  
def load_json(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data

def load_jsonl(path):
    data = []
    with open(path, 'r') as fp:
        for item in fp:
            data.append(json.loads(item))
    return data
  
def save_json(data, output_path, rank):
    os.makedirs(output_path, exist_ok=True)
    with open(output_path+'768max_token_stripped_newline_rank'+str(rank)+'.json', 'w') as fp:
        json.dump(data, fp)

def save_jsonl(data, path):
    with open(path, 'w') as fp:
        for item in data:
            fp.write(json.dumps(item) + "\n")

def merge_json_files(path):
    filenames = sorted(i for i in os.listdir(path) if i.endswith(".json"))
    result = list()
    #read_path = os.path.join(pathf)
    for fil in filenames:
        if "merged" not in fil:
            with open(os.path.join(path, fil), 'r') as infile:
                result.extend(json.load(infile))
        #print("Done so far: ", fil)
    print("result length: ", len(result))
    ##unique_results = [dict(t) for t in {tuple(d.items()) for d in result}]
    unique_results = result
    print("unique result length: ", len(unique_results))
    with open(os.path.join(path, 'merged_results.json'), 'w') as output_file:
        json.dump(unique_results, output_file)

