import os
import sys
import json
import pickle
import torch
import torch.nn as nn
 
#from collections import OrderedDict
from sklearn import metrics

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
    with open(output_path+'rank'+str(rank)+'.json', 'w') as fp:
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

