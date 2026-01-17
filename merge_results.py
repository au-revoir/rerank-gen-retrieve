import os
import json
        
def merge_JsonFiles(path, filename):
    result = list()
    for f1 in filename:
        #if 'reader_T0pp_dpr_BF16_new_prompt' in f1:
        #if 'reader_TOP2_reranked_DPR_T0ppGPT_BF16_new_prompt' in f1 and 'merged' not in f1:
        #if 'stripped_newline' in f1 and 'merged' not in f1:
        if '768max_token_stripped' in f1 and 'merged' not in f1:
        #if 'rank' in f1 and 'merged' not in f1 and 'doublererankedinput.json' not in f1:
        #if True:
        #if 'T0pp_dpr_BF16_new_prompt' in f1 and 'merged' not in f1:
        #if 'DPR_T0pp_new_prompt' in f1 and 'merged' not in f1:
        #if 'reranked_DPR_T0pp_new_prompt' in f1 and 'merged' not in f1:
        #if 'TOP2_reranked_DPR_T0ppGPT_new_prompt' in f1 and 'merged' not in f1:
        #if 'T0pp_dpr_BF16' in f1:
        #if 'DPR_T0pp' in f1:
            with open(path+f1, 'r') as infile:
                result.extend(json.load(infile))
            print("Done so far: ", f1)
    #print("combined total length: ", len(result))
    #unique_results = [dict(t) for t in {tuple(d.items()) for d in result}]
    #print("combined total length: ", len(unique_results))
    with open(path+'webq_merged_readerresults_test.jsonl', 'w') as fp:
        for i in result:
            fp.write(json.dumps(i) + "\n")
        #json.dump(result, fp)
#path = "/data/local/gg676/ACL/retrieved_docs/nq/test/rerankinput_bm25_with_background/"#"/data/local/gg676/KILT/oracle/fever/results/retrieval/t0pp/test/read/"
#path = "/data/local/gg676/ACL/outputs/fm2/test/reader/doctype_bm25/"
path = "/data/local/gg676/ACL/outputs/webq/test/reader/doctype_rerankt0pp/"
all_listings = sorted(os.listdir(path))#/data/local/gg676/KILT/knowledge_base/reranked/fever/dev/'))#'/data/local/gg676/KILT/genread/openai/test_6/read/'))
#print(all_listings)
all_files = []
for i in all_listings:
    if i.endswith(".json"):
        all_files.append(i)
merge_JsonFiles(path, all_files)
