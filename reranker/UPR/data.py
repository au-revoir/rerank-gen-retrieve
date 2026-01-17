import pickle
import torch

class FeverDataset(torch.utils.data.Dataset):

    def __init__(self, data, verbalizer, verbalizer_head, rerank_type, tokenizer,  max_len, mode='dev'):
        self.data = data
        #print("data: ", self.data)
        self.tokenizer = tokenizer
        #self.prompt = prompt
        self.verbalizer = verbalizer
        self.verbalizer_head = verbalizer_head
        self.rerank_type = rerank_type
        self.max_len = max_len

        #self.mode = mode
        
        #P(paragraphs|claim)
        self.claims = []
        self.paragraphs = []

        self.sample = []

        #self._build()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #sample = self.data[idx]
        #claim_ids = self.inputs["input_ids"][idx].squeeze()
        #claim_mask = self.inputs["attention_mask"][idx].squeeze()
        #if self.mode == 'train':
        #    target_ids = self.targets["input_ids"][idx].squeeze()
        #print("shape", claim_ids.shape)
        #claim_mask = self.inputs["attention_mask"][idx].squeeze()
        #    target_mask = self.targets["attention_mask"][idx].squeeze()

        #    return {"id": ids, "claim_ids": claim_ids, "claim_mask": claim_mask, "target_ids": target_ids, "target_mask": target_mask}
        #else:
        #    return {"id": ids, "claim_ids": claim_ids, "claim_mask": claim_mask}
        #answer = [i["answer"] for i in self.data[idx]["output"]][0]
        #encoded_input = tokenizer
        #print(self.inputs[idx])
        #return {"id": ids, "claim": self.data[idx]["input"], "input_ids": claim_ids, "input_mask": claim_mask, "target": self.targets[idx]}
        question, question_text, answer, paragraph, candidates = self._build(self.data[idx])
        sample = {'question': question, 'question_text': question_text, 'answer': answer, 'paragraphs': paragraph, 'candidates': candidates}
        return sample

    def _remove_newline(self, context):
        context = context.replace('\n\n', ' ')
        context = context.replace('\n', ' ')
        return context.strip()

    def _build(self, data):
        #ids = []
        wikipedia_id = []
        paragraph_id = []
        start_span = []
        end_span = []
        claims = []
        paragraph_text = []
        #paragraph_meta = []
        #candidates = []
        if self.rerank_type == 't0_gpt':
            all_candidates = data['top2_candidates']
            answer = data['answer']
        elif self.rerank_type == 'bm25':
            #all_candidates = [i['text'] for i in data['candidates']]
            all_candidates = data['candidates']
            answer = data['answers']
        elif self.rerank_type == 'cosine_similarity':
            all_candidates = data['candidates']
            answer = data['answers']

        elif self.rerank_type == 'double_rerank':
            all_candidates = data['double_rerank_candidates']
            answer = data['answer']
        else:
            all_candidates = data['candidates']
            answer = data['answers']
        for paragraph in all_candidates:
            #ids.append(paragraph['id'])
            #wikipedia_id.append(paragraph['wikipedia_id'])
            #paragraph_id.append(paragraph['paragraph_id'])
            #start_span.append(paragraph['start_span'])
            #end_span.append(paragraph['end_span'])
            #text = "{} {} {}".format(self.verbalizer_head, self._remove_newline(paragraph), self.verbalizer)
            #print("Paragraph: ")
            if self.rerank_type == 'double_rerank':
                text = "{} {} {}".format(self.verbalizer_head, paragraph, self.verbalizer)
            else:
                text = "{} {} {} {}".format(self.verbalizer_head, paragraph['title'], paragraph['text'], self.verbalizer)
            paragraph_text.append(text)
            #paragraph_meta.append({'wikipedia_id': paragraph['wikipedia_id'], 'paragraph_id': paragraph['paragraph_id'], \
                                   #'start_span': paragraph['start_span'], 'end_span': paragraph['end_span']})
    
        
        #claim_text = "Statement: {}".format(data['question'])
        if self.rerank_type == 'double_rerank':
            input_text = "Statement: {}".format(data['question'])
        else:
            input_text = "Statement: {}".format(data['input'])
        #answer = data['answer'] 
        #answer = data['answers'] 
        #candidates = data['top_candidates']
        #candidates = data['candidates']
        #print("claim: ", claim_text)
        #print("paragraph text[0]: ", paragraph_text[0])
        #print("paragraph text[1]: ", paragraph_text[1])
        #quit()
        #print("paragraph length: ", len(paragraph_text))
        #tokenized_text = self.tokenizer(paragraph_text, padding='longest')#, max_length=self.max_len, truncation=True, return_tensors='pt')
        #paragraph_tokenized = self.tokenize_data(paragraph_text)
        #claim_tokenized = self.tokenize_data(claim_text)

        #claim_text = torch.repeat_interleave(claim_text, len(paragraph_text), dim=0)
        
        #return data['question'], claim_text, answer, paragraph_text, candidates
        if self.rerank_type == 'double_rerank':
            return data['question'], input_text, answer, paragraph_text, all_candidates
        else:
            return data['input'], input_text, answer, paragraph_text, all_candidates
        """ 
            #claims = claims
            #labels = labels
            #print("claims: ", sample["claim"])
            #print("labels: ", sample["label"])
            #break
            if self.mode == 'generate':
                ans = [i['answer'] for i in sample['output']][0]
                if ans == 'SUPPORTS':
                    ans = "supported"
                if ans == 'REFUTES':
                    ans = "refuted"
                claims.append(sample["input"]+" "+"Explain?") #why this is "+ans+"?")
            #if self.mode == 'dev':
                #print("sample: ", sample)
                answer = [i['answer'] for i in sample['output']][0]
                labels.append(answer)
            
            if self.mode == 'score':
                claims.append(sample["input"]+" "+"Is this statement right or wrong?")
            answer = [i['answer'] for i in sample['output']][0]
            labels.append(answer)

        tokenized_inputs = self.tokenizer.batch_encode_plus(
                               claims, max_length=self.max_len, 
                               truncation=True,
                               padding='longest', return_tensors="pt"
                               )
        self.inputs = tokenized_inputs
        self.targets = labels

        #print(tokenized_inputs)    
        #self.inputs = tokenized_inputs
        #self.targets = tokenized_targets
        
def tensorize(data, tokenizer, max_length):
    claims = []
    labels = []

    C = []
    C_mask = []
    T = []
    T_mask = []

    for sample, label in data:
        claims_encoded = tokenizer(sample['claims'], padding='max_length', truncation=True,
                               max_length=max_length, add_special_tokens=False,
                               return_tensor='pt')

        target_encoded = tokenizer(sample['label'], padding='max_length', max_length=2,
                                return_tensor='pt')
        C.append(claims_encoded['input_ids'].unsqueeze(0))
        C_mask.append(target_encoded['attention_mask'].unsqueeze(0))

        T.append(target_encoded['input_ids'].unsqueeze(0))
        T_mask.append(target_encoded['attention_mask'].unsqueeze(0))        

    C = torch.cat(C, dim=0)
    C_mask = torch.cat(C_mask, dim=0).bool()

    T = torch.cat(T, dim=0)
    T_mask = torch.cat
"""

def load_data(file_path):
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
    return data

def save_data(data, out_path):
    with open(out_path, 'wb') as fp:
        pickle.dump(data, fp)
