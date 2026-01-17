import torch

class FeverDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, split, prompts, version="genread_paper", document_type="oracle", mode="generate", concat_top_k=None):
        self.data = data
        self.split = split
        self.mode = mode
        self.concat_top_k = concat_top_k
        self.version = version
        self.document_type = document_type
        #self.is_oracle = is_oracle
        if self.version == "question_answering":
            self.verbalizer = prompts[version][mode]["verbalizer"]
        self.head_verbalizer = prompts[version][mode]["head_verbalizer"]
        self.tail_verbalizer = prompts[version][mode]["tail_verbalizer"]
        #print(self.data[0])
    def __len__(self):
        return len(self.data)

    def _remove_newline(self, context):
        context = context.replace('\n\n', ' ')
        context = context.replace('\n', ' ')
        return context.strip()

    def __getitem__(self, idx):
        #print("data: ", self.data[idx])
        #ids = self.data[idx]["id"]
        if self.mode == "read":
            #claim = self.data[idx]["claim"]
            if self.document_type == 'oracle':
                ids = self.data[idx]["id"]
                claim = self.data[idx]["input"]
                context = self.data[idx]["output"][0]['paragraph_text']
                answer = self.data[idx]["output"][0]["answer"]

            elif self.document_type in ['bm25']:
                ids = None
                inputs = self.data[idx]["input"]
                context = self.data[idx]["candidates"][0]['text'] #only the top ranked document after retrieval
                #if self.split != 'test':
                #    answer = self.data[idx]["answers"][0]["answer"]
                #else:
                #    answer = []
                answer = self.data[idx]['answers']
                
            elif self.document_type in ['doublererankt0ppgpt', 'rerankt0pp', 't5-11b']:
                ids = None
                inputs = self.data[idx]["question"]
                context = self.data[idx]["candidates"][0]['text'] #only the top re-ranked document after retrieval
                #if self.split != 'test':
                #    answer = self.data[idx]["answer"]
                #    #answer = self.data[idx]["answer"][0]["answer"]
                #else:
                #    answer = []
                answer = self.data[idx]['answer']

            elif self.document_type == 'double_rerankt0pp':
                ids = None
                inputs = self.data[idx]["question"]
                context = self.data[idx]["candidates"][0]
                answer = self.data[idx]['answer']

            elif self.document_type in ['rerankt0pp_concat']:
                ids = None
                inputs = self.data[idx]["question"]
                context = ""
                for i in range(self.concat_top_k):
                    curr_passage = self.data[idx]["candidates"][i]['text']
                    if curr_passage[-1] == '.':
                        filler = " "
                    else:
                        filler = ". "
                    context += curr_passage + filler
                #print("Context: ", context)
                #quit()
                """
                if self.data[idx]["candidates"][0]['text'][-1] == '.':
                    filler = " "
                else:
                    filler = ". "
                context = self.data[idx]["candidates"][0]['text'] + filler + self.data[idx]['candidates'][1]['text']
                """
                answer = self.data[idx]['answer']

            elif self.document_type == 'gpt':
                ids = None
                inputs = self.data[idx]['question']
                context = self._remove_newline(self.data[idx]['gpt_output'][0])
                answer = self.data[idx]['answer']

            elif self.document_type in ['genread_bm25']:
                ids = None
                claim =  self.data[idx]['question']
                context = self.data[idx]['ctxs'][0]['text']
                if self.split != 'test':
                    answer = self.data[idx]["answers"][0]
                else:
                    answer = []
            else:
                ids = self.data[idx]["id"]
                claim = self.data[idx]["claim"]
                context = self.data[idx]["generated_context"]
                answer = self.data[idx]["answer"]

            if self.version == "question_answering":
                text = "{}{}{}{}".format(self.head_verbalizer, 
                                         context,
                                         self.verbalizer,
                                         inputs, 
                                         self.tail_verbalizer)
            else:
                text = "{}{}{}{}".format(context, 
                                     self.head_verbalizer, 
                                     inputs, 
                                     self.tail_verbalizer)
            if ids:
                input_data =  {"id": ids, 
                               "claim": claim, 
                               "generated_context": context, 
                               "text": text, 
                               "answer": answer}
            else: 
                input_data =  {
                               "input": inputs, 
                               "context": context, 
                               "text": text, 
                               "answer": answer}

        if self.mode == "generate":
            claim = self.data[idx]["input"]
            answer = self.data[idx]["output"][0]["answer"]
            text = "{}{}{}{}".format(self.head_verbalizer, 
                                     self.verbalizer, 
                                     claim, 
                                     self.tail_verbalizer)
            input_data =  {"id": ids, 
                           "claim": claim, 
                           "text": text, 
                           "answer": answer}

        return input_data

