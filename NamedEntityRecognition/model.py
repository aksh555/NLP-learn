import json
import os
import re

import torch
import torch.nn.functional as F
from nltk import word_tokenize
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,BertConfig, BertForTokenClassification)
from pytorch_pretrained_bert.tokenization import BertTokenizer
import truecase

class BertNer(BertForTokenClassification):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)
        return logits

class Ner:
    def __init__(self, model_dir: str, use_truecase=False):
        self.model , self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["label_map"]
        self.max_seq_length = self.model_config["max_seq_length"]
        self.label_map = {int(k):v for k,v in self.label_map.items()}
        self.use_truecase = use_truecase
        self.model.eval()

    def load_model(self, model_dir: str, model_config: str = "model_config.json"):
        model_config = os.path.join(model_dir,model_config)
        model_config = json.load(open(model_config))
        output_config_file = os.path.join(model_dir, CONFIG_NAME)
        output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
        config = BertConfig(output_config_file)
        model = BertNer(config, num_labels=model_config["num_labels"])
        #if torch.cuda.is_available():
        # model.load_state_dict(torch.load(output_model_file))
        #else:
        model.load_state_dict(torch.load(output_model_file,map_location='cpu'))
        tokenizer = BertTokenizer.from_pretrained(model_config["bert_model"], do_lower_case=False)
        return model, tokenizer, model_config

    def truecase_sentence(self, tokens):
        word_lst = [(w, idx) for idx, w in enumerate(tokens) if all(c.isalpha() for c in w)]
        lst = [w for w, _ in word_lst if re.match(r'\b[A-Z\.\-]+\b', w)]
        if len(lst) and len(lst) == len(word_lst):
            parts = truecase.get_true_case(' '.join(lst)).split()

            # the trucaser has its own tokenization ...
            # skip if the number of word dosen't match
            if len(parts) != len(word_lst): return tokens

            for (w, idx), nw in zip(word_lst, parts):
                tokens[idx] = nw
        return tokens

    def tokenize(self, text: str):
        """ tokenize input"""
        words = word_tokenize(text)
        if self.use_truecase:
            words = self.truecase_sentence(words)
        tokens = []
        valid_positions = []
        for i,word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def preprocess(self, text: str):
        """ preprocess """
        tokens, valid_positions = self.tokenize(text)
        ## insert "[CLS]"
        tokens.insert(0,"[CLS]")
        valid_positions.insert(0,1)
        ## insert "[SEP]"
        tokens.append("[SEP]")
        valid_positions.append(1)
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(0)
        if(len(input_ids) > self.max_seq_length):
            raise NameError("sentence length exceeds max seq length")
        return input_ids,input_mask,segment_ids,valid_positions

    def predict(self, text: str):
        try:
            input_ids,input_mask,segment_ids,valid_ids = self.preprocess(text)
        except NameError:
            raise
        input_ids = torch.tensor([input_ids],dtype=torch.long)
        input_mask = torch.tensor([input_mask],dtype=torch.long)
        segment_ids = torch.tensor([segment_ids],dtype=torch.long)
        valid_ids = torch.tensor([valid_ids],dtype=torch.long)
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask,valid_ids)
        logits = F.softmax(logits,dim=2)
        logits_label = torch.argmax(logits,dim=2)
        logits_label = logits_label.detach().cpu().numpy().tolist()[0]

        logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label)]

        logits = []
        pos = 0
        for index,mask in enumerate(valid_ids[0]):
            if index == 0:
                continue
            if mask == 1:
                logits.append((logits_label[index-pos],logits_confidence[index-pos]))
            else:
                pos += 1
        logits.pop()

        labels = [(self.label_map[label],confidence) for label,confidence in logits]
        output = {}
        ents = []
        for i, (label, confidence) in enumerate(labels):
            if label != 'O':
                ents.append(i)
            output[i] = [label,  confidence]
        '''words = word_tokenize(text)
        assert len(labels) == len(words)
        op = [{"word":word,"tag":label,"confidence":confidence} for word,(label,confidence) in zip(words,labels)]
        print(op)'''
        return output, ents

# n = Ner('./v1-bert-ner-models')
# print(n.predict("sample sent"))
