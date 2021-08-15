'''
Bert for NER
Run:
python bert-ner.py --data_dir ./conll-data --output_dir ./ner-op --cache_dir ./cache --bert_model bert-base-cased --train_batch_size 4 --do_train --no_cuda
'''
import argparse
import json
import logging
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
import truecase
from visdom import Visdom
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME, BertConfig, BertForTokenClassification)
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from seqeval.metrics import classification_report
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(message)s', datefmt = '%H:%M:%S', level = logging.INFO)
logger = logging.getLogger(__name__)

def get_labels():
    labels = ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
    # if not +1, then the loss returned may be nan
    num_labels = len(labels) + 1
    return labels, num_labels

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

class VisdomLinePlotter(object):
    """ Plots to Visdom
        Ref: https://github.com/noagarcia/visdom-tutorial/
    """
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

class Ner(BertForTokenClassification):
    def forward(self, args, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None, attention_mask_label=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len ,feat_dim ,dtype=torch.float32, device=args.device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                # print(f'{active_logits}\n{active_labels}')
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

class NerProcessor(object):
    """Processor for the CoNLL-2003 data set."""

    def get_examples(self, mode, dir):
        """
        :param mode: one element in ['train', 'valid', 'test']
        """
        sentences = self._read_tsv(os.path.join(dir, f"{mode}.txt"))
        return self._create_examples(sentences, mode) # [list of labels, list of words(no split)]

    def _create_examples(self, lines, set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

    def _read_tsv(self, input_file):
        '''read file
            return format :
            [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
        '''
        f = open(input_file)
        #f = open(input_file, encoding='latin-1') # For Spanish add encoding='latin-1'
        data = []
        sentence = []
        label= []
        for line in f:
            if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                if len(sentence) > 0:
                    data.append((sentence,label))
                    sentence = []
                    label = []
                continue    
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append(splits[-1][:-1])

        if len(sentence) >0:
            data.append((sentence,label))
            sentence = []
            label = []
        return data

class Corpus(object):
    def __init__(self, bert_model, max_seq_length, mode, tokenizer, dir=None, use_truecase=False):
        self.processor = NerProcessor()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.examples = self.processor.get_examples(self.mode, dir)
        self.truecase = use_truecase
        
    def __len__(self):
        return len(self.examples)

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
    
    def convert_examples_to_features(self):
        """Loads a data file into a list of `InputBatch`s."""
        label_list, _ = get_labels()
        label_map = {label : i for i, label in enumerate(label_list,1)}

        features = []
        for (_, example) in enumerate(self.examples):
            textlist = example.text_a.split(' ')
            if self.truecase:
                textlist = self.truecase_sentence(textlist)

            labellist = example.label
            tokens = []
            labels = []
            valid = []
            label_mask = []
            for i, word in enumerate(textlist):
                token = self.tokenizer.tokenize(word)               
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                        valid.append(1)
                        label_mask.append(1)
                    else:
                        valid.append(0)
            if len(tokens) >= self.max_seq_length - 1:
                tokens = tokens[0:(self.max_seq_length - 2)]
                labels = labels[0:(self.max_seq_length - 2)]
                valid = valid[0:(self.max_seq_length - 2)]
                label_mask = label_mask[0:(self.max_seq_length - 2)]
            ntokens = []
            segment_ids = []
            label_ids = []
            ntokens.append("[CLS]")
            segment_ids.append(0)
            valid.insert(0,1)
            label_mask.insert(0,1)
            label_ids.append(label_map["[CLS]"])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    label_ids.append(label_map[labels[i]])
            ntokens.append("[SEP]")
            segment_ids.append(0)
            valid.append(1)
            label_mask.append(1)
            label_ids.append(label_map["[SEP]"])
            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            label_mask = [1] * len(label_ids)
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                valid.append(1)
                label_mask.append(0)
            while len(label_ids) < self.max_seq_length:
                label_ids.append(0)
                label_mask.append(0)
            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            assert len(label_ids) == self.max_seq_length
            assert len(valid) == self.max_seq_length
            assert len(label_mask) == self.max_seq_length

            features.append(
                    InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=label_ids,
                                valid_ids=valid,
                                label_mask=label_mask))
        return features

    def get_batches(self, batch_size):
        features = self.convert_examples_to_features()
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids)
        if self.mode == 'train':
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def save_model(args, model):
    label_list, num_labels = get_labels()
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())
    label_map = {i : label for i, label in enumerate(label_list,1)}  
    model_config = {"bert_model":args.bert_model,"do_lower":args.do_lower_case,"max_seq_length":args.max_seq_length,"num_labels":num_labels,"label_map":label_map}
    json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"))
    logger.info(f'Model saved!')

def train_epoch(args, model, iterator, optimizer):
    model.train()
    tr_loss = 0
    for _, batch in enumerate(tqdm(iterator, desc='Training')):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch
        optimizer.zero_grad()
        loss = model(args, input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        tr_loss += loss.item()
        optimizer.step()
    
    return tr_loss / len(iterator)

def evaluate(args, model, iterator, label_map, mode='test'):
    if mode == 'valid':
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _, batch in enumerate(tqdm(iterator,desc='Validadtion')):
                batch = tuple(t.to(args.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
                loss = model(args, input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)
                val_loss += loss.item()
        return val_loss / len(iterator)

    else:
        y_true = []
        y_pred = []
        model.eval()

        for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(iterator, desc="Evaluating"):
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            segment_ids = segment_ids.to(args.device)
            valid_ids = valid_ids.to(args.device)
            label_ids = label_ids.to(args.device)
            l_mask = l_mask.to(args.device)

            with torch.no_grad():
                logits = model(args, input_ids, segment_ids, input_mask, valid_ids=valid_ids, attention_mask_label=l_mask)
            
            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
        
            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, _ in enumerate(label):
                    if j == 0:
                        continue
                    # len(label_map) -> 11 for SEP. (break at SEP)
                    elif label_ids[i][j] == len(label_map):
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])
        
        # print(y_true, y_pred)
        assert len(y_pred) == len(y_true)
        report = classification_report(y_true, y_pred, digits=4)
        logger.info("\n%s", report)
        output_eval_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_eval_file, "w") as writer:
            writer.write(report)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--use_truecase", action='store_true',
                        help="Whether to use truecase in preprocessing.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--early_stop", default=3, type=int,
                        help="No. of epochs for early stopping")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    # set seed
    set_seed(args)

    # labels defined
    label_list, num_labels = get_labels()
    label_map = {i : label for i, label in enumerate(label_list,1)}

    # Create output dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    args.tokenizer = tokenizer

    if args.use_truecase:
        logger.info("Truecasing used")
  
    if args.do_train:
        
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else str(PYTORCH_PRETRAINED_BERT_CACHE)
        model = Ner.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels = num_labels)
        model.to(args.device)

        corpus_train = Corpus(args.bert_model, args.max_seq_length, 'train', args.tokenizer, dir = args.data_dir, use_truecase=args.use_truecase)
        # for eval on validation
        corpus_valid = Corpus(args.bert_model, args.max_seq_length, 'valid', args.tokenizer, dir = args.data_dir, use_truecase=args.use_truecase)
        
        num_train_optimization_steps = int(corpus_train.__len__() / args.train_batch_size) * args.num_train_epochs
        
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion, t_total=num_train_optimization_steps)

        logger.info("***** Training *****")
        logger.info("  Num batches = %d", corpus_train.__len__())
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        train_dataloader = corpus_train.get_batches(args.train_batch_size)
        valid_dataloader = corpus_valid.get_batches(args.train_batch_size)

        best_valid_loss = float('inf')
        stop_it = 0
        for epoch in trange(args.num_train_epochs, desc='epochs'):
            start_time = time.time()
            train_loss = train_epoch(args, model, train_dataloader, optimizer)
            valid_loss = evaluate(args, model, valid_dataloader, label_map, mode="valid")
            #plotter.plot('loss', 'train', 'Model Loss', epoch, train_loss)
            #plotter.plot('loss', 'validation', 'Model Loss', epoch, valid_loss)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            # Display loss
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
            if valid_loss < best_valid_loss:
                logger.info('Validation loss reduced!')
                best_valid_loss = valid_loss
                stop_it = 0
                # Save trained model and the associated configuration
                save_model(args, model)
            else:
                stop_it += 1
                if stop_it == args.early_stop:
                    logger.info(f'Early Stop: No improvement in validation loss for {args.early_stop} iterations!')
                    break
        logger.info('Training done!')
    
    else:
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)

        config = BertConfig(output_config_file)
        model = Ner(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
        model.to(args.device)
    
        corpus_test = Corpus(args.bert_model, args.max_seq_length, 'test', args.tokenizer, dir = args.data_dir, use_truecase=args.use_truecase)
        logger.info("***** Testing *****")
        logger.info("  Num batches = %d", corpus_test.__len__())
        logger.info("  Batch size = %d", args.eval_batch_size)
        test_dataloader = corpus_test.get_batches(args.eval_batch_size)
        
        evaluate(args, model, test_dataloader, label_map)      
        
if __name__ == "__main__":
    # Plots
    #global plotter
    #plotter = VisdomLinePlotter(env_name='Bert Ner')

    main()
