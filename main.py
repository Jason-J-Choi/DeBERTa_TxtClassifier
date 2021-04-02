import os
import pathlib
import torch
from DeBERTa import deberta
#from deberta import DeBERTaTxtClassifier
import re
import string
import numpy as np 
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import time
import math
import collections
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json

nltk.download('stopwords')
ps = PorterStemmer()
lem = WordNetLemmatizer()
file_path = pathlib.Path.cwd()

pretrained_model_v2xxlarge = {
    'model_class':  deberta.pretrained_models['xxlarge-v2'],
    'model_path': file_path.joinpath('model', 'v2-xxlarge', 'pytorch_model.bin'),
    'model_config_path': file_path.joinpath('model', 'v2-xxlarge', 'config.json'),
    'model_vocab_path': file_path.joinpath('model', 'v2-xxlarge', 'spm.model'),
    'model_vocab_type': 'spm'
}

pretrained_model_base = {
    'model_class': deberta.pretrained_models['base'],
    'model_path': file_path.joinpath('model', 'base', 'pytorch_model.bin'),
    'model_config_path': file_path.joinpath('model', 'base', 'config.json'),
    'model_vocab_path': file_path.joinpath('model', 'base', 'bpe_encoder.bin'),
    'model_vocab_type': 'gpt2'
}

#setting model here. Change pretrained_model between the base and v2xxlarge
pretrained_model = pretrained_model_base
max_seq_len = 512
vocab_path = pretrained_model['model_vocab_path']
vocab_type = pretrained_model['model_vocab_type']


class Preprocess:
    def __init__(self, filepath):
        self.basepath = './datasets/raw/'
        self.filepath = filepath
        self.filename = os.path.basename(filepath)

        self.fullpath = os.path.join(self.basepath, self.filepath)
        self.output_base = './outputs/'
        self.output_dir = os.path.join(self.output_base, os.path.split(filepath)[0])
        self.output_file = os.path.join(self.output_base, filepath)

        if not os.path.exists(file_path.joinpath(self.basepath).absolute()):
            raise NotADirectoryError(self.basepath + " does not exist")
        if not os.path.exists(file_path.joinpath(self.output_dir).absolute()):
            raise NotADirectoryError()

    def clean_text(self, text):
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.lower()
        text = ' '.join(text.split())
        return text

    def text_process(self, text):
        text = text.split()
        text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
        text = ' '.join(text)
        return text

    def _imdb_concatenate(self, max_col, row, sentence):
        for j in range(1, max_col, 1):
            if not row[j] or not isinstance(row[j], str):
                continue
            sentence = sentence + str(row[j])
        return sentence

    def preprocess(self):
        if not os.path.exists(self.fullpath):
            raise AttributeError("Defined file does not exist")

        text_data, classification = self._preprocessing()

        try:
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
            df = pd.DataFrame({'sentences': text_data, 'label': classification})
            df.to_csv(self.output_file)
        except:
            print("saving didn't work")

        return text_data, classification

    def _preprocessing(self):
        if self.filepath.startswith(('fake', 'ag', 'yelp')):
            df = pd.read_csv(self.fullpath).dropna()
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            df = pd.DataFrame({'label': y})
            df.to_csv(self.output_file)
        elif self.filepath.startswith('mr'):
            df = pd.read_table(self.fullpath, header=None)
            x = []
            y = []
            for i in range(len(df.index)):
                row = df.iloc[i].to_string(index=False)
                yval = row[1]
                xval = row[3:]
                x.append(xval)
                y.append(yval)
            x = pd.DataFrame(x).iloc[:, 0]
            y = pd.DataFrame(y).iloc[:, 1]
        elif self.filepath.startswith('imdb'):
            df = pd.read_csv(self.fullpath)
            x = []
            y = []
            import time
            for i in range(0, df.shape[0]):
                row = df.iloc[i, :]
                first_split = row[0].split(' ', 1)
                if not len(first_split) == 2:
                    continue
                y.append(first_split[0])

                sentence = first_split[1]
                sentence = self._imdb_concatenate(df.shape[1], row, sentence)
                sentence = self.clean_text(sentence)
                sentence = self.text_process(sentence)
                x.append(sentence)
            x = pd.DataFrame(x).iloc[:, 0]
            y = pd.DataFrame(y)
        else:
            raise TypeError("Dataset unknown")

        x = x.apply(self.clean_text).apply(self.text_process)

        return x, y


def tokenize(input_sentences, max_seq_len=512):
    tokenized_sentences = {}
    masks = {}
    for key, sentence in input_sentences.items():
        tokenizer = deberta.tokenizers[vocab_type](vocab_path)
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[:max_seq_len - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)
        paddings = max_seq_len-len(input_ids)
        input_ids = input_ids + [0]*paddings
        input_mask = input_mask + [0]*paddings
        tokenized_sentences[key] = input_ids
        masks[key] = input_mask
    return tokenized_sentences, masks

def train(model, X_train, X_masks, Y_train):
    X_train = collections.OrderedDict(sorted(X_train.items()))
    xlist = []
    ylist = []
    masklist = []
    for key, val in X_train.items():
        xlist.append(val)
        ylist.append(Y_train[key])
        masklist.append(X_masks[key])
    train_inputs = torch.tensor(xlist)
    train_masks = torch.tensor(masklist)
    train_labels = torch.tensor(ylist)
    batch_size = 32
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    start = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    epochs = 4
    total_steps = len(train_dataloader) * epochs
    loss_fn = torch.nn.CrossEntropyLoss()
    for i in range(epochs):
        print("Epoch " + str(i))
        for step, batch in enumerate(train_dataloader):
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t for t in batch)
            model.zero_grad()
            y_pred = model(b_input_ids, b_attn_mask)
            loss = loss_fn(y_pred, b_labels)
            loss.backward()
            optimizer.step()
    print('Finished training model in %.1f sec' % ((time.time()-start)))


#proproc = Preprocess('ag/train_tok.csv')
#text_data = proproc.preprocess()
#print("Done preprocessing")

df = pd.read_csv('./outputs/ag/train_tok.csv')
df = df.dropna()
#print(df.iloc[0])
text_data = df.iloc[:, 1]
print(text_data[0:5])
tokenized, masks = tokenize(text_data)
json.dump(tokenized, open("X_ag.json", 'w'))
json.dump(masks, open("y_ag.json", 'w'))
print("Done tokenizing")
# model = DeBERTaTxtClassifier(pretrained_model['model_path'], pretrained_model['model_config_path'])
# train(model, tokenized, masks, y)
