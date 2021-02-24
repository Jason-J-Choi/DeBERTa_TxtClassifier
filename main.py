import os
import pathlib
import torch
from DeBERTa.DeBERTa.deberta import pretrained_models, tokenizers
from deberta import DeBERTaTxtClassifier
import re
import string
import numpy as np 
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import time
import collections
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json

nltk.download('stopwords')
ps = PorterStemmer()
lem = WordNetLemmatizer()
file_path = pathlib.Path.cwd()

pretrained_model_v2xxlarge = {
    'model_class': pretrained_models['xxlarge-v2'],
    'model_path': file_path.joinpath('model', 'v2-xxlarge', 'pytorch_model.bin'),
    'model_config_path': file_path.joinpath('model', 'v2-xxlarge', 'config.json'),
    'model_vocab_path': file_path.joinpath('model', 'v2-xxlarge', 'spm.model'),
    'model_vocab_type': 'spm'
}

pretrained_model_base = {
    'model_class': pretrained_models['base'],
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

def clean_text(text):
        text = re.sub('[^a-zA-Z0-9]',' ', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.lower()
        return text

def text_process(text):
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text

def preprocess(filepath, basepath = './datasets/raw/'):
    if (filepath == 'fake/train_tok.csv' or filepath == 'fake/test_tok.csv'):
        df = pd.read_csv(basepath + filepath)
        df = df.dropna()
        X = df.iloc[:,0]
        y = df.iloc[:,1]
    if (filepath == 'ag/train_tok.csv' or filepath == 'ag/test_tok.csv'):
        df = pd.read_csv(basepath + filepath)
        df = df.dropna()
        X = df.iloc[:,0]
        y = df.iloc[:,1]
    if (filepath == 'yelp/train_tok.csv' or filepath == 'yelp/test_tok.csv'):
        df = pd.read_csv(basepath + filepath)
        df = df.dropna()
        X = df.iloc[:,0]
        y = df.iloc[:,1]
    if (filepath == 'mr/train.txt' or filepath == 'mr/test.txt'):
        df = pd.read_table(basepath + filepath, header=None)
        X = []
        y = []
        for i in range(len(df.index)):
            row = df.iloc[i].to_string(index=False)
            yval = row[1]
            xval = row[3:]
            X.append(xval)
            y.append(yval)
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
    if (filepath == 'imdb/train_tok.csv' or filepath == 'imdb/test_tok.csv'):
        #error tokenizing data
        df = pd.read_csv(basepath + filepath)
        df = df.dropna()
        print(df)
    
    X = X.apply(clean_text).apply(text_process)
    df.to_csv('./outputs/' + filepath)
    return X, y

def tokenize(input_sentences, max_seq_len = 512):
    tokenized_sentences = collections.OrderedDict()
    masks = collections.OrderedDict()
    for key, sentence in input_sentences.items():
        tokenizer = tokenizers[vocab_type](vocab_path)
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
    optimizer = torch.optim.Adam(model.parameters(),
                      lr=5e-5  
                      )
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


X, y = preprocess('fake/train_tok.csv')
# Serialize data into file:

tokenized, masks = tokenize(X)
json.dump(tokenized, open("./outputs/tokenized_fakenews.json", 'w'))
json.dump(masks, open("./outputs/masks_fakenews.json", 'w'))
model = DeBERTaTxtClassifier(pretrained_model['model_path'], pretrained_model['model_config_path'])
