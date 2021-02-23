import os
import pathlib
import torch
from DeBERTa.DeBERTa.deberta import pretrained_models, tokenizers
from deberta import DeBERTaTxtClassifier, DeBERTaReconfig
import re
import string

file_path = pathlib.Path.cwd()

pretrained_model_base = {
    'model_class': pretrained_models['base'],
    'model_path': file_path.joinpath('model', 'base', 'pytorch_model.bin'),
    'model_config_path': file_path.joinpath('model', 'base', 'config.json'),
    'model_vocab_path': file_path.joinpath('model', 'base', 'bpe_encoder.bin'),
    'model_vocab_type': 'gpt2'
}

#setting model here. Change pretrained_model between the base and v2xxlarge
pretrained_model = pretrained_model_base
model = DeBERTaReconfig(pretrained_model['model_path'],
                                               pretrained_model['model_config_path'])
max_seq_len = 512
vocab_path = pretrained_model['model_vocab_path']
vocab_type = pretrained_model['model_vocab_type']
tokenizer = tokenizers[vocab_type](vocab_path)
tokens = tokenizer.tokenize('EXAMPLES INPUT TEXT OF DEBERTA - CHANGE LATER AS NEEDED')
tokens = tokens[:max_seq_len - 2]
tokens = ['[CLS]'] + tokens + ['[SEP]']
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_mask = [1]*len(input_ids)
paddings = max_seq_len-len(input_ids)
input_ids = input_ids + [0]*paddings
input_mask = input_mask + [0]*paddings
model = DeBERTaReconfig(pretrained_model['model_path'], pretrained_model['model_config_path'])
model.parameters()
output = model(torch.LongTensor(input_ids).unsqueeze(1), torch.LongTensor(input_mask).unsqueeze(1))