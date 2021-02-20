
import os
import pathlib
import torch
from DeBERTa.deberta import pretrained_models, tokenizers

from deberta import DeBERTaTxtClassifier


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
loaded_pretrained_model = DeBERTaTxtClassifier(pretrained_model['model_path'],
                                               pretrained_model['model_config_path'])
print('Model loaded')
vocab_path = pretrained_model['model_vocab_path']
vocab_type = pretrained_model['model_vocab_type']
tokenizer = tokenizers[vocab_type](vocab_path)

max_seq_len = 512
tokens = tokenizer.tokenize('EXAMPLES INPUT TEXT OF DEBERTA - CHANGE LATER AS NEEDED')
tokens = tokens[:max_seq_len - 2]
tokens = ['[CLS]'] + tokens + ['[SEP]']

input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_mask = [1]*len(input_ids)

paddings = max_seq_len-len(input_ids)
input_ids = input_ids + [0]*paddings
input_mask = input_mask + [0]*paddings

features = {
    'input_ids': torch.tensor(input_ids, dtype=torch.int),
    'input_mask': torch.tensor(input_mask, dtype=torch.int)
}

