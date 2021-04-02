
import os
import pathlib

from collections import OrderedDict, Mapping, Sequence

import torch
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset

from TextFooler.attack_classification import NLIDataset_BERT
from TextFooler.attack_classification import InputFeatures

from DeBERTa import deberta
from DeBERTa.deberta.bert import *
from DeBERTa.deberta.config import ModelConfig


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


pretrained_model = pretrained_model_base
max_seq_len = 512
vocab_path = pretrained_model['model_vocab_path']
vocab_type = pretrained_model['model_vocab_type']


class DeBERTaReconfig(torch.nn.Module):
    def __init__(self, model_path, model_config_path):
        super().__init__()
        state = torch.load(model_path, map_location='cpu')
        if 'config' in state:
            config = ModelConfig.from_dict(state['config'])
        else:
            config = ModelConfig.from_json_file(model_config_path)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.config = config
        self.pre_trained = model_path
        self.apply_state(state)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_all_encoded_layers=True,
                position_ids=None, return_att=False):
        """
        Args:
          input_ids:
            a torch.LongTensor of shape [batch_size, sequence_length] \
          with the word token indices in the vocabulary

          attention_mask:
            an optional parameter for input mask or attention mask.

            - If it's an input mask, then it will be torch.LongTensor of shape [batch_size, sequence_length] with indices \
          selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max \
          input sequence length in the current batch. It's the mask that we typically use for attention when \
          a batch has varying length sentences.

            - If it's an attention mask then it will be torch.LongTensor of shape [batch_size, sequence_length, sequence_length]. \
          In this case, it's a mask indicate which tokens in the sequence should be attended by other tokens in the sequence.

          token_type_ids:
            an optional torch.LongTensor of shape [batch_size, sequence_length] with the token \
          types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to \
          a `sentence B` token (see BERT paper for more details).

          output_all_encoded_layers:
            whether to output results of all encoder layers, default, True

        Returns:

          - The output of the stacked transformer layers if `output_all_encoded_layers=True`, else \
          the last layer of stacked transformer layers

          - Attention matrix of self-attention layers if `return_att=True`


        Example::

          # Batch of wordPiece token ids.
          # Each sample was padded with zero to the maxium length of the batch
          input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
          # Mask of valid input ids
          attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

          # DeBERTa model initialized with pretrained base model
          bert = DeBERTa(pre_trained='base')

          encoder_layers = bert(input_ids, attention_mask=attention_mask)

        """

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embedding_output = self.embeddings(input_ids.to(torch.long), token_type_ids.to(torch.long), position_ids,
                                           attention_mask)
        encoded_layers = self.encoder(embedding_output,
                                      attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers, return_att=return_att)
        if return_att:
            encoded_layers, att_matrixs = encoded_layers

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1:]

        if return_att:
            return encoded_layers, att_matrixs
        return encoded_layers

    def apply_state(self, state=None):
        """ Load state from previous loaded model state dictionary.

          Args:
            state (:obj:`dict`, optional): State dictionary as the state returned by torch.module.state_dict(), default: `None`. \
                If it's `None`, then will use the pre-trained state loaded via the constructor to re-initialize \
                the `DeBERTa` model
        """
        if self.pre_trained is None and state is None:
            return

        def key_match(key, s):
            c = [k for k in s if key in k]
            assert len(c) == 1, c
            return c[0]

        current = self.state_dict()
        print('needs to be (state_dict post BERT config): ' + str(len(current)))
        print('is (model state): ' + str(len(state.keys())))
        # TODO: figure out why the layers aren't the same. skipping for now.
        #for c in current.keys():
        #    current[c] = state[key_match(c, state.keys())]
        self.load_state_dict(current)


class NLIDataset_DeBERTa:
    def __init__(self, pretrained_dir, max_seq_length=128, batch_size=32):
        self.tokenizer = deberta.tokenizers[vocab_type](vocab_path)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids))
        return features

    def transform_text(self, data, batch_size=32):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data, self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader


class DeBERTaTxtClassifier(torch.nn.Module):
    def __init__(self, model_path, model_config_path, freeze_deberta=True, num_in=768, num_hidden=50, num_out=2,
                 max_seq_length=128, batch_size=32):
        super().__init__()
        self.model = DeBERTaReconfig(model_path, model_config_path)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.dataset = NLIDataset_DeBERTa(model_path, max_seq_length, batch_size)
        # Specify hidden size of DeBERTa, hidden size of our classifier, and number of labels
        D_in, H, D_out = num_in, num_hidden, num_out

        # Instantiate an one-layer feed-forward classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            #torch.nn.Dropout(0.5),
            torch.nn.Linear(H, D_out)
        )

        if freeze_deberta:
            for param in self.model.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        nextTensor = torch.empty(len(input_ids), 768, dtype=torch.float)
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0]
        
        for i in range(len(last_hidden_state_cls)):
            nextTensor[i] = last_hidden_state_cls[i][0]
        # Feed input to classifier to compute logits
        logits = self.classifier(nextTensor)

        return logits


class NLI_infer_Deberta(torch.nn.Module):
    def __init__(self, model_path, model_config_path, freeze_deberta=True, num_in=768, num_hidden=50, num_out=2,
                 max_seq_length=128, batch_size=32):
        super(NLI_infer_Deberta, self).__init__()
        self.model = DeBERTaTxtClassifier(model_path, model_config_path)
        self.dataset = NLIDataset_DeBERTa(model_path, max_seq_length, batch_size)

    def text_pred(self, text_data, batch_size):
        self.model.eval()
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        # for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids
            input_mask = input_mask
            segment_ids = segment_ids

            with torch.no_grad():
                logits = self.model(input_ids, input_mask)
                probs = functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)