
import os
import random
import argparse
from settings import *
from download_model import load_model_state

import pandas as pd

from transformers import DebertaTokenizer, DebertaForSequenceClassification, DebertaV2ForSequenceClassification, AdamW
import torch


class DeBERTaModelParams:
    def __init__(self, model_class):
        if model_class == 'base':
            self.params = self._get_base_params()
            self.model = DebertaForSequenceClassification.from_pretrained(self.params['pretrain_key']).to(device)
        elif model_class == 'xxlarge-v2':
            self.params = self._get_xxlarge_params()
            self.model = DebertaV2ForSequenceClassification.from_pretrained(self.params['pretrain_key']).to(device)
        else:
            NameError("Currently only supporting 'base' and 'xxlarge-v2' model.")
        self.tokenizer = DebertaTokenizer.from_pretrained(self.params['pretrain_key'])
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def _get_base_params(self):
        params = {
            'pretrain_key': 'microsoft/deberta-base'
        }
        return params

    def _get_xxlarge_params(self):
        params = {
            'pretrain_key': 'microsoft/deberta-xxlarge-v2'
        }
        return params


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True, help="Which dataset to attack.")
    parser.add_argument("--nclasses", type=int, default=2, help="How many classes for classification.")
    parser.add_argument("--target_model", type=str, required=True, choices=['base', 'xxlarge-v2'],
                        help="the DeBERTa model type.")
    parser.add_argument("--target_model_path", type=str, required=False,
                        help="The pre-trained model path. If not provided, will automatically download")
    parser.add_argument("--learning_rate", type=int, default=1e-5, help="Learning rate of training model")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train")

    args = parser.parse_args()
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError("The specified dataset does not exist")
    if not args.target_model_path:
        print("the target model was not found. Downloading...")
        load_model_state(args.target_model)
        print('pre-trained model download complete')

    print("accessing datset")
    cor_class = pd.read_csv(args.dataset_path)
    cor_class = cor_class.drop(cor_class.columns[0], axis=1)
    random.shuffle(cor_class)

    train_model = DeBERTaModelParams(args.target_model)
    optimizer = AdamW(train_model.model.parameters(), lr=args.learning_rate)
    epochs = args.num_epochs

    print("start training")
    train_model.model.train()
    losses = []

    for _ in range(epochs):
        for i in range(len(cor_class)):
            optimizer.zero_grad()
            inputs = train_model.tokenizer(cor_class[i][1], return_tensors='pt', padding=True).to(device)
            labels = torch.tensor(int(cor_class[i][0])).unsqueeze(0).to(device)
            outputs = train_model.model(**inputs, labels=labels)
            losses.append(outputs.loss.item())
            outputs.loss.backward()
            optimizer.step()
        overall_loss = sum(losses) / len(losses)
        print(overall_loss)

    torch.save(train_model.model.state_dict(), 'pytorch_model.bin')
    print("Final loss: " + str(losses[-1]))


if __name__ == "__main__":
    main()