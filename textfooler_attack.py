import argparse
import os
import random
import numpy as np
import pandas as pd

from settings import *


from TextFooler import criteria
from TextFooler.attack_classification import USE, NLI_infer_BERT, attack
from deberta import NLI_infer_Deberta


def read_datasets(path, MR=True, encoding='utf8', shuffle=False):
    df = pd.read_csv(path)
    data = df.iloc[:, 2].tolist()
    labels = df.iloc[:, 1].tolist()
    labels = [i-0 for i in labels]
    # with open(path, encoding=encoding) as fin:
    #     for line in fin:
    #         if MR:
    #             label, sep, text = line.partition(' ')
    #             label = int(label)
    #         else:
    #             label, sep, text = line.partition(',')
    #             label = int(label) - 1
    #         labels.append(label)
    #         data.append(text.split())

    if shuffle:
        perm = list(range(len(data)))
        random.shuffle(perm)
        data = [data[i] for i in perm]
        labels = [labels[i] for i in perm]

    return data, labels


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--config_path",
                        type=str,
                        required=True,
                        help="the deberta config file path")
    parser.add_argument("--nclasses",
                        type=int,
                        default=2,
                        help="How many classes for classification.")
    parser.add_argument("--target_model",
                        type=str,
                        required=True,
                        choices=['bert', 'deberta'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--target_model_type",
                        type=str,
                        choices=['base', 'xxlarge-v2'],
                        default='base',
                        help="If using DeBERTa model, what type of model is being used")
    parser.add_argument("--target_model_path",
                        type=str,
                        required=True,
                        help="pre-trained target model path")
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        required=True,
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_cache_path",
                        type=str,
                        required=True,
                        help="Path to the USE encoder cache.")
    parser.add_argument("--output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")
    parser.add_argument("--type_of_attack",
                        type=str,
                        choices=['all', 'original', 'pair_switch', 'pair_synonym'],
                        default='all',
                        help="What type of generative attack would you like to have performed")

    ## Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=15,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.5,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--data_size",
                        default=100,
                        type=int,
                        help="Data size to create adversaries")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="max sequence length for BERT target model")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # get data to attack
    # texts, labels = dataloader.read_corpus(args.dataset_path)
    texts, labels = read_datasets(args.dataset_path, shuffle=True)
    data = list(zip(texts, labels))
    data = data[:args.data_size]  # choose how many samples for adversary
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    deberta_modified = False
    if args.target_model == 'bert':
        model = NLI_infer_BERT(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length).to(device)
    elif args.target_model == 'deberta':
        model = NLI_infer_Deberta(args.target_model_type, args.target_model_path, args.config_path,
                                  max_seq_length=args.max_seq_length, num_labels=args.nclasses).to(device)
        deberta_modified = True
    else:
        raise LookupError("The inputted target model does not exist")

    print("Model built!")

    model.eval()
    predictor = model.text_pred

    # prepare synonym extractor
    # build dictionary via the embedding file
    idx2word = {}
    word2idx = {}

    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, 'r', encoding="utf8") as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        # load pre-computed cosine similarity matrix if provided
        print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        cos_sim = np.load(args.counter_fitting_cos_sim_path)
    else:
        # calculate the cosine similarity matrix
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(args.counter_fitting_embeddings_path, 'r', encoding="utf8") as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        product = np.dot(embeddings, embeddings.T)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        cos_sim = product / np.dot(norm, norm.T)
    print("Cos sim import finished!")

    # build the semantic similarity module
    use = USE(args.USE_cache_path)

    # start attacking
    orig_failures = 0.
    adv_failures = 0.
    changed_rates = []
    nums_queries = []
    orig_texts = []
    adv_texts = []
    true_labels = []
    new_labels = []
    log_file = open(os.path.join(args.output_dir, 'results_log'), 'a')

    stop_words_set = criteria.get_stopwords()
    print('Start attacking!')
    model.eval()
    for idx, (text, true_label) in enumerate(data):
        if idx % 20 == 0:
            print('{} samples out of {} have been finished!'.format(idx, args.data_size))
        new_text, num_changed, orig_label, \
        new_label, num_queries = attack(text, true_label, predictor, stop_words_set,
                                        word2idx, idx2word, cos_sim, sim_predictor=use,
                                        sim_score_threshold=args.sim_score_threshold,
                                        import_score_threshold=args.import_score_threshold,
                                        sim_score_window=args.sim_score_window,
                                        synonym_num=args.synonym_num,
                                        batch_size=args.batch_size,
                                        deberta_modified=deberta_modified,
                                        attack_type=args.type_of_attack)

        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)
        if true_label != new_label:
            adv_failures += 1

        changed_rate = 1.0 * num_changed / len(text.split(' '))

        if true_label == orig_label and true_label != new_label:
            changed_rates.append(changed_rate)
            orig_texts.append(' '.join(text))
            adv_texts.append(new_text)
            true_labels.append(true_label)
            new_labels.append(new_label)

    message = 'For target model {}: original accuracy: {:.3f}%, adv accuracy: {:.3f}%, ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}\n'.format(args.target_model,
                                                                           (1 - orig_failures / int(args.data_size)) * 100,
                                                                           (1 - adv_failures / int(args.data_size)) * 100,
                                                                           np.mean(changed_rates) * 100,
                                                                           np.mean(nums_queries))
    print(message)
    log_file.write(message)

    with open(os.path.join(args.output_dir, 'adversaries.txt'), 'w') as ofile:
        for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
            ofile.write(
                'orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))


if __name__ == "__main__":
    main()
