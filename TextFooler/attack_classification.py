
import os
import numpy as np
from settings import *
from . import criteria
import random

import tensorflow as tf
import tensorflow_hub as hub

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset

from .BERT.tokenization import BertTokenizer
from .BERT.modeling import BertForSequenceClassification, BertConfig

import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')


class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/4"
        self.embed = hub.load(module_url)
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.build_graph()
        self.sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.compat.v1.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.compat.v1.placeholder(tf.string, shape=(None))

        embed_sig_1 = self.embed(self.sts_input1)
        embed_sig_2 = self.embed(self.sts_input2)

        sts_encode1 = tf.nn.l2_normalize(embed_sig_1['outputs'], axis=1)
        sts_encode2 = tf.nn.l2_normalize(embed_sig_2['outputs'], axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


class NLI_infer_BERT(nn.Module):
    def __init__(self, pretrained_dir, nclasses, max_seq_length=128, batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses)

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        # for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self, pretrained_dir, max_seq_length=128, batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
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


def get_original_importance_score(text_ls, len_text, predictor, orig_label, orig_prob, orig_probs, batch_size, num_queries):
    leave_1_texts = [text_ls[:ii] + ['<oov>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
    leave_1_probs = predictor(leave_1_texts, batch_size=batch_size)
    num_queries += len(leave_1_texts)
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
            leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                              leave_1_probs_argmax))).data.cpu().numpy()

    return import_scores, num_queries, leave_1_texts


def get_modified_importance_score(text_ls, len_text, predictor, orig_label, orig_prob, orig_probs,
                                  batch_size, num_queries):
    leave_2_texts = [text_ls[:ii] + ['<oov>'] + ['<oov>'] + text_ls[min(ii + 1, len_text) + 1:] for ii in
                     range(len_text)][:-1]
    switch_2_texts = [text_ls[:ii] + [text_ls[ii + 1]] + [text_ls[ii]] + text_ls[min(ii + 1, len_text) + 1:]
                      for ii in range(len_text - 1)]

    if not leave_2_texts or not switch_2_texts:
        return None, None, num_queries, None, None

    leave_2_probs = predictor(leave_2_texts, batch_size=batch_size)
    num_queries += len(leave_2_texts)
    leave_2_probs_argmax = torch.argmax(leave_2_probs, dim=-1)
    import_score_leave_2 = (orig_prob - leave_2_probs[:, orig_label] + (leave_2_probs_argmax != orig_label).float() * (
            leave_2_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_2_probs_argmax))).data.cpu().numpy()

    switch_2_probs = predictor(switch_2_texts, batch_size=batch_size)
    num_queries += len(switch_2_texts)
    switch_2_probs_argmax = torch.argmax(switch_2_probs, dim=-1)
    import_score_switch_2 = (
            orig_prob - switch_2_probs[:, orig_label] + (switch_2_probs_argmax != orig_label).float() * (
            switch_2_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                               switch_2_probs_argmax))).data.cpu().numpy()

    return import_score_leave_2, import_score_switch_2, num_queries, leave_2_texts, switch_2_texts


def get_semantics_text_ranges(idx, half_sim_score_window, sim_score_window, len_text):
    if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = idx - half_sim_score_window
        text_range_max = idx + half_sim_score_window + 1
    elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = 0
        text_range_max = sim_score_window
    elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
        text_range_min = len_text - sim_score_window
        text_range_max = len_text
    else:
        text_range_min = 0
        text_range_max = len_text

    return text_range_min, text_range_max


def find_pair_synonyms(original_texts, pertubed_text):
    w1, w2 = None, None
    for i, word in enumerate(original_texts):
        if word != pertubed_text[i]:
            if w1 is None:
                w1 = (pertubed_text[i], i)
            else:
                w2 = (pertubed_text[i], i)
                return w1, w2

    return w1, w2


def attack(text_ls, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32, deberta_modified=False, attack_type='all'):
    # first check the prediction of the original text
    orig_probs = predictor([text_ls], batch_size=batch_size).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    print('true label: ' + str(true_label) + ', prediction: ' + str(orig_label) + ', with: ' + str(orig_probs))
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        text_ls = text_ls.split(' ')
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        # get importance score
        import_scores, num_queries, leave_1_texts = get_original_importance_score(
            text_ls, len_text, predictor, orig_label, orig_prob, orig_probs, batch_size, num_queries
        )

        import_score_leave_2, import_score_switch_2, leave_2_texts, switch_2_texts = None, None, None, None
        if deberta_modified is True:
            import_score_leave_2, import_score_switch_2, num_queries, leave_2_texts, switch_2_texts = get_modified_importance_score(
                text_ls, len_text, predictor, orig_label, orig_prob, orig_probs, batch_size, num_queries
            )
            if import_score_leave_2 is None:
                deberta_modified = False

        # get words to perturb ranked by importance score for word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and text_ls[idx] not in stop_words_set:
                    words_perturb.append((idx, text_ls[idx]))
            except:
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))

        words_perturb2 = []
        if import_score_leave_2 is not None:
            # the idx here is the index of the pair of words.
            for idx, score in sorted(enumerate(import_score_leave_2), key=lambda x: x[1], reverse=True):
                try:
                    if score > import_score_threshold and text_ls[idx] not in stop_words_set and text_ls[idx + 1] not in stop_words_set:
                        words_perturb2.append((idx, text_ls[idx], text_ls[idx + 1]))
                except:
                    print(idx, len(text_ls), import_score_leave_2.shape, text_ls, len(leave_2_texts))

        words_switched = []
        if import_score_switch_2 is not None:
            for idx, score in sorted(enumerate(import_score_switch_2), key=lambda x: x[1], reverse=True):
                try:
                    if score > import_score_threshold and text_ls[idx] not in stop_words_set and text_ls[idx + 1] not in stop_words_set:
                        words_switched.append((idx, text_ls[idx], text_ls[idx + 1]))
                except:
                    print(idx, len(text_ls), import_score_switch_2.shape, text_ls, len(switch_2_texts))

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        synonyms_all2 = []
        if len(words_perturb2) > 0:
            words_perturb2_idx = []
            for idx, word, word2 in words_perturb2:
                if word in word2idx and word2 in word2idx:
                    words_perturb2_idx.append(word2idx[word])
                    words_perturb2_idx.append(word2idx[word2])
            synonym_words2, _ = pick_most_similar_words_batch(words_perturb2_idx, cos_sim, idx2word, synonym_num, 0.5)
            for idx, word, word2 in words_perturb2:
                if word in word2idx and word2 in word2idx:
                    synonyms = synonym_words2.pop(0)
                    synonyms2 = synonym_words2.pop(0)
                    if synonyms and synonyms2:
                        synonyms_all2.append((idx, synonyms, synonyms2))

        switched_words = [(idx, word, word2) for idx, word, word2 in words_switched]

        max_syms = max(len(synonyms_all), len(synonyms_all2), len(switched_words))
        synonyms_all += [(999, '<placeholder>') for _ in range(max_syms-len(synonyms_all))]
        synonyms_all2 += [(999, '<placeholder>', '<placeholder>') for _ in range(max_syms-len(synonyms_all2))]
        switched_words += [(999, '<placeholder>', '<placeholder>') for _ in range(max_syms-len(switched_words))]

        # start replacing and attacking
        text_prime, text_prime2, text_prime3 = text_ls[:], text_ls[:], text_ls[:]
        text_cache, text_cache2 = text_prime[:], text_prime2[:]
        num_changed, num_changed2, num_changed3 = 0, 0, 0
        attack_type = 1  # alternating between 1, 2, and 3 as single substitution, pair substitution, and pair switch respectively
        attack_1, attack_2, attack_3 = True, True, True
        if attack_type == 'original':
            attack_2, attack_3 = False
        elif attack_type == 'pair_switch':
            attack_1, attack_2 = False
        elif attack_type == 'pair_synonym':
            attack_1, attack_3 = False

        for (idx, synonyms), (idx2, synonyms2_1, synonyms2_2), (idx3, switch_1, switch_2) in zip(synonyms_all, synonyms_all2, switched_words):
            if attack_1 is True and synonyms != '<placeholder>':
                attack_type = 1
                new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
                new_probs = predictor([' '.join(nt) for nt in new_texts], batch_size=batch_size)

                # compute semantic similarity
                text_range_min, text_range_max = get_semantics_text_ranges(
                    idx=idx, half_sim_score_window=half_sim_score_window, sim_score_window=sim_score_window,
                    len_text=len_text
                )
                semantic_sims = sim_predictor.semantic_sim(
                    [' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                    list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts))
                )[0]

                num_queries += len(new_texts)
                if len(new_probs.shape) < 2:
                    new_probs = new_probs.unsqueeze(0)

                # using cpu here not gpu
                new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
                new_probs_mask_prediction = new_probs_mask
                # prevent bad synonyms
                new_probs_mask *= (semantic_sims >= sim_score_threshold)
                # prevent incompatible pos
                synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                                   if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
                pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
                new_probs_mask *= pos_mask

                # if successfully alter prediction
                if np.sum(new_probs_mask) > 0 and np.sum(new_probs_mask_prediction) > 0:
                    text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                    num_changed += 1
                    break
                elif np.sum(new_probs_mask) > 0 and np.sum(new_probs_mask_prediction) == 0:
                    attack_1 = False
                else:
                    # select word with least confidence score of label y as best replacement and reset for single substitution
                    new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                        (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().to(device)
                    new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                    del new_label_probs

                    if new_label_prob_min < orig_prob:
                        text_prime[idx] = synonyms[new_label_prob_argmin]
                        num_changed += 1

                    del new_label_prob_min
                    del new_label_prob_argmin

                text_cache = text_prime[:]

            # instead of starting back, compute for pair substitution and switch
            # the single substitution was calculated first because it would provide the lesser perturbation ratio
            # pair switching will always only perturb 2 words so it will go next
            # and pair substitution last as it would potentially have the highest perturbation ratio

            if attack_3 is True and switch_1 != '<placeholder>' and switch_2 != '<placeholder>':
                # starting pair switch attack
                attack_type = 3
                new_text = [text_prime3[:idx3] + [switch_2] + [switch_1] + text_prime3[min(idx3 + 1, len_text) + 1:]][0]
                new_text_concat = [' '.join(new_text)]
                new_prob = predictor(new_text_concat, batch_size=batch_size)
                text_range_min, text_range_max = get_semantics_text_ranges(
                    idx=idx3, half_sim_score_window=half_sim_score_window, sim_score_window=sim_score_window,
                    len_text=len_text
                )
                semantic_sim = sim_predictor.semantic_sim([' '.join(text_prime2)], new_text_concat)[0]
                num_queries += 1
                if len(new_prob.shape) < 2:
                    new_prob = new_prob.unsqueeze(0)
                new_prob_mask = (orig_label != torch.argmax(new_prob, dim=-1)).data.cpu().numpy()
                new_prob_mask_prediction = new_prob_mask

                new_prob_mask *= (semantic_sim >= sim_score_threshold)
                if len(new_text) > 10:
                    synonyms_pos_ls = [criteria.get_pos(new_text[max(idx3 - 4, 0):idx3 + 5])[min(4, idx3)]]
                else:
                    synonyms_pos_ls = [criteria.get_pos(new_text)[idx3]]
                pos_mask = np.array(criteria.pos_filter(pos_ls[idx3], synonyms_pos_ls))
                new_prob_mask *= pos_mask

                if np.sum(new_prob_mask) > 0 and np.sum(new_prob_mask_prediction) > 0:
                    text_prime3[idx3] = switch_2
                    text_prime3[idx3+1] = switch_1
                    num_changed3 += 1

                    num_changed = num_changed3
                    break
                elif np.sum(new_prob_mask) > 0 and np.sum(new_prob_mask_prediction) == 0:
                    attack_3 = False
                else:
                    new_label_prob = new_prob[:, orig_label] + torch.from_numpy(
                        (semantic_sim < sim_score_threshold) + (1 - pos_mask).astype(float)).float().to(device)
                    new_label_prob_min, new_label_prob_argmin = torch.min(new_label_prob, dim=-1)
                    del new_label_prob

                    if new_label_prob_min < orig_prob:
                        # text_prime3 change omitted as it shouldn't be replacing
                        num_changed3 += 1
                    del new_label_prob_min
                    del new_label_prob_argmin

            if attack_2 is True and synonyms2_1 != '<placeholder>' and synonyms2_2 != '<placeholder>':
                # starting pair substitution attack
                attack_type = 2
                new_texts = []
                for synonym1 in synonyms2_1:
                    for synonym2 in synonyms2_2:
                        pot_new_text = text_prime2[:idx2] + [synonym1] + [synonym2] + text_prime2[min(idx2 + 1, len_text) + 1:]
                        if len(list(set(pot_new_text) - set(text_prime2))) >= 2:
                            new_texts.append(pot_new_text)
                new_texts = new_texts[:-1]
                if not new_texts:
                    break
                new_probs = predictor([' '.join(nt) for nt in new_texts], batch_size=batch_size)
                text_range_min, text_range_max = get_semantics_text_ranges(
                    idx=idx2, half_sim_score_window=half_sim_score_window, sim_score_window=sim_score_window,
                    len_text=len_text
                )
                semantic_sims = sim_predictor.semantic_sim(
                    [' '.join(text_cache2[text_range_min:text_range_max])] * len(new_texts),
                    list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts))
                )[0]
                num_queries += len(new_texts)
                if len(new_probs.shape) < 2:
                    new_probs = new_probs.unsqueeze(0)
                # using cpu here not gpu
                new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
                new_probs_mask_prediction = new_probs_mask

                new_probs_mask *= (semantic_sims >= sim_score_threshold)
                synonyms_pos_ls = [criteria.get_pos(new_text[max(idx2 - 4, 0):idx2 + 5])[min(4, idx2)]
                                   if len(new_text) > 10 else criteria.get_pos(new_text)[idx2] for new_text in new_texts]
                pos_mask = np.array(criteria.pos_filter(pos_ls[idx2], synonyms_pos_ls))
                new_probs_mask *= pos_mask

                # if successfully alter prediction for pair substitution
                if np.sum(new_probs_mask) > 0 and np.sum(new_probs_mask_prediction) > 0:
                    word1, word2 = find_pair_synonyms(text_prime2, new_texts[(new_probs_mask * semantic_sims).argmax()])
                    text_prime2[idx2] = word1[0]
                    text_prime2[idx2+1] = word2[0]
                    if num_changed2 == 0:
                        num_changed2 += 2
                    else:
                        num_changed2 += 1

                    num_changed = num_changed2
                    break
                elif np.sum(new_probs_mask) > 0 and np.sum(new_probs_mask_prediction) == 0:
                    attack_2 = False
                else:
                    new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                        (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().to(device)
                    new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                    del new_label_probs

                    if new_label_prob_min < orig_prob:
                        word1, word2 = find_pair_synonyms(text_prime2, new_texts[new_label_prob_argmin])
                        text_prime2[idx2] = word1[0]
                        text_prime2[idx2+1] = word2[0]
                        if num_changed2 == 0:
                            num_changed2 += 2
                        else:
                            num_changed2 += 1

                    del new_label_prob_min
                    del new_label_prob_argmin

                text_cache2 = text_prime2[:]

        if attack_type == 2:
            text_prime = text_prime2
        elif attack_type == 3:
            text_prime = text_prime3

        return ' '.join(text_prime), num_changed, orig_label, \
               torch.argmax(predictor([text_prime], batch_size=batch_size)), num_queries
