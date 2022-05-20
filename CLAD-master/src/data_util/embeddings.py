#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random

import torch
from transformers import AutoTokenizer
from transformers import AutoModel
from torchnlp.word_to_vector import GloVe, FastText
from torchnlp.encoders.text import SpacyEncoder
from torch.utils.data import TensorDataset, DataLoader
#  from keras_preprocessing.sequence import pad_sequences

from sentence_transformers import SentenceTransformer

from config import device
import config

import pdb


def avg_bert_embed(sentences):
    embedding_model = AutoModel.from_pretrained(config.LM)
    embeddings = []
    tokenizer = AutoTokenizer.from_pretrained(config.LM, do_lower_case=True)

    for i, sent in enumerate(sentences):

        if (i + 1) % 100 == 0:
            print("embedding " + str(i + 1) + " out of " + str(len(sentences)))

        tokens = tokenizer.encode(sent, max_length=512)

        tokens = torch.tensor(tokens).unsqueeze(0).to(device)
        embedding_model.cuda()
        bert_embeddings = embedding_model(tokens)[0][
            0]  # get the embedding from the pretrained bert model

        num_tokens = len(bert_embeddings)
        embedding_size = len(bert_embeddings[0])
        avg_bert = torch.zeros([embedding_size],
                               dtype=torch.float32).to(device)

        for index in range(embedding_size):
            for token in bert_embeddings:
                avg_bert[index] += token[index].item()
            avg_bert[index] = avg_bert[index] / num_tokens

        # add as a numpy array
        embeddings.append(avg_bert.cpu().numpy())

    return embeddings


def sentence_bert_embed(sentences):

    #model = SentenceTransformer('bert-base-nli-mean-tokens')
    #sentence_embeddings = model.encode(sentences)
    sentence_embeddings = [1 for i in sentences]
    return sentence_embeddings, sentences.tolist()


def avg_glove_embed(sentences, word_vectors):

    encoder = SpacyEncoder(sentences, min_occurrences=3,
                           append_eos=False)  #tokenizer
    word_vectors = word_vectors[encoder.vocab]
    embeddings = []

    for row in sentences:
        s = encoder.encode(row)
        v = word_vectors[s]
        m = torch.mean(v, dim=0)
        embeddings.append(m.numpy())
    return embeddings, sentences


def avg_fasttext_embed(sentences, word_vectors):

    encoder = SpacyEncoder(sentences, min_occurrences=3,
                           append_eos=False)  #tokenizer
    word_vectors = word_vectors[encoder.vocab]
    embeddings = []

    for row in sentences:
        s = encoder.encode(row)
        v = word_vectors[s]
        m = torch.mean(v, dim=0)
        embeddings.append(m.numpy())
    return embeddings, sentences


def preprocess_with_avg_Glove(train, test):

    word_vectors = GloVe(name='6B', dim=300)
    train_x, train_text = avg_glove_embed(train.sentence.values.tolist(),
                                          word_vectors)
    train_y = train.label.values
    test_x, test_text = avg_glove_embed(test.sentence.values.tolist(),
                                        word_vectors)
    test_y = test.label.values
    return train_x, train_y, test_x, test_y, train_text, test_text


def preprocess_with_avg_fasttext(train, test):

    word_vectors = FastText(language='en')
    train_x, train_text = avg_fasttext_embed(train.sentence.values.tolist(),
                                             word_vectors)
    train_y = train.label.values
    test_x, test_text = avg_fasttext_embed(test.sentence.values.tolist(),
                                           word_vectors)
    test_y = test.label.values
    return train_x, train_y, test_x, test_y, train_text, test_text


def preprocess_with_s_bert(train, test):

    #  pdb.set_trace()
    train_x, train_text = sentence_bert_embed(train.sentence.values)
    train_y = train.label.values
    test_x, test_text = sentence_bert_embed(test.sentence.values)
    test_y = test.label.values
    return train_x, train_y, test_x, test_y, train_text, test_text


def preprocess_with_avg_bert(train, test):

    train_x = avg_bert_embed(train.sentence.values)
    train_y = train.label.values
    #  val_x = torch.tensor(avg_bert_embed(val.sentence.values))
    #  val_y = torch.tensor(val.label.values)
    test_x = avg_bert_embed(test.sentence.values)
    test_y = test.label.values

    return train_x, train_y, test_x, test_y


def bert_test_data(sentence):
    input_ids = []
    attention_masks = []
    tokenizer = AutoTokenizer.from_pretrained(config.LM)
    length = len(sentence)
    encoded_dict = tokenizer(sentence, padding=True, truncation=True)

    data = testDataset(encoded_dict, length)

    return data


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class testDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, length):
        self.encodings = encodings
        self.length = length

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        return item

    def __len__(self):
        return self.length
