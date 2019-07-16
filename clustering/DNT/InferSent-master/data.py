# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import torch
import io


def get_batch(batch, word_vec):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths


def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in sent.split():
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict


def get_glove(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with io.open(glove_path, encoding='utf8') as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with glove vectors'.format(
                len(word_vec), len(word_dict)))
    return word_vec


def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec


def get_nli(data_path):
    s1 = {}
    s2 = {}
    target = {}

    dico_label = {'entailment': 0,  'neutral': 1, 'contradiction': 2}

    for data_type in ['train', 'dev', 'test']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
        s1[data_type]['path'] = os.path.join(data_path, 's1.' + data_type)
        s2[data_type]['path'] = os.path.join(data_path, 's2.' + data_type)
        target[data_type]['path'] = os.path.join(data_path,
                                                 'labels.' + data_type)

        s1[data_type]['sent'] = [line.rstrip() for line in
                                 open(s1[data_type]['path'], 'r')]
        s2[data_type]['sent'] = [line.rstrip() for line in
                                 open(s2[data_type]['path'], 'r')]
        target[data_type]['data'] = np.array([dico_label[line.rstrip('\n')]
                for line in open(target[data_type]['path'], 'r')])

        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == \
            len(target[data_type]['data'])

        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
                data_type.upper(), len(s1[data_type]['sent']), data_type))

    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
             'label': target['train']['data']}
    dev = {'s1': s1['dev']['sent'], 's2': s2['dev']['sent'],
           'label': target['dev']['data']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],
            'label': target['test']['data']}
    return train, dev, test

def get_sts(data_path, MTL_index, transfer, nclass, start=0, size=1000000):
    if data_path == '../SICK/':
        train_file = 'sick-train.sick'
        dev_file = 'sick-dev.sick'
        test_file = 'sick-test.sick'
    if data_path == '../human_activity_phrase_data/':
        train_file = 'activities-train-all.sick'
        dev_file = 'activities-dev-all.sick'
        test_file = 'activities-test-all.sick'
    if data_path == '../human_activity_lists/':
        train_file = ""
        dev_file = ""
        test_file = "addtl-activity-list.csv"
    if data_path == '../SemEval13/typed/':
        train_file = 'typed_train.sick'
        dev_file = 'typed_dev.sick'
        test_file = 'typed_test.sick'
    if data_path == '../stsbenchmark/':
        train_file = 'sts-train.sick'
        dev_file = 'sts-dev.sick'
        test_file = 'sts-test.sick'
    
    train_valid_test = []
    for file in [test_file]:
        current = 0
        with io.open(data_path + file, encoding='utf-8') as f:
            s = []
            sid = []
            for line in f:
                if current >= start and current < start+size:
                    entries = line.strip().split('\t')
                    if len(entries) < 2:
                        print(line)
                    s.append(entries[1])
                    sid.append(entries[0])
                current += 1
        train_valid_test.append({'s': list(s), 'sid': list(sid)}
                )

    return train_valid_test

def encode_labels(labels, nclass, transfer):
    """
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    """
    if transfer == 'DNT':
        # No encoding if DNT
        return labels
    Y = np.zeros((len(labels), nclass)).astype('float32')
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i+1 == np.floor(y) + 1:
                Y[j, i] = y - np.floor(y)
            if i+1 == np.floor(y):
                Y[j, i] = np.floor(y) - y + 1
    return Y
