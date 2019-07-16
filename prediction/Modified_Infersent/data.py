# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# MODIFIED BY STEVE WILSON, 2018

import os
import pathlib
import numpy as np
import torch
import random

def get_batch(batch, word_vec, emb_dim=300):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), emb_dim))

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
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with glove vectors'.format(
                len(word_vec), len(word_dict)))
    return word_vec

def get_glove_vocab(glove_path):
    vocab = set([])
    with open(glove_path) as glove:
        for line in glove.readlines():
            word, vec = line.split(' ',1)
            vocab.add(word)
    return vocab

def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec

def load_vocab(vocab_path):
    vocab = set([])
    with open(vocab_path) as vocab_file:
        for line in vocab_file.readlines():
            vocab.add(line.strip())
    return vocab

def build_vocab_over_dir(data_path, glove_path, vocab_file_name):

    vocab_path = data_path.rstrip(os.sep) + os.sep + vocab_file_name
    if os.path.exists(vocab_path):
        vocab = load_vocab(vocab_path)

    else:
        glove_vocab = get_glove_vocab(glove_path)
        vocab = set([])
        for subset in ['train','dev','test']:
            for text_dir in ['tweets','profiles']:
                for f in os.listdir(data_path + os.sep + subset + os.sep + text_dir):
                    with open(data_path + os.sep + subset + os.sep + text_dir + os.sep + f) as text_file:
                        for line in text_file.readlines():
                            words = line.strip().split()
                            vocab |= set([w.lower().strip("""#()[]{}-=~.,?!:;"'""") for w in words if w.lower().strip("""#()[]{}-=~.,?!:;"'""") in glove_vocab])

        with open(vocab_path,'w') as vocab_file:
            for word in vocab:
                vocab_file.write(word+'\n')

    embeddings = get_glove(vocab, glove_path)
    return embeddings

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

def alliter(path):
    p = pathlib.Path(path)
    for sub in p.iterdir():
        if not sub.is_dir():
            yield sub.name

# alternative-- before each epoch begins, read all of the file names, shuffle, 
   # then save to a tmp file that the generator iterates through
def alliter2(path):
    tmp_path = "/scratch/mihalcea_fluxg/steverw/tmp/shuffled_files.out"
    files = os.listdir(path)
    random.shuffle(files)
    with open(tmp_path,'w') as tmp_file:
        tmp_file.write('\n'.join(files))
    del files
    with open(tmp_path,'r') as tmp_file:
        for line in tmp_file:
            yield line.strip()

def alliter3(path):
    files = os.listdir(path)
    random.shuffle(files)
    for f in files:
        yield f

# output should yield sets of files in the data path that are batch size
    # (or smaller if there are not enough files left)
# data_path: where to get the files
# data_subset: which subdir to look in
def batch_generator(data_path, batch_size, data_subset):
    batch = []
    # this assumes that the profiles directory contains all of the users that will be included
    # in the subset of data that we are currently batching our way through
    all_files_generator = alliter3(data_path + os.sep + data_subset + os.sep + 'profiles')
    for f in all_files_generator:
        batch.append(f)
        if len(batch) == batch_size:
            yield batch
            batch = []
    # make sure to give the one final, smaller batch, if there were things leftover
    if batch:
        yield batch


       #done -- make set of tweets randomly shuffle per user
        # done -- make aggregator function robust to missing profiles/tweets (reduce batch size)
        # done -- add into code with torch dataloader to see if this improves the performance!
        # done -- test with max tweets = 100 first! why not start with that?
        # done -- test with fewer hidden units first!
class APDataSet(torch.utils.data.Dataset):
    
    def __init__(self, data_path, word_vec, word_emb_dim, n_classes, use_values=False, subset='train', max_num_tweets=100, use_activities=False, no_profiles=False, no_tweets=False, lmap={}, shuffle_input=True):

        self.files = os.listdir( os.path.join(data_path,subset,'profiles') )
        self.word_vec = word_vec
        self.word_emb_dim = word_emb_dim
        self.n_classes = n_classes
        self.use_values = use_values
        #self.subset = subset
        self.max_num_tweets = max_num_tweets
        self.prefix = os.path.join(data_path,subset) + os.sep
        self.use_activities = use_activities
        self.no_profiles = no_profiles
        self.no_tweets = no_tweets
        self.lmap = lmap
        self.shuffle_input = shuffle_input

    def __getitem__(self, index):

        userid = self.files[index]
        tweet_mat = None
        tweet_lengths = None
        profile_vec = []
        profile_length = 0
        values = []
        values_length = 0
        target = None

        doc_type = 'activities' if self.use_activities else 'tweets'

        if not self.no_tweets:
            with open(self.prefix + doc_type + os.sep + userid) as tweet_file:
                tweets = [prepare_sentence(tweet,self.word_vec) for tweet in tweet_file.readlines()]
                tweets = [tweet for tweet in tweets if tweet]
                # is this causing problems? do we need the same set of tweets each time in order to correctly learn?
                if self.shuffle_input:
                    random.shuffle(tweets)
                if self.max_num_tweets:
                    tweets = tweets[:self.max_num_tweets]
                tweet_lengths = np.array([len(tweet) for tweet in tweets])
                max_len = np.max(tweet_lengths)
                num_tweets = len(tweets)
                tweet_mat = np.zeros((max_len, num_tweets, self.word_emb_dim))
    #            to_delete = []
                for i in range(num_tweets):
                    length_i = tweet_lengths[i]
                    # instead of deleting, treat empty tweets as if they contained 1 word
                    # which we represent with a vector of all 0s
                    if length_i <= 0:
    #                    to_delete.append(i)
                        tweet_lengths[i] = 1
                    for j in range(length_i):
                        tweet_mat[j, i, :] = self.word_vec[tweets[i][j]]
    #            for del_idx in sorted(to_delete, reverse=True):
    #                tweet_lengths = np.delete(tweet_lengths, del_idx)
    #                tweet_mat = np.delete(tweet_mat,del_idx,1)
                tweet_mat = torch.from_numpy(tweet_mat).float()

        if not self.no_profiles:
            with open(self.prefix + 'profiles' + os.sep + userid) as profile_file:
                profile = prepare_sentence(profile_file.read().strip(),self.word_vec)
                profile_length = len(profile)
                if profile_length <= 0:
                    print("emtpy profile for user:",userid)
                profile_vec = np.zeros((profile_length, self.word_emb_dim))
                for j in range(profile_length):
                    profile_vec[j, :] = self.word_vec[profile[j]]
                #profile_vec = torch.from_numpy(profile_vec).float()

        # Values: just one vector
        if self.use_values:
            with open(self.prefix + 'values' + os.sep + userid) as values_file:
                values = [float(x) for x in values_file.read().strip().split()]
                values_length = len(values)
            #values = np.array(values)
            #values = torch.from_numpy(values).float()

        # Target: correct cluster id
        with open(self.prefix + 'clusters_' + str(self.n_classes) + os.sep + userid) as targets_file:
            ids = [int(x) for x in targets_file.read().strip().split()]

            # new way, just use first id as the target value
            target = ids[0]
            if self.lmap:
                target = self.lmap[target]

        #TODO remove this:
        return (tweet_mat, tweet_lengths, profile_vec, profile_length, values, values_length, target)

    def __len__(self):

        return len(self.files)

#    def __add__(self, other):
#
#        raise NotImplementedError
#        #TODO?

# expected input: tweet_mat, tweet_lengths, profile_vec, profile_length, values, values_length, target
def APcollate_fn(batch):

#    print("Initial batch size:",len(batch))
#    initial_batch_size = len(batch)

    tweet_mats = []
    tweet_length_arrs = []
    profile_vec_list = []
    profile_lengths = []
    values = []
    values_lengths = []
    targets = []
    for item in batch:
#        if item[3] > 0:
        if item[0] is not None:
            tweet_mats.append(item[0])
        tweet_length_arrs.append(item[1])
        profile_vec_list.append(item[2])
        profile_lengths.append(item[3])
        if item[4] and item[5]:
            values.append(item[4])
            values_lengths.append(item[5])
        targets.append(item[6])

    batch_size = len(targets)
#    print("Final batch size:",batch_size)
#    assert len(tweet_mats) == batch_size
#    assert len(profile_vec_list) == batch_size
#    assert initial_batch_size == batch_size

    # tweets should be good to go

    # need to do padding for profiles
    profile_mat = torch.Tensor()
    max_profile_length = max(profile_lengths)
    if max_profile_length:
        profile_mat = np.zeros((max_profile_length, batch_size, profile_vec_list[0].shape[1]))
        for i in range(batch_size):
            if profile_lengths[i]:
                for j in range(profile_lengths[i]):
                    profile_mat[j,i,:] = profile_vec_list[i][j]
            # make empty profiles appear as length 1 containing only an all-zero vector word
            # otherwise rnns will crash...
            else:
                profile_lengths[i] = 1
        profile_mat = torch.from_numpy(profile_mat).float()

    # only need to include values if they are nonempty
    if values and values_lengths:
        values = np.array(values)
        values = torch.from_numpy(values).float()
        values_lengths = np.array(values_lengths)

    # targets should be good to go

    return tweet_mats, tweet_length_arrs, profile_mat, np.array(profile_lengths), values, values_lengths, np.array(targets)

def prepare_sentence(sentence, word_vec):
    return [w.lower().strip("""#()[]{}-=~.,?!:;"'""") for w in sentence.strip().replace(r'\n','\n').split() if w.lower().strip("""#()[]{}-=~.,?!:;"'""") in word_vec]

# this is where we actually load the data into memory
# output needs to be: tweets_batch, tweets_length, profile_batch, profile_length
    # values_batch, values_length, target_batch
def load_batch(data_path, files_list, word_vec, word_emb_dim, n_classes, use_values=False, subset='train', max_num_tweets=100):
    
    prefix = data_path + os.sep + subset + os.sep
    batch_size = len(files_list)

    # Tweets: list of tensors
    # NOTE: instead of using batch_size for dimension 1, use num_tweets
    # Tweet_lengths: list of arrays of lengths of the tweets
    tweet_mats = []
    tweet_length_arrs = []
#    print(subset)
    for f_num,f in enumerate(files_list):
#        print("file",f)
        with open(prefix + 'tweets' + os.sep + f) as tweet_file:
            single_user_tweets = [prepare_sentence(t,word_vec) for t in tweet_file.readlines()]
            single_user_tweets = [t for t in single_user_tweets if t!=[]]
            if max_num_tweets:
                single_user_tweets = single_user_tweets[:max_num_tweets]
            single_user_lengths = np.array([len(t) for t in single_user_tweets])
            max_len = np.max(single_user_lengths)
            num_tweets = len(single_user_tweets)
            tweet_mat = np.zeros((max_len, num_tweets, word_emb_dim))
            to_delete = []
            for i in range(num_tweets):
                length_i = len(single_user_tweets[i])
#                print(length_i)
                if length_i <= 0:
                    to_delete.append(i)
                for j in range(length_i):
                    tweet_mat[j, i, :] = word_vec[single_user_tweets[i][j]]
#            if to_delete:
#                print("deleting",to_delete)
            for del_idx in sorted(to_delete, reverse=True):
                single_user_lengths = np.delete(single_user_lengths,del_idx)
                tweet_mat = np.delete(tweet_mat,del_idx,1)
#            print(single_user_lengths.shape)
#            print(single_user_lengths)
#            print(tweet_mat.shape)
#            print(tweet_mat)
        tweet_mats.append(torch.from_numpy(tweet_mat).float())
        tweet_length_arrs.append(single_user_lengths)
        if len(tweet_mats) != f_num + 1:
            print("Batch count off when processing file:",f,"len(tweet_mats), len(tweet_length_arrs:",len(tweet_mats),len(tweet_length_arrs))
            print("User_tweets:",single_user_tweets)
            print("Will try to avoid an error by skipping this user...")
            files_list.remove(f)
    #tweet_mats = torch.stack(tweet_mats)

    # Profiles: just one tensor
    # Profiles_lengths: array of lenghts of profiles
    profiles = []
    for f in files_list:
        with open(prefix + 'profiles' + os.sep + f) as profile_file:
            profiles.append(prepare_sentence(profile_file.read(),word_vec))
    profile_lengths = np.array([len(p) for p in profiles])
    max_len = np.max(profile_lengths)
    # counts zeros
#    num_zeros = np.count_nonzero(profile_lengths==0)
    profile_mat = np.zeros((max_len, batch_size, word_emb_dim))
    to_delete = []
    for i in range(batch_size):
        length_i = len(profiles[i])
        if length_i <= 0:
            to_delete.append(i)
        for j in range(length_i):
            profile_mat[j, i, :] = word_vec[profiles[i][j]]
    for del_idx in to_delete:
        profile_lengths = np.delete(profile_lengths,del_idx)
        profile_mat = np.delete(profile_mat,del_idx,1)
        # also need to delete the tweets for this user since we will no longer use them
        print("Skipping user because of missing profile",files_list[del_idx],"profile:",profiles[del_idx])
        files_list.pop(del_idx)
        tweet_mats.pop(del_idx)
        tweet_length_arrs.pop(del_idx)
        # now, things *should* line up
        print("Verify that sizes match-- len(tweet_mats), len(tweet_length_arrs), profile_mat.shape",len(tweet_mats), len(tweet_length_arrs), profile_mat.shape)
    profile_mat = torch.from_numpy(profile_mat).float()

    # Values: just one matrix
    values = []
    values_lengths = []
    if use_values:
        for f in files_list:
            with open(prefix + 'values' + os.sep + f) as values_file:
                values.append([float(x) for x in values_file.read().strip().split()])
        values_lengths = [len(v) for v in values]
        values_lengths = np.array(values_lengths)
        values = np.array(values)
        values = torch.from_numpy(values).float()

    # Targets: one hot encodings (used to be)
    # targets = np.zeros((batch_size, n_classes))

    # Targets: correct cluster ids, in a list, one for each item in the batch
    targets_list = []
    for i,f in enumerate(files_list):
        with open(prefix + 'clusters_' + str(n_classes) + os.sep + f) as targets_file:
            ids = [int(x) for x in targets_file.read().strip().split()]

#           old way-- assuming that we were going to do multilabel predictions
#            targets[[i]*len(ids),ids] = 1

            # new way, just use first id as the target value
            targets_list.append(ids[0])
    targets = np.array(targets_list)

    if len(tweet_mats) != batch_size:
        print("Warning, tweet mats is only size:",len(tweet_mats),'!')

    return tweet_mats, tweet_length_arrs, profile_mat, profile_lengths, values, values_lengths, targets

    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), emb_dim))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths

def load_weights(path):
    with open(path) as wfile:
        weights = [float(x) for x in wfile.read().split()]
    return torch.tensor(weights)

def load_map(path, do_load):
    lmap = {}
    if do_load:
        with open(path) as mfile:
            for line in mfile:
                if line:
                    parts = line.strip().split()
                    lmap[int(parts[0])] = int(parts[1])
    return lmap

def save_weights(w,path):
    with open(path,'w') as wfile:
        wfile.write(' '.join([str(wgt) for wgt in w]))

def save_map(m,path):
    with open(path,'w') as mfile:
        for k,v in m.items():
            mfile.write(str(k) + ' ' + str(v) + '\n')

def load_train_targets(datasetpath, n_classes, map_labels):
    targets_dir = os.path.join(datasetpath,'train','clusters_' + str(n_classes))
    y = []
    lmap = {}
    mmin = 0
    for f in os.listdir(targets_dir):
        with open(os.path.join(targets_dir,f)) as tfile:
            targets = [int(x) for x in tfile.read().strip().split()]
            if map_labels:
                t = targets[0]
                if t not in lmap:
                    lmap[t] = mmin
                    mmin += 1
                y.append(lmap[t])
            else:
                y.append(targets[0])
    return y, lmap

# set weights to 1/count_in_training_data instead of 1 for everything
def get_weight_tensor(params):
    weights_path = params.datasetpath.rstrip(os.sep) + os.sep + 'weights_' + str(params.n_classes) + '.out'
    map_path = params.datasetpath.rstrip(os.sep) + os.sep + 'map_' + str(params.n_classes) + '.out'
    if os.path.exists(weights_path) and (not params.map_labels or os.path.exists(map_path)):
        return load_weights(weights_path), load_map(map_path, params.map_labels)
    else:
        y,l_map = load_train_targets(params.datasetpath, params.n_classes, params.map_labels)
        n_samples = len(y)
        y = np.array(y)
        counts = np.bincount(y)
        #print(counts, counts.shape)
        weights = n_samples / (params.n_classes * counts)
        weights[weights==np.inf] = 0
        save_weights(weights, weights_path)
        if l_map:
            save_map(l_map, map_path)
        return (torch.from_numpy(weights).float(), l_map)

