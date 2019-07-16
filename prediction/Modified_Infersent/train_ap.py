# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified by Steve Wilson, Fall 2018

import os
import sys
import time
import argparse
import shutil

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

from data import get_nli, get_batch, build_vocab_over_dir, batch_generator, load_batch, APDataSet, APcollate_fn, get_weight_tensor
from mutils import get_optimizer
from models import APNet, print_model_info

parser = argparse.ArgumentParser(description='Activity Prediction training')
# paths
parser.add_argument("--datasetpath", type=str, default='/scratch/mihalcea_fluxg/steverw/ap_data_test/', help="AP data path (test, small or full)")
parser.add_argument("--outputdir", type=str, default='/scratch/mihalcea_fluxg/steverw/ap_savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--word_emb_path", type=str, default="/scratch/mihalcea_fluxg/steverw/GloVe/glove.twitter.27B.100d.txt", help="word embedding file path")
parser.add_argument("--vocab_file_name", type=str, default="vocab.txt", help="cached vocabulary file path")
parser.add_argument("--debug", action='store_true', help="print additional debug information while running (slow!)")
parser.add_argument("--map_labels", action='store_true', help="don't assume class labels are consectuive indexes into label array")

# training
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--tweet_encoder_type", type=str, default='InferSent', help="see list of encoders")
parser.add_argument("--profile_encoder_type", type=str, default='InferSent', help="see list of encoders")
parser.add_argument("--user_encoder_type", type=str, default='ConvNetEncoder', help="see list of encoders")
parser.add_argument("--tweet_enc_dim", type=int, default=1024, help="encoder nhid dimension")
parser.add_argument("--profile_enc_dim", type=int, default=1024, help="encoder nhid dimension")
parser.add_argument("--user_enc_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
parser.add_argument("--use_values", action='store_true', help="load values vectors")
parser.add_argument("--use_activities", action='store_true', help="use activities text instead of full tweets")
parser.add_argument("--no_profiles", action='store_true', help="don't include any profile information in the model")
parser.add_argument("--no_tweets",action='store_true',help="don't use tweets (or activities) text in the model at all")
parser.add_argument("--evaluate_only", action='store_true', help="just load the model stored at the output_path path and evaluate")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=48109, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=100, help="word embedding dimension")
parser.add_argument("--values_dim", type=int, default=50, help="precomputed values vector dimension")
parser.add_argument("--max_num_tweets", type=int, default=100, help="maximum number of tweets per user to pass as input to the model")
parser.add_argument("--no_shuffle", action='store_true', help="don't shuffle the order of the tweets/activities")

params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)

"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""

# write this function to load AP data *generators*
# avoid loading the entire dataset into memory like infersent seems to do...
# label field needs to be a 1-hot vector of length num_clusters, 1s for clusters mentioned by that person
# actually, let's just push this entire thing to later on...
#train, valid, test = get_ap(params.ap_path)

#train, valid, test = get_nli(params.nlipath)

# if the vocab file exists, use that. Otherwise, load it. Store it in the top level datatset dir
# so that it will be re-created when changing datasets.
word_vec = build_vocab_over_dir(params.datasetpath, params.word_emb_path, params.vocab_file_name)
#word_vec = build_vocab(train['tweets'] + train['profile'] +
#                       valid['tweets'] + valid['profile'] +
#                       test['tweets'] + test['profile'], params.word_emb_path)

# This builds dictionaries in the form train[s1] = [<s>, word1, word2, word3, ... , wordN, </s>]
# since we want to load the data as we go, we will do this later in the pipeline
# there will be a bit of a hit since we have to load the data again for each epoch, but it will use way less memory
# -> could potentially convert all text files into ids into the glove embeddings file-- will compress things a little bit...
#for split in ['s1', 's2']:
#    for data_type in ['train', 'valid', 'test']:
#        eval(data_type)[split] = np.array([['<s>'] +
#            [word for word in sent.split() if word in word_vec] +
#            ['</s>'] for sent in eval(data_type)[split]])


"""
MODEL
"""
# model config
config_ap_model = {
    'n_words'              :  len(word_vec)               ,
    'word_emb_dim'         :  params.word_emb_dim         ,
    'tweet_enc_dim'        :  params.tweet_enc_dim        ,
    'profile_enc_dim'      :  params.profile_enc_dim      ,
    'user_enc_dim'         :  params.user_enc_dim         ,
    'n_enc_layers'         :  params.n_enc_layers         ,
    'dpout_model'          :  params.dpout_model          ,
    'dpout_fc'             :  params.dpout_fc             ,
    'fc_dim'               :  params.fc_dim               ,
    'bsize'                :  params.batch_size           ,
    'n_classes'            :  params.n_classes            ,
    'pool_type'            :  params.pool_type            ,
    'nonlinear_fc'         :  params.nonlinear_fc         ,
    'tweet_encoder_type'   :  params.tweet_encoder_type   ,
    'profile_encoder_type' :  params.profile_encoder_type ,
    'user_encoder_type'    :  params.user_encoder_type    ,
    'values_dim'           :  params.values_dim           ,
    'use_values'           :  params.use_values           ,
    'no_profiles'          :  params.no_profiles          ,
    'no_tweets'            :  params.no_tweets            ,
    'use_cuda'             :  True                        ,

}

# model
encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.tweet_encoder_type in encoder_types, "tweet_encoder_type must be in " + \
                                             str(encoder_types)
assert params.profile_encoder_type in encoder_types, "profile_encoder_type must be in " + \
                                             str(encoder_types)
assert params.user_encoder_type in encoder_types, "user_encoder_type must be in " + \
                                             str(encoder_types)

ap_net = APNet(config_ap_model)
print(ap_net)

#nli_net = NLINet(config_nli_model)
#print(nli_net)

# loss
#weight = torch.FloatTensor(params.n_classes).fill_(1)
weight,lmap = get_weight_tensor(params)
if params.debug:
    print(lmap)
loss_fn = nn.CrossEntropyLoss(weight=weight)
#loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(ap_net.parameters(), **optim_params)

# cuda by default
ap_net.cuda()
loss_fn.cuda()


"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    sys.stdout.flush()
    ap_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    total = 0.
    dcount = 0
    # shuffle the data
    #permutation = np.random.permutation(len(train['profiles']))

    #tweets = train['tweets'][permutation]
    #profiles = train['profile'][permutation]
    #target = train['label'][permutation]

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    training_dataset = APDataSet(params.datasetpath, word_vec, params.word_emb_dim, params.n_classes, params.use_values, 'train', params.max_num_tweets, params.use_activities, params.no_profiles, params.no_tweets, lmap, params.no_shuffle)

    # basic idea: create a generator over the list of all of the files in the tweets/profiles directory
    # the set of files should be the same. make the generator itself shuffle the order that it gives
    # filenames. Given a filename, we can load the tweets, profiles, [values], and labels, then add them
    # to the current batch. This will definitely be slower, but should be much more memory efficient.
    #for user_batch in batch_generator(params.datasetpath, params.batch_size, 'train'):
    for batch in DataLoader(training_dataset, batch_size=params.batch_size, shuffle=True, num_workers=4, collate_fn = APcollate_fn):

        tweets_batch, tweets_length, profile_batch, profile_length, \
            values_batch, values_length, target_batch = batch

        #print(len(tweets_batch), profile_batch.shape, values_batch.shape, target_batch.shape)

        # load the batch based on the set of users selected
#        tweets_batch, tweets_length, profile_batch, profile_length, \
#            values_batch, values_length, target_batch = \
#            load_batch(params.datasetpath, user_batch, word_vec, \
#                       params.word_emb_dim, params.n_classes, params.use_values, \
#                       max_num_tweets=params.max_num_tweets)
        tweets_batch, profile_batch = [Variable(tweets.cuda()) for tweets in tweets_batch], \
                       Variable(profile_batch.cuda())
        #print("this better be true:",tweets_batch[0].requires_grad)
        if params.use_values:
            values_batch = Variable(values_batch.cuda())
        target_batch = Variable(torch.LongTensor(target_batch)).cuda()
#        print(profile_batch.size())
        k = target_batch.size(0)
        total += k
#        print(k,total)

        # model forward
        output = None
        if params.use_values:
            output = ap_net((tweets_batch, tweets_length), (profile_batch, profile_length), \
                values_batch)
        else:
            output = ap_net((tweets_batch, tweets_length), (profile_batch, profile_length))

#        print(output.data)

        pred = output.data.max(1)[1]
        if params.debug:
            dcount += 1
            if dcount == 10:
                print(output.data)
                print(pred)
                print(target_batch.data)
                dcount = 0
        correct += float(pred.long().eq(target_batch.data.long()).cpu().sum())
#        if params.debug:
#            assert len(pred) == len(target_batch)

        # loss
        loss = loss_fn(output, target_batch)
        #all_costs.append(loss.data[0])
        all_costs.append(loss.item())
#        print("len all_costs",len(all_costs))

        # just for tracking purposes
        words_count += (sum([t.nelement() for t in tweets_batch]) + profile_batch.nelement()) / params.word_emb_dim

#        print("computing loss...",words_count)
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0
         
        for p in ap_net.parameters():
            #print(p)
            if p.requires_grad and p.grad is not None:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            print("correct,total",correct,total)
            logs.append('{0} ; loss {1} ; users/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            total,
                            round(np.mean(all_costs), 3),
                            round(float(len(all_costs)) * params.batch_size / (time.time() - last_time),3),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            100.0*correct/(total)))
            print(logs[-1])
            sys.stdout.flush()
            last_time = time.time()
            words_count = 0
            all_costs = []
            if params.debug:
                print_model_info(ap_net)
    print(correct,total)
    train_acc = 100.0 * correct/total
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
#    if params.debug:
#        print_model_info(ap_net)
    return train_acc

# TODO: create a file in tmp. After each prediction, write the predictions and the correct values to the file
    # if the eval accuracy is the new best, copy that file to the outputdir
    # otherwise, re-create it in the next eval loop, and write over the previous one?
        # maybe no? we could actually create a directory and save one for each epoch and then
def evaluate(epoch, eval_type='valid', final_eval=False):
    ap_net.eval()
    correct = 0.
    per_class_correct = torch.zeros(params.n_classes)
    class_counts = torch.zeros(params.n_classes)
    total = 0.
    global val_acc_best, lr, stop_training, adam_stop

    predictions_dir = os.path.join(params.outputdir, params.outputmodelname.rsplit('.',1)[0] + "_predictions/")
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    subset = 'dev' if eval_type == "valid" else 'test'

    eval_dataset = APDataSet(params.datasetpath, word_vec, params.word_emb_dim, params.n_classes, params.use_values, subset, params.max_num_tweets, params.use_activities, params.no_profiles, params.no_tweets, lmap, params.no_shuffle)
   
    predictions_file_path = predictions_dir + eval_type + '_' + str(epoch)
    with open(predictions_file_path,'w+') as predictions_file:
        #for user_batch in batch_generator(params.datasetpath, params.batch_size, subset):
            # prepare batch
        for batch in DataLoader(eval_dataset, params.batch_size, shuffle=True, num_workers=4, collate_fn = APcollate_fn):

            tweets_batch, tweets_length, profile_batch, profile_length, \
                values_batch, values_length, target_batch = batch
#            tweets_batch, tweets_length, profile_batch, profile_length, \
#                values_batch, values_length, target_batch =\
#                load_batch(params.datasetpath, user_batch, word_vec, params.word_emb_dim, \
#                           params.n_classes, params.use_values, subset, params.max_num_tweets)
            tweets_batch, profile_batch = [Variable(tweets.cuda()) for tweets in tweets_batch], Variable(profile_batch.cuda())
            if params.use_values:
                values_batch = Variable(values_batch.cuda())
            target_batch = Variable(torch.LongTensor(target_batch)).cuda()
            k = target_batch.size(0)
            total += k

            # model forward
            output = None
            if params.use_values:
                output = ap_net((tweets_batch, tweets_length), (profile_batch, profile_length), \
                    values_batch)
            else:
                output = ap_net((tweets_batch, tweets_length), (profile_batch, profile_length))

            pred = output.data.max(1)[1]
            matches = pred.long().eq(target_batch.data.long()).cpu()
            new_correct = float(matches.sum())
            correct += new_correct
            if new_correct > 0:
                per_class_correct += pred[matches].bincount(minlength=params.n_classes).cpu().float()
            class_counts += target_batch.bincount(minlength=params.n_classes).cpu().float()
#            correct += float(pred.long().eq(target_batch.data.long()).cpu().sum())

            # save predictions so that we can compute various eval metrics later on
            for i,row in enumerate(output.data):
                predictions_file.write(str(target_batch.data[i].item()) + '|' + ','.join([str(round(x.item(),5)) for x in row]) + '\n')

    # save model
    eval_acc = round(100 * float(correct) / total, 3)
    quotient = per_class_correct / class_counts
    # to avoid any nans if one of the classes didn't exist in class_counts for some reason
    quotient[quotient!=quotient] = 0
    per_class_eval_acc = round(100 * float(quotient.mean()), 3)
    if final_eval:
        print('finalgrep : accuracy {0} : {1} {2}'.format(eval_type, eval_acc, per_class_eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2} ; per-class accuracy: {3}'.format(epoch, eval_type, eval_acc, per_class_eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if per_class_eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(ap_net.state_dict(), os.path.join(params.outputdir,
                       params.outputmodelname))
            # also save the actual predictions
            shutil.copy(predictions_file_path, predictions_dir + 'valid_best')
            val_acc_best = per_class_eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
#                adam_stop = True
    return per_class_eval_acc


"""
Train model on Natural Language Inference task
"""
epoch = 1

if params.evaluate_only:
    stop_training = True

while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, 'valid')
    epoch += 1
    # for ap models, this is definitely going to be overfitting territory
    if train_acc > 90:
        stop_training = True

# Run best model on test set.
ap_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))

print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, 'valid', True)
evaluate(0, 'test', True)

# Save full model -- not necessary because the best model from the validation set will be saved already
# and will be the one that was used during test time
#torch.save(ap_net.state_dict(), os.path.join(params.outputdir, params.outputmodelname + '.encoder.pkl'))
