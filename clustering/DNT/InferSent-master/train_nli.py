# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Modified by Li Zhang

import os
import sys
import time
import argparse

import numpy as np
from scipy.stats import spearmanr, pearsonr

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli, get_sts, get_batch, build_vocab
from mutils import get_optimizer
from models import NLINet


def main(args):

    GLOVE_PATH = "dataset/GloVe/glove.840B.300d.txt"

    parser = argparse.ArgumentParser(description='NLI training')
    # paths
    parser.add_argument("--nlipath", type=str, default='dataset/SNLI/', help="NLI data path (SNLI or MultiNLI)")
    parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
    parser.add_argument("--outputmodelname", type=str, default='model.pickle')

    # dataset, dimensions, transfer learning   
    parser.add_argument("--dataset", type=str, required=True, help="Semantic similarity dataset")
    parser.add_argument('--dimension', nargs='+', required=True, help='Dimension(s) on the dataset')
    parser.add_argument('--transfer', default='DNT', help='Transfer learning approach')
    parser.add_argument('--save', default='no', help='Save trained model')
    parser.add_argument('--load_model', default='no', help='If load model, do not perform training, just evalute')

    # training
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
    parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
    parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
    parser.add_argument("--optimizer", type=str, default="sgd,lr=5", help="adam or sgd,lr=0.1")
    parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
    parser.add_argument("--decay", type=float, default=1., help="lr decay")
    parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
    parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

    # model
    parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
    parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
    parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
    parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
    parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
    parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

    # gpu
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1236, help="seed")


    params, _ = parser.parse_known_args(args)

    # set gpu device
    torch.cuda.set_device(params.gpu_id)

    # print parameters passed, and all parameters
    #print('\ntogrep : {0}\n'.format(sys.argv[1:]))
    #print(params)

    def trainepoch(epoch):
        print('TRAINING : Epoch ' + str(epoch))
        nli_net.train()
        logs = []

        last_time = time.time()
        #correct = 0.
        # shuffle the data
        permutation = np.random.permutation(len(train['s1']))

        s1 = train['s1'][permutation]
        s2 = train['s2'][permutation]
        
        targets = [x[permutation] for x in train['labels']]

        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
            and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
        #print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))


        for stidx in range(0, len(s1), params.batch_size):
            tgt_batches = []
            # prepare batch
            s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                            word_vec)
            s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                            word_vec)
            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            for i, _ in enumerate(MTL_index):
                tgt_batches.append(Variable(torch.FloatTensor(targets[i][stidx:stidx + params.batch_size])).cuda())
            
            #for dim in [1,2,3,4]:
            # model forward
            outputs = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        
            # loss
            if params.transfer == 'DNT':
                #print(outputs[0])
                #print((tgt_batches[0] - 1)/(params.n_classes-1))
                losses = [nli_net.loss_fn(outputs[i], (tgt_batches[i] - 1)/(params.n_classes-1)) for i,_ in enumerate(MTL_index)]
            elif params.transfer == 'NT':
                losses = [nli_net.loss_fn(outputs[i], tgt_batches[i]) for i,_ in enumerate(MTL_index)]
            #if 'kl' in MTL_index:
            #    output1 = torch.log(output1)
            
            loss = np.sum(losses)


            #loss = loss1 + loss2 + loss3 + loss4# + loss5 + loss6 + loss7 + loss8
            #ADDED
            #optimizer.zero_grad()
            #loss1.backward(retain_graph=True)
            #loss2.backward(retain_graph=True)
            #loss3.backward(retain_graph=True)
            #loss4.backward(retain_graph=True)
            #optimizer.step()
            #END ADDED
            """
            if dim == 1:
                loss = nli_net.loss_fn(output1, tgt_batch1)
            elif dim == 2:
                loss = nli_net.loss_fn(output2, tgt_batch2)
            elif dim == 3:
                loss = nli_net.loss_fn(output3, tgt_batch3)
            elif dim == 4:
                loss = nli_net.loss_fn(output4, tgt_batch4)
            """    
            # backward
            optimizer.zero_grad()
            loss.backward()

            # optimizer step
            optimizer.step()

    def evaluate(epoch, eval_type='valid', flag='', correlation=spearmanr, transfer='NT'):
        nli_net.eval()
        #correct = 0.
        preds = []
        r = np.arange(1, 1 + nli_net.n_classes)
        global val_acc_best, lr, stop_training, adam_stop

        if eval_type == 'valid':
            print('VALIDATION : Epoch {0}'.format(epoch))
            s1 = valid['s1']
            s2 = valid['s2']
            targets = valid['scores']
        elif eval_type == 'test':
            print('TEST : Epoch {0}'.format(epoch))
            s1 = test['s1']
            s2 = test['s2']
            targets = test['scores']
        elif eval_type == 'train':
            print('EVAL ON TRAIN : Epoch {0}'.format(epoch))
            s1 = train['s1']
            s2 = train['s2']
            targets = train['scores']
        else:
            raise ValueError('Wrong eval_type.')
        
        probas = [[] for _ in MTL_index]
        correct = 0.

        for i in range(0, len(s1), params.batch_size):
            # prepare batch
            s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec)
            s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec)
            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())

            # model forward
            outputs = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

            for i, _ in enumerate(MTL_index):
                if len(probas[i]) == 0:
                    probas[i] = outputs[i].data.cpu().numpy()
                else:
                    probas[i] = np.concatenate((probas[i], outputs[i].data.cpu().numpy()), axis=0)

            """
            if 2 in MTL_index:
                if 'e' in MTL_index:
                    tgt_batch2 = Variable(torch.LongTensor(target2[i:i + params.batch_size])).cuda()
                    pred2 = output2.data.max(1)[1]
                    correct += pred2.long().eq(tgt_batch2.data.long()).cpu().sum()
                else:
                    if len(probas2) == 0:
                        probas2 = output2.data.cpu().numpy()
                    else:
                        probas2 = np.concatenate((probas2, output2.data.cpu().numpy()), axis=0)
           """
        
        if transfer == 'NT':
            ret = [correlation(np.dot(x, r), y)[0] for x,y in zip(probas,targets)]
        elif transfer == 'DNT':
            ret = [correlation(x, y)[0] for x,y in zip(probas,targets)]
        else:
            raise ValueError('Wrong transfer.')
        """
        if 2 in MTL_index:      
            if 'e' in MTL_index:
                ret.append(round(100 * correct/len(s1), 2))
            else:
                yhat2 = np.dot(probas2, r)
                p2 = spearmanr(yhat2, target2)[0]
                ret.append(p2)
        else:
            ret.append(0)
        """

        return ret

    """
    SEED
    """
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    """
    DATA
    """
    #for i in range(1,9):
    #    print(i)
    #    print('----------')
    dataset_path = {
        'stsbenchmark': '../stsbenchmark/',
        'sts12': '../SemEval12/',
        'sick': '../SICK/',
        'activities': '../human_activity_phrase_data/',
        'sag': '../ShortAnswerGrading_v2.0/data/processed/',
        'typed': '../SemEval13/typed/'
    }
    #MTL_index = [1,2,3,4, 'mse'] #'e'
    MTL_index = [int(x) for x in params.dimension]
    train, valid, test = get_sts(dataset_path[params.dataset], MTL_index, params.transfer, params.n_classes)
    
    word_vec = build_vocab(train['s1'] + train['s2'] +
                            valid['s1'] + valid['s2'] +
                            test['s1'] + test['s2'], GLOVE_PATH)

    for split in ['s1', 's2']:
        for data_type in ['train', 'valid', 'test']:
            eval(data_type)[split] = np.array([
                [word for word in sent.split() if word in word_vec] 
                for sent in eval(data_type)[split]])
            #eval(data_type)[split] = np.array([['<s>'] +
            #    [word for word in sent.split() if word in word_vec or word[:2] == 'dc'] +
            #    ['</s>'] for sent in eval(data_type)[split]])

    params.word_emb_dim = 300

    """
    MODEL
    """
    # model config
    config_nli_model = {
        'n_words'        :  len(word_vec)          ,
        'word_emb_dim'   :  params.word_emb_dim   ,
        'enc_lstm_dim'   :  params.enc_lstm_dim   ,
        'n_enc_layers'   :  params.n_enc_layers   ,
        'dpout_model'    :  params.dpout_model    ,
        'dpout_fc'       :  params.dpout_fc       ,
        'fc_dim'         :  params.fc_dim         ,
        'bsize'          :  params.batch_size     ,
        'n_classes'      :  params.n_classes      ,
        'pool_type'      :  params.pool_type      ,
        'nonlinear_fc'   :  params.nonlinear_fc   ,
        'encoder_type'   :  params.encoder_type   ,
        'use_cuda'       :  True                  ,
        'MTL_index'      :  MTL_index             ,
        'transfer'       :  params.transfer       
    }

    # model
    encoder_types = ['BLSTMEncoder', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                        'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                        'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
    assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                                    str(encoder_types)
    perfs_all = []
    for rd in range(1):
        print("Round", rd)
        if params.load_model == 'no':
            nli_net = NLINet(config_nli_model)
            nli_net.encoder = torch.load('encoder/infersent.allnli.pickle', map_location={'cuda:1' : 'cuda:0', 'cuda:2' : 'cuda:0'})
        else:
            nli_net = torch.load(params.load_model)
        print(nli_net)

        # optimizer
        optim_fn, optim_params = get_optimizer(params.optimizer)
        optimizer = optim_fn(nli_net.parameters(), **optim_params)

        # cuda by default
        nli_net.cuda()


        """
        TRAIN
        """
        val_acc_best = -1e10
        adam_stop = False
        stop_training = False
        lr = optim_params['lr'] if 'sgd' in params.optimizer else None

        last_result = 0
        last_test_result = 0
        drop_count = 0
        """
        Train model on Natural Language Inference task
        """
        correlation = spearmanr if params.dataset == 'activities' else pearsonr
        epoch = 0
        perfs_valid = evaluate(epoch, 'valid', 'begin', correlation, params.transfer)
        perfs_test = evaluate(epoch, 'test', 'begin', correlation, params.transfer)
        print(perfs_valid, perfs_test)
        epoch += 1

        if params.load_model == 'no':
            while not stop_training and epoch <= params.n_epochs:
                trainepoch(epoch)
                perfs_valid = evaluate(epoch, 'valid', '', correlation, params.transfer)
                perfs_test = evaluate(epoch, 'test', '', correlation, params.transfer)
                print(perfs_valid, perfs_test)

                epoch += 1
            #perfs_all.append(perfs)
        if params.save != 'no':
            torch.save(nli_net, params.save)
    
    #print(perfs_all)

if __name__ == "__main__":
    sys.exit(int(main() or 0))