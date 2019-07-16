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
    parser.add_argument('--start_line',type=int, default=0, help="line in the test file to start at")

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

        s = test['s']
        sid = test['sid']

        for i in range(0, len(s), params.batch_size):
            print("starting new batch at: "+str(i))
            # prepare batch
            s_batch, s_len = get_batch(s[i:i + params.batch_size], word_vec)
#            sid_batch, sid_len = get_batch(sid[i:i + params.batch_size], word_vec)
            s_batch = Variable(s_batch.cuda())

            # model forward
            sid_batch = sid[i:i+params.batch_size]
            outputs = nli_net((s_batch, s_len), (np.array([si for si in sid_batch])))

        return None

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
        'activities': '../human_activity_lists/',
        'sag': '../ShortAnswerGrading_v2.0/data/processed/',
        'typed': '../SemEval13/typed/'
    }

    #MTL_index = [1,2,3,4, 'mse'] #'e'
    MTL_index = [int(x) for x in params.dimension]
    start = int(params.start_line)
    test = get_sts(dataset_path[params.dataset], MTL_index, params.transfer, params.n_classes, start)[0]
    
    word_vec = build_vocab(test['s'], GLOVE_PATH)

    for split in ['s']:
        for data_type in ['test']:
#            eval(data_type)[split] = np.array([
#                [word for word in sent.split() if word in word_vec] 
#                for sent in eval(data_type)[split]])

            sentences = []
            for i,sent in enumerate(eval(data_type)[split]):
                sent_arr = [word for word in sent.split() if word in word_vec]
                if sent_arr == []:
                    eval(data_type)['sid'].pop(i)
                else:
                    sentences.append(sent_arr)
            assert len(sentences) == len(eval(data_type)['sid'])
            eval(data_type)[split] = np.array(sentences)

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
            nli_net = torch.load(params.load_model, map_location=lambda storage, loc: storage.cuda(0))
#        print(nli_net)
#        print("Cuda? "+str(torch.cuda.is_available()))
#        print(str(next(nli_net.parameters()).is_cuda))
#        print(str(next(nli_net.encoder.parameters()).is_cuda))

        # optimizer
        optim_fn, optim_params = get_optimizer(params.optimizer)
        optimizer = optim_fn(nli_net.parameters(), **optim_params)

        # cuda by default
        nli_net.cuda()

        nli_net.encoder.cuda()
        nli_net.encoder.enc_lstm.cuda()

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
        perfs_test = evaluate(epoch, 'test', 'begin', correlation, params.transfer)


if __name__ == "__main__":
    sys.exit(int(main() or 0))
