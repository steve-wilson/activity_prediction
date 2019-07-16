# DNT.py
# Driver to run DNT using different models, datasets and transfer settings
# Li Zhang

import sys
import os
import argparse

parser = argparse.ArgumentParser(description='DNT driver')
parser.add_argument("--model", type=str, default='infersent', help="Sentence embedding model (infersent|bilstmavg|gran)")
parser.add_argument("--dataset", type=str, default='stsbenchmark', help="Semantic similarity dataset")
parser.add_argument('--dimension', nargs='+', default=['1'], help='Dimension(s) on the dataset')
parser.add_argument('--transfer', default='DNT', help='Transfer learning approach')
parser.add_argument('--save', default='no', help='Save trained model')
parser.add_argument('--load_model', default='no', help='If load model, specify model path. No training will be performed, just evalute')

params, _ = parser.parse_known_args()
if params.transfer == 'DNT' and len(params.dimension) > 1:
    raise ValueError('DNT can only handle 1 dimension. For multi-label learning, use NT.')

# Load model
if params.model == 'infersent':
    sys.path.insert(0, '../InferSent-master/')
    import train_nli_custom as model
    os.chdir('../InferSent-master/')
elif params.model == 'bilstmavg':
    sys.path.insert(0, '../para-nmt-50m-master/main')
    import trainh as model
    os.chdir('../para-nmt-50m-master/main')
elif params.model == 'gran':
    sys.path.insert(0, '../Wieting2017/main')
    import trainh as model
    os.chdir('../Wieting2017/main')
else:
    raise ValueError('Incorrect entry for --model.')

dataset_to_nclasses = {
    'activities': '5'
    }

start = 1
while start <22:
    print(start)
    sys.stdout.flush()
    model.main(['--dataset', params.dataset, '--n_classes', dataset_to_nclasses[params.dataset], '--dimension'] + params.dimension + ['--transfer', params.transfer, '--load_model', params.load_model, '--save', params.save, '--start_line',str(start*1000000)])
