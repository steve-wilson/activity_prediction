
# make a predictions file for various baseline approaches
# the main one that we want to do: assign probabilities based on the frequency that the class appears in the training dataset
# we don't actually need to run random guess-- we can just give the theoretical value
# another would be to assign equal probabilities to everything-- this will make sure that we are breaking ties in a fair way

# load the dev/test target labels first
# the output file needs to look like: target|prob_0,prob_1,...,prob_N
# and we generate the probs based on the baseline algorithm

import collections
import os
import sys
import random

def write_to_file(targets,predictions_list,path):
    with open(path,'w') as out:
        for i in range(len(targets)):
            out.write(str(targets[i]) + '|' + ','.join(str(round(x,5)) for x in predictions_list[i]) + '\n')

# actually, just need the targets themselves because we can just count them, do a softmax, and then write the output
#def frequency_baseline(targets, output_path):
#    count_dict = collections.Counter(targets)
#    total = len(targets)
#    num_classes = max(targets)+1
#    predictions = [count_dict[c]/float(total) for c in range(num_classes)]
#    prediction_list = []
#    for t in targets:
#        prediction_list.append(predictions)
#    write_to_file(targets,prediction_list,output_path)

def frequency_baseline(targets, train_targets, output_path, num_classes):
    count_dict = collections.Counter(train_targets)
    total = len(train_targets)
    predictions = [count_dict[c]/float(total) for c in range(num_classes)]
    prediction_list = []
    for t in targets:
        prediction_list.append(predictions)
    write_to_file(targets,prediction_list,output_path)

# just give a probability of 1/num_unique_targets to everything
def equal_baseline(targets, output_path, num_classes):
    total = len(set(targets))
    predictions = [1.0/float(num_classes) for c in range(num_classes)]
    prediction_list = []
    for t in targets:
        prediction_list.append(predictions)
    write_to_file(targets,prediction_list,output_path)

# just for fun
def random_baseline(targets, output_path, num_classes):
    prediction_list = []
    for t in targets:
        prediction_list.append([random.random() for c in range(num_classes)])
    write_to_file(targets,prediction_list,output_path)

def load_targets(targets_dir):
    targets = []
    for f in os.listdir(targets_dir):
        with open(targets_dir + os.sep + f) as thefile:
            item = int(thefile.read().strip().split()[0])
            targets.append(item)
    return targets

def convert_targets(ts, ac_lookup):

    out_ts = []
    for t in ts:
        out_ts.append(ac_lookup[t])
    return out_ts

def generate_all_baselines(targets_dir, train_targets_dir, output_dir):
    targets = load_targets(targets_dir)
    train_targets = load_targets(train_targets_dir)
    all_classes = sorted(list(set(targets + train_targets)))
    ac_lookup = {item:idx for idx,item in enumerate(all_classes)}
    targets = convert_targets(targets, ac_lookup)
    train_targets = convert_targets(train_targets, ac_lookup)
    num_classes = len(all_classes)

    frequency_baseline(targets, train_targets, os.path.join(output_dir,'frequency_train_baseline.out'), num_classes)
    frequency_baseline(targets, targets, os.path.join(output_dir,'frequency_test_baseline.out'), num_classes)
    equal_baseline(targets, os.path.join(output_dir,'equal_baseline.out'), num_classes)
    random_baseline(targets, os.path.join(output_dir,'random_baseline.out'), num_classes)

if __name__ == "__main__":
    generate_all_baselines(sys.argv[1], sys.argv[2], sys.argv[3]) #test_targets_dir, train_targets_dir, output_dir
