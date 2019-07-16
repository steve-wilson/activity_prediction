
# take a directory as input. Each file in the directory will be passed through the evaluation scheme.
# input file format is: 
#   target_idx|prob_0,prob_1,prob_2,...,prob_N

# metrics to use:
# - accuracy@K
# - mrr/mrar
# we could get fancy and do something like "distance between predicted cluster and target cluster",
#   but that probably isn't necessary

import collections
import os
import random
import sys

import numpy as np

# function borrowed from: https://nolanbconaway.github.io/blog/2017/softmax-numpy
def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

def load_predictions(pred_path):
    targets, probs = [],[]
    with open(pred_path) as pred_file:
        for line in pred_file.readlines():
            target,pred_list = line.strip().split('|')
            targets.append(int(target))
            probs.append([float(x) for x in pred_list.split(',')])
    assert len(targets) == len(probs)
    return targets, probs

# on ties, return the average
def my_index(l,x):
    i = l.index(x)
    i1 = i
    length = len(l)
    while i1 < length and  l[i1] == x:
        i1 += 1
    i1 -= 1
    return float(i+i1)/float(2)

def listavg(l):
    return float(sum(l))/len(l)

def get_comparison_rank(target, predictions, other_targets, other_predictions, sub, num_others=1000):
    target_prob = predictions[target - sub]
    order = list(range(len(other_targets)))
    random.shuffle(order)
    other_probs = []
    for i in order:
        if len(other_probs) >= num_others:
            break
        other_target,other_prediction = other_targets[i],other_predictions[i]
        if other_target != target:
            other_probs.append(other_prediction[target - sub])
    r = 0
    other_probs = sorted(other_probs,reverse=True)
    opl = len(other_probs)
#    print target_prob, other_probs
    while r < opl and other_probs[r] > target_prob:
        r += 1
#    print r
    r2 = r
    while r2 < opl and other_probs[r2] == target_prob:
        r2 += 1
    return float(r+r2)/2

def all_evals(targets, predictionss, sub_one, k_list = [1,2,3,5,10,25,50,75,100,200,300,500]):
    k2count = collections.defaultdict(int)
    per_class_k2count = {}
    per_class_total = collections.defaultdict(int)
    rrs = []
    probs = []
    total = len(targets)
    sub = 1 if sub_one else 0
    unique_targets = sorted(list(set(targets)))
    comparison_ranks = []
    top_preds = []
    for i in range(total):
        target, predictions = targets[i], predictionss[i]
        target_prob = predictions[target - sub]
        predicted_item = predictions.index(max(predictions))
        top_preds.append(predicted_item)
        per_class_total[target] += 1
#sorted_predictions = sorted(predictions)
#ranks = [my_index(sorted_predictions,x) for x in predictions]
#target_idx = ranks[target]
        target_idx = my_index(sorted(predictions,reverse=True),target_prob)
        for k in k_list:
            if target_idx + 1  <= k:
                k2count[k] += 1
                if target not in per_class_k2count:
                    per_class_k2count[target] = collections.defaultdict(int)
                per_class_k2count[target][k] += 1
        rrs.append(float(1)/(target_idx+1))
        sm_probs = softmax(np.array(predictions))
        target_sm_prob = sm_probs[target-sub]
        probs.append(target_sm_prob)
        comparison_ranks.append(get_comparison_rank(target, predictions, targets, predictionss, sub))
    for k in k_list:
        print("Accuracy @",k,":",100*k2count.get(k,0)/float(total),'%')
        macro_scores = [per_class_k2count.get(t,{}).get(k,0)/float(per_class_total.get(t,1)) for t in unique_targets]
        total_scores = [per_class_total.get(t,0) for t in unique_targets]
        print("Macro Accuracy @",k,":",100*listavg(macro_scores),'%')
        print("Top 5 classes @",k,":",sorted(zip(unique_targets,macro_scores,total_scores),key=lambda x:x[1],reverse=True)[:5])
    print("MRR :",sum(rrs)/float(total))
    print("Average rank:",sum([1/rr for rr in rrs])/float(total))
    print("Average probability of target:",sum(probs)/float(total))
    print("Average comparison rank:",listavg(comparison_ranks),"/",1000)
    print("Top 10 Predictions:",collections.Counter(top_preds).most_common(10))
#    print("Top targets:",collections.Counter(targets).most_common(10))

def evaluate_file(file_path, sub_one):
    targets, predictions = load_predictions(file_path)
    all_evals(targets, predictions, sub_one)

def evaluate_files_in_dir(dir_path, sub_one=False):

    for f in os.listdir(dir_path):
        print("Predictions file:",f)
        evaluate_file(os.path.join(dir_path,f), sub_one)
        print()

if __name__ == "__main__":
    if len(sys.argv) == 3:
        evaluate_files_in_dir(sys.argv[1],True) # use this for the baselines only
    else:
        evaluate_files_in_dir(sys.argv[1]) # use this for the real prediction

