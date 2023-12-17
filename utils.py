import torch
import rbo
import scipy.stats as st
from statsmodels.stats.proportion import proportion_confint
import numpy as np


def get_pred_label(model, X, device):
    model.eval()
    X = X.to(device)
    outputs = model.forward(X).detach().cpu()
    _, predicted = torch.max(outputs.data, 1)
    return predicted

def lp_norms(deltas):
    l2_norms = []
    linfty_norms = []
    for delta in deltas:
        flattened_delta = torch.flatten(delta, start_dim=1)
        for single in flattened_delta:
            linfty_norms.append(torch.norm(single, p=torch.inf).numpy())
            l2_norms.append(torch.norm(single, p=2).numpy())
    return linfty_norms, l2_norms


def confidence_interval(data, confidence=0.95):
    low_int, up_int = st.norm.interval(confidence, loc=np.mean(data), scale=st.sem(data))
    mean, rad = (low_int+up_int)/2, (up_int-low_int)/2
    return mean, rad

def confidence_interval_bin(num_inputs, succ):
    l, u = proportion_confint(count=succ, nobs=num_inputs)
    avg, intv = ((l+u)/2)*100, ((u-l)/2)*100
    return avg, intv

def calc_similarity(idx1, idx2):
    similarity = []
    #print(idx1)
    for i in range(idx1.shape[0]): 
        rank1 = idx1[i].numpy()
        rank2 = idx2[i].numpy()
        similarity.append(rbo.RankingSimilarity(rank1, rank2).rbo(p=1))
        #similarity.append(None)
    return similarity

def weight_avg(nums, weights):
    if len(nums) != len(weights):
        print("Weighted average is used incorrectly")
        raise Exception
    score = 0.0
    for i, num in enumerate(nums):
        score += num * weights[i]
    avg, intv = confidence_interval_bin(sum(weights), score/100)
    return avg, intv
