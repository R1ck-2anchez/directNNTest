import numpy as np
import sys
import os
import pickle
import scipy.stats

import torch
from ATS.ATS import ATS
from model import feature_extractor
from utils import get_pred_label

def gen_delta(x, deltas, idx):
    #given the idx of deltas, generate the x_transform
    delta = torch.zeros_like(x)
    for i, j in enumerate(idx):
        delta[i,:] = deltas[int(j.numpy())][i,:]
    return delta


class DeepGauge:
    def __init__(self, model, device, model_type, loader, profile_path, feature_interm=True, score='SD'):
        """
        :param path_to_stats: path to where neuron statistics are stored in pickle
        """
        self.device = device
        if score == 'SD':
            self.score = self.SD_Compute
        elif score == 'BD':
            self.score = self.BD_Compute

        self.feat_extractor = feature_extractor(model, model_type)
        self.feature_interm = feature_interm
        self.profile_path = profile_path
        if not os.path.exists(profile_path):
            self.profiling(loader)

    def profiling(self, loader):
        # loader: train loader
        print('preparing neuron profiles.....')
        profile = {}
        batch = 0
        for x, _ in loader:
            extracted = self.feat_extractor(x.to(self.device))  # dictionary containing both intermediate features and logit
            keys = list(extracted.keys())

            for key in keys:
                feat = extracted[key].cpu().detach().numpy()
                if batch == 0:
                    profile[key] = {'min': feat.min(0), 'max': feat.max(0)}
                else:
                    profile[key]['min'] = np.minimum(profile[key]['min'], feat.min(0))
                    profile[key]['max'] = np.maximum(profile[key]['max'], feat.max(0))
            batch += 1

        print(f'saving neuron profile to {self.profile_path}....')
        with open(self.profile_path, 'wb') as f:
            pickle.dump(profile, f)

    def kmul(self, layer_outputs, k=1000):
        # layer_outputs: 2D feat representation of single, dim=(num_neurons,)

        with open(self.profile_path, 'rb') as f:
            profile = pickle.load(f)
        keys = list(profile.keys())
        if self.feature_interm:  # use intermediate features
            lower_bound = np.mean(profile[keys[0]]['min'], axis=(-2, -1))
            upper_bound = np.mean(profile[keys[0]]['max'], axis=(-2, -1))
        else: # use logits
            lower_bound = profile[keys[-1]]['min']
            upper_bound = profile[keys[-1]]['max']

        # num_neurons = layer_outputs.shape[1]
        num_neurons = len(layer_outputs)
        section_indexes = []
        for neuron_idx in range(num_neurons):
            unit_range = (upper_bound[neuron_idx] - lower_bound[neuron_idx]) / k
            output = layer_outputs[neuron_idx]
            if unit_range == 0:
                section_indexes.append(-sys.maxsize - 1)
                continue
            if output > upper_bound[neuron_idx] or output < lower_bound[neuron_idx]:
                section_indexes.append(-sys.maxsize - 1)
                continue
            subrange_index = int((output - lower_bound[neuron_idx]) / unit_range)
            if subrange_index == k:
                subrange_index -= 1
            section_indexes.append(subrange_index)

        return section_indexes

    def SD_Compute(self, a, b):
        count = 0
        total = 0
        invalid = 0
        min_num = -sys.maxsize - 1
        for i in range(len(a)):
            total += 1
            if a[i] == min_num:
                invalid += 1
                continue
            if a[i] == b[i]:
                count += 1
        if total == invalid:
            return 1
        sd = 1 - (count / (total))
        return sd

    """
    def nbc(self, function_pickle_path, layer_outputs, layer_name, model_type):
        cov_dict = OrderedDict()
        with open(self.profile_path, 'rb') as f:
            min = pickle.load(f)
        with open(self.profile_path, 'rb') as f:
            max = pickle.load(f)
        #TODO: merge into one pickle file and load to variable info

        for neuron_idx in range(len(layer_outputs)):
            output = layer_outputs[neuron_idx]
            lower_bound = min[(layer_name, neuron_idx)]
            upper_bound = max[(layer_name, neuron_idx)]
            if output < lower_bound:
                cov_dict[(layer_name, neuron_idx)] = 0
            elif output > upper_bound:
                cov_dict[(layer_name, neuron_idx)] = 2
            else:
                cov_dict[(layer_name, neuron_idx)] = 1
        return cov_dict
    """

    def BD_Compute(self, c_s, c_f):
        count = 0
        for i in c_s:
            if c_s[i] == c_f[i] and c_s[i] != 1:
                count += 1
        return 1 - count / len(c_s)

    def run(self, f, f_trans):
        """
        :param f: feature representations of original inputs
        :param f_trans: feature representations of transformed inputs
        """
        bs = f.shape[0]

        if self.feature_interm:  # intermediate feat
            orig_outputs_s = torch.mean(f, dim=(-2, -1)).cpu().detach().numpy()
            orig_outputs_f = torch.mean(f_trans, dim=(-2, -1)).cpu().detach().numpy()
        else:  # logits
            orig_outputs_s = f.cpu().detach().numpy()
            orig_outputs_f = f_trans.cpu().detach().numpy()

        score = []
        for i in range(bs):
            section_indexes_s = self.kmul(orig_outputs_s[i], k=1000)
            section_indexes_f = self.kmul(orig_outputs_f[i], k=1000)
            score.append(self.score(section_indexes_s, section_indexes_f))

        return torch.unsqueeze(torch.Tensor(np.array(score)), -1)

    def single_DG(self, x, orig_x, delta, grads, margins):
        x_trans = (x + delta)

        def extract(_x):
            extracted = self.feat_extractor(_x.to(self.device))  # dictionary containing both intermediate features and logit
            keys = list(extracted.keys())
            return extracted[keys[0]], extracted[keys[1]]

        feat, logit = extract(orig_x)
        feat_trans, logit_trans = extract(x_trans)

        if self.feature_interm:
            scores = self.run(feat, feat_trans)
        else:
            scores = self.run(logit, logit_trans)
        return scores

    def ranking(self, x, orig_x, deltas, topk=1, backward=False, descending=True):
        scores = []
        if topk > len(deltas):
            topk = len(deltas)
        margins = None
        grads = None

        for delta in deltas:
            score = self.single_DG(x, orig_x, delta, grads, margins)  
            scores.append(score)
        score_tensor = torch.cat(scores, dim=1) #dim = (bs, num_transforms)
        _, indices = torch.sort(score_tensor, descending=descending)
        picked_deltas = []
        for i in range(topk):
            idx = indices[:, i]
            delta = gen_delta(x, deltas, idx)
            picked_deltas.append(delta)
        return indices, picked_deltas



class BoostingDiversity:
    # reference: https://github.com/imcsq/ASE22-MPPrioritize
    def __init__(self, model, device, model_type="vgg", feature_interm=True, score='KL'):
        if score == 'KL':
            self.score = self.KL_Div
        elif score == 'JS':
            self.score = self.JS_Div
        elif score == 'Wasserstein':
            self.score = self.Wasserstein
        elif score == 'Hellinger':
            self.score = self.Hellinger
        self.device = device
        self.model = model
        self.feature_interm = feature_interm
        self.model_type = model_type
        self.feat_extractor = feature_extractor(model, model_type)

    def scale(self, layer_output, rmax=1, rmin=0):
        X_std = (layer_output - layer_output.min()) / float(
            layer_output.max() - layer_output.min())
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled

    def KL_Div(self, p, q):
        kl = scipy.stats.entropy(p, q)
        if kl == np.inf:
            kl = 1
        return kl

    def JS_Div(self, p, q):
        p = np.asarray(p)
        q = np.asarray(q)
        M = (p + q) / 2
        js = 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)
        return js

    def Wasserstein(self, p, q):
        p = p / np.sum(p)
        q = q / np.sum(q)
        wd = scipy.stats.wasserstein_distance(p, q)
        return wd

    def Hellinger(self, p, q):
        p = p / np.sum(p)
        q = q / np.sum(q)
        hd = 1 / np.sqrt(2) * np.linalg.norm(np.sqrt(p) - np.sqrt(q))
        return hd

    def run(self, f, f_trans):
        """
        :param f: feature representations of original inputs
        :param f_trans: feature representations of transformed inputs
        """
        bs = f.shape[0]

        f_scaled = torch.stack([self.scale(_f) for _f in f])
        f_trans_scaled = torch.stack([self.scale(_f) for _f in f_trans])

        if len(f_scaled.shape) > 2: # intermediate feat
            scale_outputs_s = torch.mean(f_scaled, dim=(-2, -1)).cpu().detach().numpy()
            scale_outputs_f = torch.mean(f_trans_scaled, dim=(-2, -1)).cpu().detach().numpy()

        else: # logits
            scale_outputs_s = f_scaled.cpu().detach().numpy()
            scale_outputs_f = f_trans_scaled.cpu().detach().numpy()

        score = [self.score(scale_outputs_s[i], scale_outputs_f[i]) for i in range(bs)]
        return torch.unsqueeze(torch.Tensor(np.array(score)), -1)

    def single_BD(self, x, orig_x, delta, grads, margins):
        x_trans = (x + delta)

        def extract(_x):
            extracted = self.feat_extractor(_x.to(self.device))  # dictionary containing both intermediate features and logit
            keys = list(extracted.keys())
            return extracted[keys[0]], extracted[keys[1]]

        feat, logit = extract(orig_x)
        feat_trans, logit_trans = extract(x_trans)
        if self.feature_interm:
            scores = self.run(feat, feat_trans)
        else:
            scores = self.run(logit, logit_trans)
        return scores

    def ranking(self, x, orig_x, deltas, topk=1, backward=False, descending=True):
        scores = []
        if topk > len(deltas):
            topk = len(deltas)
        margins = None
        grads = None

        for delta in deltas:
            score = self.single_BD(x, orig_x, delta, grads, margins)  #dim = (bs, 1)
            scores.append(score)
        score_tensor = torch.cat(scores, dim=1) #dim = (bs, num_transforms)
        _, indices = torch.sort(score_tensor, descending=descending)
        picked_deltas = []
        for i in range(topk):
            idx = indices[:, i]
            delta = gen_delta(x, deltas, idx)
            picked_deltas.append(delta)

    # indices: tensor of dim=(bs, num_transforms)
    # picked_deltas: list of len topk, each element is of dim = (bs, 3, 32, 32) (same as x)
        return indices, picked_deltas



class ATSDiversity:
    def __init__(self, model, n_classes, device, parallel=True):
        self.model = model
        self.device = device
        self.n_classes = n_classes
        self.parallel = parallel

    def data_transform(self, x, deltas):
        transformed_data = []
        for delta in deltas:
            transformed_data.append((x+delta).unsqueeze(1))
        transformed_x = torch.cat(transformed_data, dim=1)
        return transformed_x
    
    def single_rank(self, data):
        rank = self.test_ATS(data)
        rank = torch.tensor(rank).unsqueeze(0)
        return rank

    def ranking(self, x, y, deltas, topk = 1):
        transformed_x = self.data_transform(x, deltas)
        ranking = []
        for i, data in enumerate(transformed_x):
            rank =self.single_rank(data)
            ranking.append(rank)
        indices = torch.cat(ranking)
        picked_deltas = []
        for i in range(topk):
            idx = indices[:,i]
            delta = gen_delta(x, deltas, idx)
            picked_deltas.append(delta)
        return indices, picked_deltas
    

    def test_ATS(self, X):
        y_pred = get_pred_label(self.model, X, self.device)
        ats = ATS()
        div_rank, _, _ = ats.get_priority_sequence(X, y_pred, self.n_classes, self.model, th=0.001)
        return div_rank