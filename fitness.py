import numpy as np

import torch
import torch.nn.functional as F

from diversity import get_pred_label

class Fitness:
    def __init__(self, model, n_class, device):
        self.model = model
        self.n_class = n_class
        self.device = device

    def forward(self, x, y):
        #This function produces the forward score of x at class y
        self.model.eval()
        x = x.to(self.device)
        logits = self.model(x).detach().cpu()
        idx = torch.unsqueeze(y, -1)
        y_score = logits.gather(1, idx)
        y_scores = y_score.repeat(1, self.n_class-1)
        res = logits[~torch.zeros(logits.shape,dtype=torch.bool).scatter_(1, idx, torch.tensor(True).expand(idx.shape))].view(logits.size(0), logits.size(1) - idx.size(1))
        margins = res - y_scores
        return margins

    def backward(self, x, y):
        print("compute backward...")
        #This function produces the pairwise gradient of x at class y
        self.model.eval()
        x = x.to(self.device).requires_grad_()
        logits = self.model(x)
        y = y.to(self.device)
        idx = torch.unsqueeze(y, -1)
        y_score = logits.gather(1, idx)
        y_scores = y_score.repeat(1, self.n_class-1)
        indices = torch.zeros(logits.shape,dtype=torch.bool).to(self.device)
        indices = indices.scatter_(1, idx, torch.tensor(True).expand(idx.shape).to(self.device))
        res = logits[~indices]
        res = res.view(logits.size(0), logits.size(1) - idx.size(1))
        margins = res - y_scores
        grads = []
        for i in range(self.n_class-1):
            grads.append(torch.unsqueeze(torch.autograd.grad(margins[:, i], x, grad_outputs=torch.ones_like(margins[:, i]), retain_graph=True)[0].detach().cpu(), 1))
        grads_tensor = torch.cat(grads, 1)
        #The shape of the tensor is now batch_size * (n_classes - 1) * image_size
        return grads_tensor


    def forward_margin(self, x, y, delta, grads, margins):
        x_trans = (x+delta).to(self.device)
        self.model.eval()
        logits = self.model(x_trans).detach().cpu()
        idx = torch.unsqueeze(y, -1)
        y_score = logits.gather(1, idx)
        y_scores = y_score.repeat(1, self.n_class-1)
        res = logits[~torch.zeros(logits.shape,dtype=torch.bool).scatter_(1, idx, torch.tensor(True).expand(idx.shape))].view(logits.size(0), logits.size(1) - idx.size(1))
        margins = res - y_scores
        #if max margin are > 0, then the prediction will not be y
        max_margins = torch.unsqueeze(torch.max(margins, 1).values, -1)
        return max_margins
    
    def forward_ce(self, x, y, delta, grads, margins):
        self.model.eval()
        #print(delta)
        x_trans = (x+delta).to(self.device)
        logits = self.model(x_trans).detach().cpu()
        scores = torch.unsqueeze(F.cross_entropy(logits, y, reduction="none"), -1)
        return scores

    def ce_backward(self, x, y, delta, grads, margins):
        self.model.eval()
        x = x.to(self.device).requires_grad_()
        y = y.to(self.device)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, [x])[0].detach().cpu()
        scores = torch.unsqueeze(torch.sum(grad * delta, dim=[1,2,3]), -1)
        return scores
    
    def lp_projection(self, X, radius, infty=True):
        x_shape = X.shape
        flattened_X = torch.flatten(X, start_dim=1)
        if infty:
            projection = radius * torch.sign(flattened_X)
        else:
            norms = torch.norm(flattened_X, p=2, dim=1)
            projection = flattened_X/(norms[:, None])*radius
        projection = torch.reshape(projection, x_shape)
        return projection

    def ce_backward_lp(self, x, y, radius, infty=True):
        self.model.eval()
        x = x.to(self.device).requires_grad_()
        y = y.to(self.device)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, [x])[0].detach().cpu()
        #Assume there is a channel dimension
        projection = self.lp_projection(grad, radius, infty)
        return projection

    def margin_backward_lp(self, x, y, radius, infty=True):
        margins = self.forward(x, y)
        grads = self.backward(x, y)
        projs = []
        for i in range(grads.shape[1]):
            i_grad = grads[:,i,:]
            i_proj = self.lp_projection(i_grad, radius, infty)
            i_proj = i_proj.unsqueeze(1)
            projs.append(i_proj)
        projs = torch.cat(projs, dim=1)
        changes = torch.sum(grads * projs, dim=[2,3,4])
        scores = margins + changes
        idx = torch.argmax(scores, dim=1)
        idx = torch.unsqueeze(idx, -1)
        shape = projs.shape
        idx = idx.repeat(1, shape[2], shape[3], shape[4])
        idx = idx.view(-1, 1, shape[2], shape[3], shape[4])
        max_proj = projs.gather(1, idx)[:,0,:]
        return max_proj
    
    
    def margin_backward(self, x, y, delta, grads, margins):
        deltas = delta.unsqueeze(1)
        deltas = deltas.repeat(1, self.n_class-1, 1, 1, 1)
        changes = torch.sum(grads * deltas, dim=[2,3,4])
        scores = margins + changes
        scores = torch.unsqueeze(torch.max(scores, 1).values, -1)
        return scores
    
    def random_score(self, x, y, delta, grads, margins):
        rd_score = torch.rand(x.shape[0])
        scores = torch.unsqueeze(rd_score, -1)
        return scores

    def ranking_margin_forward(self, x, y, deltas, topk = 1):
        score_func = self.forward_margin
        return self.ranking(x, y, deltas, score_func, topk, backward=False)

    def ranking_margin_forward_reversed(self, x, y, deltas, topk = 1):
        score_func = self.forward_margin
        return self.ranking(x, y, deltas, score_func, topk, backward=False, descending=False)

    def ranking_ce_forward(self, x, y, deltas, topk = 1):
        score_func = self.forward_ce
        return self.ranking(x, y, deltas, score_func, topk, backward=False)

    def ranking_ce_backward(self, x, y, deltas, topk = 1, testing=True):
        score_func = self.ce_backward
        return self.ranking(x, y, deltas, score_func, topk, backward=False, testing=testing)

    def ranking_margin_backward(self, x, y, deltas, topk = 1):
        score_func = self.margin_backward
        return self.ranking(x, y, deltas, score_func, topk, backward=True)
    
    def ranking_random(self, x, y, deltas, topk=1):
        score_func = self.random_score
        return self.ranking(x, y, deltas, score_func, topk, backward=False)

    def ce_grad_gen(self, x, y, l2_rad, linfty_rad):
        proj1 = self.ce_backward_lp(x, y, l2_rad, False)
        proj2 = self.ce_backward_lp(x, y, linfty_rad, True)
        return [proj1, proj2]
    
    def margin_grad_gen(self, x, y, l2_rad, linfty_rad):
        proj1 = self.margin_backward_lp(x, y, l2_rad, False)
        proj2 = self.margin_backward_lp(x, y, linfty_rad, True)
        return [proj1, proj2]

    def ranking(self, x, y, deltas, score_func, topk = 1, backward=False, descending=True, testing=True):
        scores = []
        if topk > len(deltas):
            topk = len(deltas)
        margins = None
        grads = None
        if backward:
            margins = self.forward(x, y)
            grads = self.backward(x, y)
        for delta in deltas:
            score = score_func(x, y, delta, grads, margins)
            scores.append(score)
        score_tensor = torch.cat(scores, dim=1)
        _, indices = torch.sort(score_tensor, descending=descending)
        picked_deltas = []
        for i in range(topk):
            idx = indices[:,i]
            delta = self.gen_delta(x, deltas, idx)
            picked_deltas.append(delta)
        return indices, picked_deltas

    def filter_deltas(self, deltas, l2_rad, linf_rad):
        filtered = []
        for delta in deltas:
            flattened_delta = torch.flatten(delta, start_dim=1)
            l2_norms, linfty_norms = np.array([]), np.array([])
            for single in flattened_delta:
                linfty_norms = np.append(linfty_norms, torch.norm(single, p=torch.inf).numpy())
                l2_norms = np.append(l2_norms, torch.norm(single, p=2).numpy())
            avg_linf_norm = np.average(linfty_norms)
            avg_l2_norm = np.average(l2_norms)
            if avg_l2_norm <= l2_rad and avg_linf_norm <= linf_rad:
                continue
            else:
                filtered.append(delta)
        return filtered

    def ranking_margin_mixed(self, x, y, deltas, l2_rad = 0.08, linf_rad = 0.01, topk=1):
        projs = self.margin_grad_gen(x, y, l2_rad, linf_rad)
        deltas = self.filter_deltas(deltas, l2_rad, linf_rad)
        deltas += projs
        return self.ranking(x, y, deltas, self.forward_margin, topk, backward=False)
    
    def ranking_ce_mixed(self, x, y, deltas, l2_rad = 0.08, linf_rad = 0.01, topk=1):
        projs = self.ce_grad_gen(x, y, l2_rad, linf_rad)
        deltas = self.filter_deltas(deltas, l2_rad, linf_rad)
        deltas += projs
        return self.ranking(x, y, deltas, self.forward_ce, topk, backward=False)
    
    def ranking_margin_mixed_backward(self, x, y, deltas, l2_rad = 0.08, linf_rad = 0.01, topk=1):
        projs = self.margin_grad_gen(x, y, l2_rad, linf_rad)
        deltas = self.filter_deltas(deltas, l2_rad, linf_rad)
        deltas += projs
        return self.ranking(x, y, deltas, self.margin_backward, topk, backward=True)
    
    def ranking_ce_mixed_backward(self, x, y, deltas, l2_rad = 0.08, linf_rad = 0.01, topk=1):
        projs = self.ce_grad_gen(x, y, l2_rad, linf_rad)
        deltas = self.filter_deltas(deltas, l2_rad, linf_rad)
        deltas += projs
        return self.ranking(x, y, deltas, self.ce_backward, topk, backward=False)

    def gen_delta(self, x, deltas, idx):
        delta = torch.zeros_like(x)
        for i, j in enumerate(idx):
            delta[i,:] = deltas[j.numpy()][i,:]
        return delta
