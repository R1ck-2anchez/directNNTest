import numpy as np

import torch
import torch.nn as nn

from test import Testing
import random

from transform import DataGen
from fitness import Fitness
from args import create_args, parse_ranking_function, parse_data_model
from utils import lp_norms, calc_similarity, weight_avg, get_pred_label, confidence_interval, confidence_interval_bin
import os
from diversity import ATSDiversity, BoostingDiversity, DeepGauge

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_datagen(args):
    gen = DataGen(l2_nums=args.sd_samples, linf_nums=args.sd_samples, linf_rad=args.linf_radius, l2_rad=args.l2_radius)
    return gen

def set_all_seeds(seed):
    # for reproducibility
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def ranking_experiment(args, x, y, fitness, ats, bd, dg, deltas):
    ranking1, method1 = parse_ranking_function(args.ranking_method1, fitness, ats, bd, dg)
    ranking2, method2 = parse_ranking_function(args.ranking_method2, fitness, ats, bd, dg)
    #print(ranking1)
    if args.ranking_method1 == "bd" or args.ranking_method1 == "dg":
        indices1, _ = ranking1(x, x, deltas)
    else:
        indices1, _ = ranking1(x, y, deltas)
    if args.ranking_method2 == "bd" or args.ranking_method2 == "dg":
        indices2, _ = ranking2(x, x, deltas)
    else:
        indices2, _ = ranking2(x, y, deltas)
    sim_scores = calc_similarity(indices1, indices2)
    return sim_scores


def create_gen_rank(args, ranking_method, score_function, ats, bd, dg, gen):
    if args.transformation == "sd":
        delta_gen = gen.lp_gen
    elif args.transformation == "natural":
        delta_gen = gen.natural_gen
    elif args.transformation == "mixed":
        delta_gen = gen.mixed_gen
        
    forward = True
    if ranking_method == "ats":
        ranking = ats.ranking
    if ranking_method == "bd":
        ranking = bd.ranking
    if ranking_method == "dg":
        ranking = dg.ranking
    if ranking_method == "margin_forward":
        ranking = score_function.ranking_margin_forward
    elif ranking_method == "ce_forward":
        ranking = score_function.ranking_ce_forward
    elif ranking_method == "margin_backward":
        ranking = score_function.ranking_margin_backward
        if args.transformation == "mixed":
            ranking = score_function.ranking_margin_mixed_backward
        if args.transformation == "sd":
            delta_gen = score_function.margin_grad_gen
            forward = False
    elif ranking_method == "ce_backward":
        ranking = score_function.ranking_ce_backward
        if args.transformation == "mixed":
            ranking = score_function.ranking_ce_mixed_backward
        if args.transformation == "sd":
            delta_gen = score_function.ce_grad_gen
            forward = False
    elif ranking_method == "margin_mixed":
        ranking = score_function.ranking_margin_mixed
    elif ranking_method == "ce_mixed":
        ranking = score_function.ranking_ce_mixed
    return [ranking, delta_gen, forward]

def testing_experiment(args, model, data, score_function, ats, bd, dg, gen):
    gen_rank_config = create_gen_rank(args, args.testing, score_function, ats, bd, dg, gen)
    ranking, delta_gen, forward = gen_rank_config[0], gen_rank_config[1], gen_rank_config[2]

    acc = []
    for i in range(args.iterations+1):
        acc.append([])
    weights = []
    for x, y in data.valid_loader:
        if args.test_mode == "metamorphic":
            y = get_pred_label(model, x, device)
        weights.append(x.shape[0])
        if args.testing == 'bd' or args.testing == 'dg':
            test = Testing(model, x, y, ranking, args.iterations, args.topk, device, diversity=True)
        else:
            test = Testing(model, x, y, ranking, args.iterations, args.topk, device)
        test.testing(delta_gen, l2_rad=args.l2_radius, linfty_rad=args.linf_radius, forward=forward)
        for i, iter_acc in enumerate(test.accuracy):
            acc[i].append(min(iter_acc))
    min_acc_all = []
    min_acc_all_intv = []
    for iter_acc in acc:
        avg, intv = weight_avg(iter_acc, weights)
        min_acc_all.append(avg)
        min_acc_all_intv.append(intv)
    
    print("Accracy after experiment:", min_acc_all, ", and the corresponding confidence intervals:", min_acc_all_intv)


def train_model(data, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    model.train_model(data.train_loader, optimizer, criterion)
    model.save_model()
    
    acc = model.evaluate(data.valid_loader)
    print(f'Accuracy of the network: {acc} %')

def main(args):
    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists('profiles'):
        os.makedirs('profiles')

    
    set_all_seeds(args.random_seed)
    model_data_config = parse_data_model(args, device)
    model, data, num_class, model_path = model_data_config[0], model_data_config[1], model_data_config[2], model_data_config[3]
    
    if not os.path.exists(model_path):
        train_model(data, model)
    else:
        print("Load trained model...")
        model.load_model()
        if args.clean_evaluate:
            print("Evaluate over clean data...")
            acc = model.evaluate(data.valid_loader)
            print(f'Accuracy of the network on the 10000 test images: {acc} %')

    gen = create_datagen(args)
    profile_path = "profiles/"+args.model+"_"+args.dataset+"_"+str(args.epochs)+".prof"
    score_function = Fitness(model, num_class, device)
    ats = ATSDiversity(model, num_class, device)
    if args.bd_layer == "int":
        bd = BoostingDiversity(model = model, device=device, model_type= args.model, score='Hellinger')
    else:
        bd = BoostingDiversity(model = model, device=device, feature_interm= False, model_type= args.model, score='Hellinger')
    dg = DeepGauge(model=model, device=device, model_type = args.model, loader = data.train_loader, profile_path= profile_path, feature_interm=True)
    if args.experiment == "ranking":
        l2_norms = []
        linfty_norms = []
        similarity = []
        for i, (x, y) in enumerate(data.valid_loader):
            if args.transformation == "sd":
                sd_deltas = gen.lp_gen(x)
                linfty, l2 = lp_norms(sd_deltas)
                linfty_norms += linfty
                l2_norms += l2
                sim_scores = ranking_experiment(args, x, y, score_function, ats, bd, dg, sd_deltas)
                similarity += sim_scores
            elif args.transformation == "natural":
                natural_deltas = gen.natural_gen(x)
                linfty, l2 = lp_norms(natural_deltas)
                linfty_norms += linfty
                l2_norms += l2
                sim_scores = ranking_experiment(args, x, y, score_function, ats, bd, dg, natural_deltas)
                similarity += sim_scores
            elif args.transformation == "mixed":
                natural_deltas = gen.natural_gen(x)
                sd_deltas = gen.lp_gen(x)
                deltas = natural_deltas + sd_deltas
                linfty, l2 = lp_norms(deltas)
                linfty_norms += linfty
                l2_norms += l2
                sim_scores = ranking_experiment(args, x, y, score_function, ats, bd, dg, deltas)
                similarity += sim_scores
        similarity_np, l2_norms_np, linfty_norms_np = np.array(similarity), np.array(l2_norms), np.array(linfty_norms)
        m_sim, int_sim = confidence_interval(similarity_np)
        m_2_norm, int_2_norm = confidence_interval(l2_norms_np)
        m_inf_norm, int_inf_norm = confidence_interval(linfty_norms_np)
        print("l2 norm mean:", m_2_norm, ", confidence interval:", int_2_norm)
        print("linf norm mean:", m_inf_norm, ", confidence interval:", int_inf_norm)
        print("Similarity mean:", m_sim, ", confidence interval:", int_sim)
    if args.experiment == "testing":
        testing_experiment(args, model, data, score_function, ats, bd, dg, gen=gen)
        

if __name__ == '__main__':
    parser = create_args()
    args = parser.parse_args()

    main(args)