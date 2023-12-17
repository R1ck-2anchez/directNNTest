import argparse
import model
import data

def create_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", 
        nargs='?', 
        const="vgg", 
        default="vgg", 
        choices=['vgg', 'resnet9', 'resnet18', 'lenet1', 'lenet5'], 
        help="which model to use")
    
    parser.add_argument("--dataset", 
        nargs='?', 
        const="cifar10", 
        default="cifar10", 
        choices=['cifar10', 'cifar100', 'svhn', 'mnist'], 
        help="which dataset to use")
    
    parser.add_argument("--batch_size", 
        type=int,
        default=128,
        help="how large is a batch during training")
    
    parser.add_argument("--test_batch_size", 
        type=int,
        default=1000,
        help="how large is a batch during training")

    parser.add_argument("--experiment", 
        nargs='?', 
        const="ranking", 
        default="ranking", 
        choices=['ranking', 'testing'], 
        help="which experiment to run")
    
    parser.add_argument("--ranking_method1", 
        nargs='?', 
        const="margin_forward", 
        default="margin_forward", 
        choices=['margin_forward', 'ce_forward', 'margin_backward', 'ce_backward','random','ats', 'bd', 'dg'], 
        help="which ranking method to compare")
    
    parser.add_argument("--ranking_method2", 
        nargs='?', 
        const="ce_forward", 
        default="ce_forward", 
        choices=['margin_forward', 'ce_forward', 'margin_backward', 'ce_backward', 'random', 'margin_forward_reverse','ats', 'bd', 'dg'], 
        help="which ranking method to compare")

    parser.add_argument("--testing", 
        nargs='?', 
        const="forward", 
        default="forward", 
        choices=['margin_forward', 'ce_forward', 'margin_backward', 'ce_backward', 'margin_mixed', 'ce_mixed', 'ats', 'bd', 'dg'], 
        help="which testing method to use")
    
    parser.add_argument("--sd_samples", 
        type=int,
        default=20,
        help="how many sampels of lp perturbations are used")
    
    parser.add_argument("--linf_radius", 
        type=float,
        default=0.006, 
        help="the radius for l_infty perturbations")
    
    parser.add_argument("--l2_radius", 
        type=float,
        default=0.03, 
        help="the radius for l_2 perturbations")
    
    parser.add_argument("--sd", 
        nargs='?', 
        const="forward", 
        default="forward", 
        choices=['forward', 'ce_backward', 'nat_backward'], 
        help="which testing method to use")
    
    parser.add_argument('--clean_evaluate', 
        action='store_true', 
        help="whether to evaluate the trained data")
    
    parser.add_argument("--epochs", 
        type=int,
        default=100,
        help="how many epochs used for training")
    
    parser.add_argument("--topk", 
        type=int,
        default=1,
        help="how many topk ranked inputs to retain")
    
    parser.add_argument("--iterations", 
        type=int,
        default=5,
        help="how many iterations used to test the model")
    
    parser.add_argument("--transformation", 
        nargs='?', 
        const="sd", 
        default="sd", 
        choices=['sd', 'natural', 'mixed'], 
        help="which testing method to use")
    
    parser.add_argument("--random_seed", 
        type=int,
        default=0,
        help="fix random seed to ensure reproducibility")
    
    parser.add_argument("--bd_layer", 
        nargs='?', 
        const="int", 
        default="int", 
        choices=['int', 'final'], 
        help="which layer to use for boosting diversity")
    
    parser.add_argument("--test_mode", 
        nargs='?', 
        const="truth", 
        default="truth", 
        choices=['truth', 'metamorphic'], 
        help="which testing mode to use")
    
    return parser

def parse_ranking_function(choice, score_function, ats, bd, dg):
    if choice == "margin_forward":
        return score_function.ranking_margin_forward, "margin forward ranking"
    if choice == "ce_forward":
        return score_function.ranking_ce_forward, "ce forward ranking"
    if choice == "margin_backward":
        return score_function.ranking_margin_backward, "margin backward ranking"
    if choice == "ce_backward":
        return score_function.ranking_ce_backward, "ce backward ranking"
    if choice == "random":
        return score_function.ranking_random, "random ranking"
    if choice == "margin_forward_reverse":
        return score_function.ranking_margin_forward_reversed, "margin forward ranking reversed"
    if choice == "ats":
        return ats.ranking, "ATS ranking method"
    if choice == "bd":
        return bd.ranking, "Boosting diversity ranking method"
    if choice == "dg":
        return dg.ranking, "Geep Gauge method"

def parse_model_path(args):
    return "models/"+args.model+"_"+args.dataset+"_"+str(args.epochs)+".pth"


def parse_data_model(args, device):
    if args.dataset == "mnist":
        input_dim = 28
    else:
        input_dim = 32
    model_path = parse_model_path(args)
    dataset = getattr(data, args.dataset.upper())(args.batch_size, input_dim, args.test_batch_size)
    N_CLASSES = dataset.__num_classes__()
    net = getattr(model, args.model.upper())(args.epochs, model_path, device, N_CLASSES).to(device)
    return [net, dataset, N_CLASSES, model_path]
