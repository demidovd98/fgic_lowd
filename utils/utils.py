import os
import random
import numpy as np
import torch
import torch.distributed as dist



class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    assert preds.shape == labels.shape, "preds, labels shapes are not equal"
    return (preds == labels).mean()


def save_model(args, model, logger):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    #
    #torch.cuda.manual_seed(args.seed)
    #

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class DropoutMC(torch.nn.Module):
    def __init__(self, p: float, activate=False):
        super().__init__()
        self.activate = activate
        self.p = p
        self.p_init = p

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.dropout(
            x, self.p, training=self.training or self.activate
        )

def convert_dropouts(model, inference_prob=0.1):
    """This function replace all model dropouts with custom dropout layer."""
    dropout_ctor = lambda p, activate: DropoutMC(
        p=inference_prob, activate=False
    )
    convert_to_mc_dropout(model, {"Dropout": dropout_ctor, "StableDropout": dropout_ctor})

def convert_to_mc_dropout(
    model, substitution_dict
):
    for i, layer in enumerate(list(model.children())):
        proba_field_name = "dropout_rate" if "flair" in str(type(layer)) else "p"
        module_name = list(model._modules.items())[i][0]
        layer_name = layer._get_name()
        proba_field_name = "drop_prob" if layer_name == "StableDropout" else proba_field_name #DeBERTA case
        if layer_name in substitution_dict.keys():
            model._modules[module_name] = substitution_dict[layer_name](
                p=getattr(layer, proba_field_name), activate=False
            )
        else:
            convert_to_mc_dropout(model=layer, substitution_dict=substitution_dict)

def activate_mc_dropout(
    model: torch.nn.Module, activate: bool, random: float = 0.0, verbose: bool = False
):
    for layer in model.children():
        if isinstance(layer, DropoutMC):
            if verbose:
                print(layer)
                print(f"Current DO state: {layer.activate}")
                print(f"Switching state to: {activate}")
            layer.activate = activate
            if activate and random:
                layer.p = random
            if not activate:
                layer.p = layer.p_init
        else:
            activate_mc_dropout(
                model=layer, activate=activate, random=random, verbose=verbose
            )

def entropy(x):
    return np.sum(-x * np.log(np.clip(x, 1e-8, 1)), axis=-1)

def bald_func(sampled_probabilities):
    predictive_entropy = entropy(np.mean(sampled_probabilities, axis=1))
    expected_entropy = np.mean(entropy(sampled_probabilities), axis=1)

    return predictive_entropy - expected_entropy

def probability_variance(sampled_probabilities, mean_probabilities=None):
    """Expectes sampled_probabilites of shape [B, T, C]
        Where B is batch size
                T is number of models made inference 
                C is number of classes predicted 
    """
    if mean_probabilities is None:
        mean_probabilities = np.mean(sampled_probabilities, axis=1)
    mean_probabilities = np.expand_dims(mean_probabilities, axis=1)

    return ((sampled_probabilities - mean_probabilities) ** 2).mean(1).sum(-1)

def sampled_max_prob(sampled_probabilities):
    """Expectes sampled_probabilites of shape [B, T, C]
        Where B is batch size
                T is number of models made inference 
                C is number of classes predicted 
    """
    mean_probabilities = np.mean(sampled_probabilities, axis=1)
    top_probabilities = np.max(mean_probabilities, axis=-1)
    return 1 - top_probabilities

def rcc_auc(conf, risk, return_points=False):
    # risk-coverage curve's area under curve
    n = len(conf)
    cr_pair = list(zip(conf, risk))
    cr_pair.sort(key=lambda x: x[0], reverse=True)

    cumulative_risk = [cr_pair[0][1]]
    for i in range(1, n):
        cumulative_risk.append(cr_pair[i][1] + cumulative_risk[-1])

    points_x = []
    points_y = []

    auc = 0
    for k in range(n):
        auc += cumulative_risk[k] / (1 + k)
        points_x.append((1 + k) / n)  # coverage
        points_y.append(cumulative_risk[k] / (1 + k))  # current avg. risk

    if return_points:
        return auc, points_x, points_y
    else:
        return auc
