import os
import random
import numpy as np
import torch
import torch.distributed as dist


class DistillScheduler(object):

    def __init__(self, args):
        self.args = args
        self.total_steps = args.num_steps
        self.max_coeff = args.dist_coef  # Assuming this is the initial (maximum) value
        self.min_coeff = args.min_dist_coeff
        self.decay_rate = args.dist_coef_decay_rate

        self.decay_per_step = (self.max_coeff - self.min_coeff) / self.total_steps
        self.dist_coeff = self.max_coeff

    def step(self, global_step):
        """
        Update the coefficient based on the current step.
        
        Args:
            global_step (int): Current global step in the training process.
        """

        if self.args.dist_coef_dacay_type == 'decay': 
            self.linear_decay(global_step)
        elif self.args.dist_coef_dacay_type == 'increase':
            self.linear_increase(global_step)
        elif self.args.dist_coef_dacay_type == 'constant':
            self.dist_coeff = self.max_coeff
        else:
            raise ValueError("Unknown distillation coefficient schedule: %s" % self.args.decay)
    
    def linear_increase(self, global_step):
        # Example: Linear decay
        decay = (self.total_steps - global_step) * self.decay_rate / self.total_steps
        # print("decay: ", decay)
        self.dist_coeff = max(self.min_coeff, self.dist_coeff - decay)

    def linear_decay(self, global_step):
        self.dist_coeff = max(self.min_coeff, self.max_coeff - global_step * self.decay_per_step) 


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