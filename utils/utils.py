import os
import random
import numpy as np
import torch
import torch.distributed as dist

#import time
from datetime import datetime

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
    #model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name) # owerrites existing checkpoints!

    if args.checkpoint_name == "": # first initialisation
        #args.checkpoint_name = ( "%s_checkpoint.bin" % args.name )
        args.checkpoint_name = ( "%s_checkpoint_%s.bin" % (args.name, datetime.now()) )  # date append

        model_checkpoint = os.path.join(args.output_dir, args.checkpoint_name)

        # # if os.path.isfile(model_checkpoint): # basic (unnecessary now)
        # #     model_checkpoint = os.path.join(args.output_dir, "%s_new_checkpoint.bin" % args.name)

        # while(os.path.isfile(model_checkpoint)): # windows-like (unnecessary now)
        #     iterator = 1
        #     args.checkpoint_name = ( "%s_checkpoint_%s.bin" % (args.name, iterator) )
        #     model_checkpoint = os.path.join(args.output_dir, args.checkpoint_name)
        #     iterator += 1

        if os.path.isfile(model_checkpoint): # date append (unnecessary now)
            args.checkpoint_name = ( "%s_checkpoint_%s.bin" % (args.name, datetime.now()) )
            model_checkpoint = os.path.join(args.output_dir, args.checkpoint_name)
    else:
        model_checkpoint = os.path.join(args.output_dir, args.checkpoint_name)


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