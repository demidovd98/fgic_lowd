# coding=utf-8
from __future__ import absolute_import, division, print_function

# WnB:
import wandb
wandb.init(project="fgic_ld", entity="fgic_lowd")
#

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta
import time

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size


### My:
import torch.nn.functional as F
from torchvision import models
#import timm

from utils.utils import *


### SAM:
from _extra.SAM.models.classifier import Classifier
from _extra.SAM.models.method import SAM

from _extra.SAM.src.utils import load_network, load_data



logger = logging.getLogger(__name__)



def setup(args):

    # Prepare dataset
    if args.dataset == "cifar10":
        num_classes=10
        dataset_path = ''
    elif args.dataset == "cifar100":
        num_classes=100
        dataset_path = ''
    elif args.dataset == "soyloc":
        num_classes=200
        dataset_path = ''
    elif args.dataset== "cotton":
        num_classes=80
        dataset_path = ''
    elif args.dataset == "dogs":
        num_classes = 120
        dataset_path = 'Stanford Dogs/Stanford_Dogs'
    elif args.dataset == "CUB":
        num_classes=200
        dataset_path = 'CUB_200_2011/CUB_200_2011/CUB_200_2011'
    elif args.dataset == "cars":
        num_classes=196
        dataset_path = 'Stanford Cars/Stanford Cars'
    elif args.dataset == 'air':
        num_classes = 100
        dataset_path = 'FGVC-Aircraft-2013/fgvc-aircraft-2013b'
    elif args.dataset == 'CRC':
        num_classes = 8
        dataset_path = 'CRC_colorectal_cancer_histology'
    else:
        raise Exception(f"[ERROR] Undefined dataset {args.dataset}") 

    args.data_root = '{}/{}'.format(args.data_root, dataset_path)

    if args.split is not None:
        print(f"[INFO] A {args.split} split is used")


    # Prepare model
    if args.vanilla:
        print("[INFO] A vanilla (unmodified) model is used")

    if args.model_type == "vit":
        config = CONFIGS[args.model_name]
        if args.feature_fusion:
            config.feature_fusion=True
        config.num_token = args.num_token

    if not args.timm_model:
        if args.model_type == "cnn":

            if args.sam:
                if (args.model_name == 'resnet34') or (args.model_name == 'resnet18'):
                    proj_size = 512
                else:
                    proj_size = 2048

                '''
                backbone_name = 'resnet152'
                #backbone_name = 'resnet101'
                #backbone_name = 'resnet50'
                proj_size = 2048

                #backbone_name = 'resnet34'
                #backbone_name = 'resnet18'
                #proj_size = 512
                '''

                #pretrained_path = '~/.torch/models/moco_v2_800ep_pretrain.pth.tar'
                pretrained_path = None

                #pretrained_path = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
                #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

                projector_dim = 1024 # basically number of classes
                
                # Initialize model
                network, feature_dim = load_network(args.model_name)
                model = SAM(network=network, backbone=args.model_name, projector_dim=projector_dim,
                                class_num=num_classes, pretrained=True, pretrained_path=pretrained_path)#.to(args.device)
                classifier = Classifier(proj_size, num_classes)#.to(args.device)   #2048/num of bilinear 2048*16
                
                # mb initialise classifier ?
                # classifier.classifier_layer.apply(init_weights)

                print("[INFO] A pre-trained ECCV ResNet-50 model is used")

            else:
                '''
                #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True) 

                #model = models.resnet18(pretrained=True) #, num_classes=200)
                #model = models.resnet34(pretrained=True) #, num_classes=200)
                model = models.resnet50(pretrained=True) #, num_classes=200)
                ##model = models.resnet50(pretrained=True, zero_init_residual=True) #, num_classes=200)
                #model = models.resnet101(pretrained=True) #, num_classes=200)
                #model = models.resnet152(pretrained=True) #, num_classes=200)
                '''

                #exec(f"model = models.{args.model_name}(pretrained=True)")
                model = eval(f"models.{args.model_name}(pretrained=True)")

                #print(model)

                model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
                
                #model.fc.apply(init_weights) #?
                model.fc.weight.data.normal_(0, 0.01)
                model.fc.bias.data.fill_(0.0)

                print("[INFO] A pre-trained ResNet-50 model is used")

        elif args.model_type == "vit":
            model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, vis=True, smoothing_value=args.smoothing_value, dataset=args.dataset)
            
            if args.pretrained_dir != "":
                print("[INFO] A pre-trained model is used")
                model.load_from(np.load(args.pretrained_dir))
            else:
                print("[INFO] A model will be trained from scratch")
        else:
            raise Exception(f"[ERROR] Undefined model type {args.model_type}") 


    else:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        #     map_location="cpu", check_hash=True,
        # )
        # model.load_state_dict(checkpoint["model"])
        
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        #     map_location="cpu", check_hash=True
        # )
        # msg = model.load_state_dict({x: y for x, y in checkpoint["model"].items() if x not in ["head.weight",
        #                                                                                         "head.bias",
        #                                                                                         "pos_embed"]},
        #                             strict=False)
        # print(msg)


        #model.load_state_dict(torch.load("deit_base_patch16_224-b5f2ef4d.pth", map_location=torch.device('cpu')))
        #model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value)
        # model.load_state_dict(torch.load("deit_base_patch16_224-b5f2ef4d.pth", map_location=torch.device('cpu')))


        #model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

        raise NotImplementedError()
        model = timm.create_model('deit3_base_patch16_224_in21ft1k', img_size=400, pretrained=True, num_classes=200) #.cuda()

        #deit_base_patch16_224
        #deit3_base_patch16_224
        #deit3_base_patch16_224_in21ft1k

        
        # #deit_base_patch16_224-b5f2ef4d.pth
        # #deit_3_base_224_1k.pth 
        # #deit_3_base_224_21k.pth
        # checkpoint = torch.load("deit_base_patch16_224-b5f2ef4d.pth", map_location=torch.device('cpu'))
        
        # # torch.hub.load_state_dict_from_url(
        # #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        # #     map_location="cpu", check_hash=True
        # # )
        # msg = model.load_state_dict({x: y for x, y in checkpoint["model"].items() if x not in ["head.weight",
        #                                                                                         "head.bias",
        #                                                                                         "pos_embed"]},
        #                             strict=False)
        # print(msg)
        
        #print(model)


    if args.sam:
        model.to(args.device)
        classifier.to(args.device)

        print(model)
        print(classifier)
    
        print("[INFO] Backbone params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6 / 2))
        print("[INFO] Classifier params: {:.2f}M".format(sum(p.numel() for p in classifier.parameters()) / 1e6))
    else:
        model.to(args.device)
        num_params = count_parameters(model)

        print(model)

        save_model(args, model, logger)

        if args.model_type == "vit": logger.info("{}".format(config))
        logger.info("[INFO] Training parameters %s", args)
        logger.info("[INFO] Total Parameters: \t%2.1fM" % num_params)
        print(num_params)

    if args.sam:
        return args, model, classifier, num_classes
    else:
        return args, model, num_classes



#def valid(args, model, writer, test_loader, global_step):
def valid(args, model, writer, test_loader, global_step, classifier=None):

    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    if args.sam:
        model.eval()
        classifier.eval()
    else:
        model.eval()
    
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        #wandb.log({"step": step})

        batch = tuple(t.to(args.device) for t in batch)
        #x, y = batch

        if args.saliency:
            # With mask:
            x, y, mask = batch
        else:
            x, y = batch

        if args.dataset == 'air': # my
            y = y.view(-1)

        with torch.no_grad():
            if args.sam:
                feat_labeled = model(x)[0]
                logits = classifier(feat_labeled.cuda())[0] #feat_labeled/bp_out_feat

            else:
                if args.saliency:
                    #logits = model(x)[0]

                    # With mask:
                    y_temp = None
                    x_crop_temp = None
                    mask_crop_temp =None

                    logits = model(x, x_crop_temp, y_temp, mask, mask_crop_temp)[0]
                    #logits, attn_weights = model(x, y_temp, mask)
                else:              
                    logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            #eval_loss = loss_fct(logits.view(-1, 200), y.view(-1))

            # transFG:
            #eval_loss = eval_loss.mean() # for contrastive learning!!!

            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    
    print("Valid Accuracy:", accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    wandb.log({"acc_test": accuracy})
    
    return accuracy



#def train(args, model):
def train(args, model, classifier=None, num_classes=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
        
    best_step=0
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    #set_seed(args) # my

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    if args.model_type == "cnn":
        lr_ratio = args.lr_ratio # ? round(100 / args.split) #0.1 #5.0 #10.0 # 1.0, 2.0 # useful for CUB
        lr_ratio_feats = 2.0

        #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        if not args.sam:
            layer_names = []
            for idx, (name, param) in enumerate(model.named_parameters()):
                layer_names.append(name)
                #print(f'{idx}: {name}')
        
            parameters = []
            # store params & learning rates
            for idx, name in enumerate(layer_names):
                lr = args.learning_rate
                
                # append layer parameters
                if (name == "fc.weight") or (name == "fc.bias"):
                    lr = args.learning_rate * lr_ratio
                    print(f'{idx}: lr = {lr:.6f}, {name}')
                # elif (150 <= idx <= 158): # 150 (last block of layer 4), 129 (full layer 4)
                #     lr = args.learning_rate * lr_ratio_feats
                #     print(f'{idx}: lr = {lr:.6f}, {name}')
                else:
                    lr = args.learning_rate

                parameters += [{'params': [p for n, p in model.named_parameters() if ((n == name) and (p.requires_grad))],
                                'lr':      lr}]
                #print(f'{idx}: lr = {lr:.6f}, {name}')
            
            optimizer = torch.optim.SGD(parameters, 
                        lr= args.learning_rate, 
                        momentum=0.9, 
                        weight_decay=args.weight_decay, 
                        nesterov=True)
            
            '''
            # optimizer = torch.optim.SGD(model.parameters(), 
            #             lr= args.learning_rate, 
            #             momentum=0.9, 
            #             weight_decay=args.weight_decay, 
            #             nesterov=True, # True
            #             )

            # optimizer = torch.optim.SGD([
            #             {'params': model.base.parameters()},
            #             {'params': model.fc.parameters(), 'lr': args.learning_rate * lr_ratio}, ], 
            #             lr= args.learning_rate, 
            #             momentum=0.9, 
            #             weight_decay=args.weight_decay, 
            #             nesterov=True)

            #milestones = [6000, 12000, 18000, 24000, 30000]
            #milestones = [8000, 16000, 24000, 32000, 40000]
            #milestones = [10000, 20000, 30000, 40000]    

            # milestones = [ int(args.num_steps * 0.2), # 8`000
            #             int(args.num_steps * 0.4), # 16`000
            #             int(args.num_steps * 0.6), # 24`000
            #             int(args.num_steps * 0.8), # 32`000
            #             int(args.num_steps * 1.0) ] # 40`000              
            '''

            milestones = [ int(args.num_steps * 0.5), # 20`000
                        int(args.num_steps * 0.75), # 30`000
                        int(args.num_steps * 0.90), # 36`000
                        int(args.num_steps * 0.95), # 38`000
                        int(args.num_steps * 1.0) ] # 40`000            

            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

        else:
            optimizer = torch.optim.SGD([
                        {'params': model.parameters()},
                        {'params': classifier.parameters(), 'lr': args.learning_rate * lr_ratio}, ], 
                        lr= args.learning_rate, 
                        momentum=0.9, 
                        weight_decay=args.weight_decay, 
                        nesterov=True)
            
            milestones = [6000, 12000, 18000, 24000, 30000]

            '''
            # milestones = [ int(args.num_steps * 0.5), # 20`000
            #             int(args.num_steps * 0.75), # 30`000
            #             int(args.num_steps * 0.90), # 36`000
            #             int(args.num_steps * 0.95), # 38`000
            #             int(args.num_steps * 1.0) ] # 40`000

            # milestones = [ int(args.num_steps * 0.25), # 10`000
            #             int(args.num_steps * 0.5), # 20`000
            #             int(args.num_steps * 0.75), # 30`000
            #             int(args.num_steps * 0.9), # 36`000
            #             int(args.num_steps * 1.0) ] # 40`000
            # milestones = [ int(args.num_steps * 0.2), # 8`000
            #             int(args.num_steps * 0.4), # 16`000
            #             int(args.num_steps * 0.6), # 24`000
            #             int(args.num_steps * 0.8), # 32`000
            #             int(args.num_steps * 1.0) ] # 40`000
            '''

            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
        
        t_total = args.num_steps #
        
    if args.model_type == "vit":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
        '''
        # for hybrid
        optimizer = torch.optim.SGD([{'params':model.transformer.parameters(),'lr':args.learning_rate},
                                    {'params':model.head.parameters(),'lr':args.learning_rate}],
                                    lr=args.learning_rate,momentum=0.9,weight_decay=args.weight_decay)
        '''
        
        t_total = args.num_steps
        if args.decay_type == "cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        else:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)


    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    start_time = time.time()
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0

    while True:
        if args.sam:
            model.train(True)
            classifier.train(True)
            #optimizer.zero_grad()
        else:
            #model.train()
            model.train(True)

        # for param_group in optimizer.param_groups:
        #     print(param_group)
        #     print(param_group['lr'])

        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])

        all_preds, all_label = [], []

        for step, batch in enumerate(epoch_iterator):
            #print(step)
            wandb.log({"step": step})

            batch = tuple(t.to(args.device) for t in batch)
            # x, y = batch
            # loss, logits = model(x, y)
            # #loss = loss.mean() # for contrastive learning

            # print("____________________________")
            # print(args.vanilla)
            # print(args.aug_type)
                

            if args.saliency:
                # With mask:
                if args.aug_type == "double_crop":
                    x, x_crop, x_crop2, y, mask, mask_crop = batch
                elif args.aug_type == "single_crop":
                    x, x_crop, y, mask, mask_crop = batch
                else:                
                    if args.vanilla:
                        x, y, mask = batch
                    else:
                        raise NotImplementedError()
            else:
                if args.aug_type == "double_crop":
                    x, x_crop, x_crop2, y = batch
                elif args.aug_type == "single_crop":
                    x, x_crop, y = batch
                else:
                    if args.vanilla:
                        x, y = batch
                    else:
                        raise NotImplementedError()

            if args.dataset == 'air': # my
                y = y.view(-1)

            loss_fct = torch.nn.CrossEntropyLoss()
            #refine_loss_criterion = FocalLoss() # F.kl_div()
            refine_loss_criterion = torch.nn.CrossEntropyLoss()
            # for pytroch >= 1.6.0 #kl_loss = F.kl_div(reduction="batchmean", log_target=True)

            if args.sam:
                # y_test = model(x)
                # print(len(y_test))
                # print(y_test[0].size())
                # print(y_test[1].size())
                # print(y_test[2].size())

                feat_labeled = model(x)[0] 
                #print(feat_labeled.size())
                
                logits = classifier(feat_labeled.cuda())[0]  #feat_labeled/bp_out_feat

                if not args.vanilla:
                    feat_labeled_crop = model(x_crop)[0]
                    logits_crop = classifier(feat_labeled_crop.cuda())[0] #feat_labeled/bp_out_feat

                    ce_loss = loss_fct(logits.view(-1, num_classes), y.view(-1))

                    if args.aug_type == "double_crop":
                        feat_labeled_crop2 = model(x_crop2)[0]
                        logits_crop2 = classifier(feat_labeled_crop2.cuda())[0] #feat_labeled/bp_out_feat

                        ##refine_loss = refine_loss_criterion(logits_crop.view(-1, 200), logits.argmax(dim=1).view(-1))  #.view(-1, self.num_classes)) #.long())
                        
                        #refine_loss = refine_loss_criterion(logits_crop.view(-1, 200), logits_crop2.argmax(dim=1).view(-1))  #.view(-1, self.num_classes)) #.long())
                        
                        #refine_loss = 3.0 * abs( F.kl_div(logits_crop.log_softmax(dim=-1), logits_crop2.softmax(dim=-1), reduction='batchmean') ) #reduction='sum')
                        
                        #refine_loss = abs( F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='batchmean') ) #reduction='sum')
                        #refine_loss = 0.00001 * abs( F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='batchmean') ) #reduction='sum')
                        #refine_loss = 0.0001 * abs( F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='batchmean') ) #reduction='sum')
                        #refine_loss = 0.00005 * abs( F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='batchmean') ) #reduction='sum')
                        refine_loss = 0.00001 * abs( F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='batchmean') ) #reduction='sum')

                        #refine_loss = 0.00005 * abs( F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='batchmean', log_target=True) )
                        #refine_loss = 0.0001 * abs( F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='sum') )
                        #refine_loss = 0.1 * abs( F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='none') )

                        #refine_loss = 0.1 * abs( F.kl_div(logits_crop.log_softmax(dim=-1), logits_crop2.softmax(dim=-1), reduction='sum') ) #reduction='sum')
                        #refine_loss = 0.1 * abs( F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='sum') ) #reduction='sum')

                        #refine_loss = 0.1 * abs( F.kl_div(logits_crop.log_softmax(dim=-1), logits_crop2.softmax(dim=-1), reduction='mean') ) #reduction='sum')
                        #refine_loss = 0.1 * abs( F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='mean') ) #reduction='sum')

                    elif args.aug_type == "single_crop":
                        #refine_loss = 0.00005 * abs( F.kl_div(feat_labeled_crop, feat_labeled, reduction='batchmean') ) #reduction='sum')
                        refine_loss = refine_loss_criterion(logits_crop.view(-1, num_classes), logits.argmax(dim=1).view(-1))  #.view(-1, self.num_classes)) #.long())
                    else:
                        raise NotImplementedError()

                    if torch.isinf(refine_loss):
                        print("[INFO]: Skip Refine Loss")
                        loss = ce_loss
                    else:
                        loss = (0.5 * ce_loss) + (0.5 * refine_loss * 0.1) #0.01
                        #loss = (0.5 * ce_loss) + (0.5 * refine_loss * 0.3) # 0.5, 0.3

                    if (step % 50 == 0): print("[INFO]: ce loss:", ce_loss.item(), "Refine loss:", refine_loss.item(), "Final loss:", loss.item())

                else:
                    ce_loss = loss_fct(logits.view(-1, num_classes), y.view(-1))
                    loss = ce_loss


            else:

                if args.saliency:
                    #loss, logits = model(x, y)
                    ##loss = loss.mean() # for contrastive learning

                    loss, logits = model(x, x_crop, y, mask, mask_crop)

                else:
                    logits, feat_labeled = model(x)

                    if not args.vanilla:

                        logits_crop, feat_labeled_crop = model(x_crop)

                        ce_loss = loss_fct(logits.view(-1, num_classes), y.view(-1))

                        if args.aug_type == "double_crop":
                            logits_crop2, feat_labeled_crop2 = model(x_crop2)

                            #refine_loss = 0.0001 * abs( F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='batchmean') ) #reduction='sum')
                            #refine_loss = 0.00001 * abs( F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='batchmean') ) #reduction='sum')
                            #refine_loss = 0.00005 * abs( F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='batchmean') ) #reduction='sum')

                            #refine_loss = 0.00005 * F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='sum') #reduction='sum')
                            #refine_loss = 0.00005 * F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='batchmean') #reduction='sum')

                            #refine_loss = F.kl_div(feat_labeled_crop.softmax(dim=-1).log(), feat_labeled_crop2.softmax(dim=-1), reduction='sum') # from SAM
                            #refine_loss = F.kl_div(feat_labeled_crop.softmax(dim=-1).log(), feat_labeled_crop2.softmax(dim=-1), reduction='batchmean')
                            
                            '''
                            print()
                            print("____Before")
                            print(feat_labeled_crop)
                            print(feat_labeled_crop.size())
                            print(feat_labeled_crop2)
                            print(feat_labeled_crop2.size())

                            feat_labeled_crop_test = feat_labeled_crop.softmax(dim=-1)
                            feat_labeled_crop_test2 = feat_labeled_crop2.softmax(dim=-1)

                            print()
                            print("____After")
                            print(feat_labeled_crop_test)
                            print(feat_labeled_crop_test2)

                            # print()
                            # print("____After log")
                            # print(feat_labeled_crop_test.log())

                            refine_loss = F.kl_div(feat_labeled_crop_test, feat_labeled_crop_test2, reduction='batchmean')
                            print()
                            print(refine_loss)
                            print(refine_loss.item())
                            '''

                            # no log (experimental); originally the input better be in log space (gives negative loss but acc better for some reason) 
                            #refine_loss = F.kl_div(feat_labeled_crop.softmax(dim=-1), feat_labeled_crop2.softmax(dim=-1), reduction='batchmean')
                            #refine_loss = F.kl_div(feat_labeled_crop.softmax(dim=-1), feat_labeled_crop2.softmax(dim=-1), reduction='sum')
                            
                            # main
                            #refine_loss = F.kl_div(feat_labeled_crop.log_softmax(dim=-1), feat_labeled_crop2.softmax(dim=-1), reduction='batchmean') #reduction='sum')
                            refine_loss = F.kl_div(feat_labeled_crop.log_softmax(dim=-1), feat_labeled_crop2.softmax(dim=-1), reduction='sum') #reduction='sum')


                            #refine_loss = F.kl_div(feat_labeled_crop.log_softmax(dim=-1), feat_labeled_crop2.log_softmax(dim=-1), reduction='batchmean') #, log_target=True) #reduction='sum')
                            #for pytroch >= 1.6.0 #refine_loss = kl_loss(feat_labeled_crop.log_softmax(dim=-1), feat_labeled_crop2.log_softmax(dim=-1)) #, reduction='batchmean', log_target=True) #reduction='sum')


                            #refine_loss = F.kl_div(logits_crop.log_softmax(dim=-1), logits_crop2.softmax(dim=-1), reduction='sum') #reduction='sum')
                            #refine_loss = F.kl_div(logits_crop.log_softmax(dim=-1), logits_crop2.softmax(dim=-1), reduction='batchmean') #reduction='sum')
                            
                            
                            #refine_loss = abs( F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='batchmean') ) #reduction='sum')                            
                            #refine_loss = refine_loss * (ce_loss/refine_loss)

                            #refine_loss = 0.00005 * abs( F.kl_div(feat_labeled_crop, feat_labeled_crop2, reduction='sum') ) #reduction='sum')

                            #refine_loss = 0.00005 * abs( F.kl_div(feat_labeled_crop2, feat_labeled_crop, reduction='batchmean') ) #reduction='sum')
                            #refine_loss = 0.00005 * abs( F.kl_div(logits_crop, logits_crop2, reduction='batchmean') ) #reduction='sum')
                            #refine_loss = abs( F.kl_div(logits_crop, logits_crop2, reduction='batchmean') ) #reduction='sum')

                            #refine_loss = refine_loss_criterion(logits_crop.view(-1, num_classes), logits_crop2.argmax(dim=1).view(-1))  #.view(-1, self.num_classes)) #.long())

                        elif args.aug_type == "single_crop":

                            #ce_loss = loss_fct(logits_crop.view(-1, self.num_classes), labels.view(-1))

                            ##refine_loss = F.kl_div(logits_crop.softmax(dim=-1).log(), logits.softmax(dim=-1), reduction='batchmean') #reduction='sum')
                            #refine_loss = F.kl_div(logits_crop.log_softmax(dim=-1), logits.softmax(dim=-1), reduction='batchmean') #reduction='sum')

                            #ce_loss = loss_fct(logits.view(-1, num_classes), y.view(-1))
                            ##ce_loss = loss_fct(logits, y)


                            # print(logits.size())
                            # print(logits.argmax(dim=1).view(-1).size())

                            # print(logits_crop.size())
                            # print(logits_crop.view(-1, num_classes).size())

                            # print("----")

                            refine_loss = refine_loss_criterion(logits_crop.view(-1, num_classes), logits.argmax(dim=1).view(-1))  #.view(-1, self.num_classes)) #.long())
                            #refine_loss = refine_loss_criterion(logits_crop, logits.argmax(dim=1))  #.view(-1, self.num_classes)) #.long())

                        else:
                            raise NotImplementedError()


                        if torch.isinf(refine_loss):
                            print("[INFO]: Skip Refine Loss")
                            loss = ce_loss
                        else:
                            #loss = ce_loss + refine_loss #0.01
                            #loss = ce_loss + (0.1 * refine_loss) #0.01
                            #loss = ce_loss + (0.01 * refine_loss) #0.01

                            loss = (0.5 * ce_loss) + (0.5 * refine_loss * args.dist_coef) #0.01 # main

                            #loss = (0.5 * ce_loss) + (0.5 * refine_loss * 10.0) #0.01 # main
                            #loss = (0.5 * ce_loss) + (0.5 * refine_loss) #0.01

                            #loss = (0.5 * ce_loss) + (0.5 * refine_loss * 0.5) #0.01
                            #loss = (0.5 * ce_loss) + (0.5 * refine_loss * 0.1) #0.01 # main
                            #loss = (0.5 * ce_loss) + (0.5 * refine_loss * 0.01) #0.01     
                                                    
                            #loss = ce_loss + (refine_loss * 0.1) #0.01

                            #loss = (0.5 * ce_loss) + (0.5 * refine_loss * 0.3) #0.01

                        #loss = criterion(logits, y)
                        if (step % 50 == 0): print("[INFO]: ce loss:", ce_loss.item(), "Refine loss:", refine_loss.item(), "Final loss:", loss.item())
                        wandb.log({"ce_loss": ce_loss.item()})
                        wandb.log({"dist_loss": refine_loss.item()})

                    else:
                        #num_classes=8

                        # print(logits.size())
                        # print(num_classes)
                        # print(y.size())

                        ce_loss = loss_fct(logits.view(-1, num_classes), y.view(-1))
                        loss = ce_loss

            # transFG:
            #loss = loss.mean() # for contrastive learning !!!
            #

            preds = torch.argmax(logits, dim=-1)

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)

                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)

                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    if args.sam:
                        accuracy = valid(args, model, writer, test_loader, global_step, classifier)
                    else:
                        accuracy = valid(args, model, writer, test_loader, global_step)
                        
                    if best_acc < accuracy:
                        save_model(args, model, logger)
                        best_acc = accuracy
                        best_step = global_step
                    logger.info("best accuracy so far: %f" % best_acc)
                    logger.info("best accuracy in step: %f" % best_step)
                    model.train()

                if global_step % t_total == 0:
                    break

        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)
        accuracy = torch.tensor(accuracy).to(args.device)
        dist.barrier()
        train_accuracy = reduce_mean(accuracy, args.nprocs)
        train_accuracy = train_accuracy.detach().cpu().numpy()

        writer.add_scalar("train/accuracy", scalar_value=train_accuracy, global_step=global_step)
        wandb.log({"acc_train": train_accuracy})

        logger.info("train accuracy so far: %f" % train_accuracy)
        logger.info("best valid accuracy in step: %f" % best_step)
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    end_time = time.time()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "soyloc", "cotton", "CUB", "dogs", "cars", "air", "CRC"], 
                        default="CUB",
                        help="Which downstream task.")
    
    parser.add_argument("--model_type", choices=["vit", "cnn"],
                        default="cnn",
                        help="Which architecture to use.")
    parser.add_argument("--model_name", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16",
                                                 "resnet18", "resnet34", "resnet50",
                                                 "resnet101", "resnet152"],
                        default="resnet50",
                        help="Which specific model to use.")

    #parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
    parser.add_argument("--pretrained_dir", type=str, default="",
                        help="Where to search for pretrained ViT models.")

    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--resize_size", default=600, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--num_token", default=12, type=int,
                        help="the number of selected token in each layer, 12 for soy.loc, cotton and cub, 24 for dog.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--feature_fusion', action='store_true',
                        help="Whether to use feature fusion")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    
    parser.add_argument('--vanilla', action='store_true',
                        help="Whether to use the vanilla model")
    parser.add_argument("--split", required=True,
                        choices=["1i", "1p", "2", "3", "4", "5", "10", "15", "30", "50", "100"],
                        help="Name of the split")

    parser.add_argument("--aug_type", choices=["single_crop", "double_crop", "none"],
                        default="double_crop",
                        help="Which architecture to use.")

    parser.add_argument('--sam', action='store_true',
                        help="Whether to use the SAM training setup")
    # parser.add_argument('--cls_head', action='store_true',
    #                     help="Whether to use classification head as a separate module")
    parser.add_argument('--timm_model', action='store_true',
                        help="Whether to use pre-trained model from the timm library")        
    parser.add_argument('--saliency', action='store_true',
                        help="Whether to use saliency information (foreground/nackground mask)")

    parser.add_argument("--lr_ratio", default=1.0, type=float, required=True,
                        help="Learning rate ratio for the last classification layer.")
    parser.add_argument("--dist_coef", default=0.1, type=float, required=True,
                        help="Coefficient of the distillation loss contribution.")
    

    #parser.add_argument('--data_root', type=str, default='./data') # Original
    #parser.add_argument('--data_root', type=str, default='/l/users/20020067/Datasets') # local
    parser.add_argument('--data_root', type=str, default='/l/users/cv-805/Datasets') # shared

    #parser.add_argument('--data_root', type=str, default='/l/users/20020067/Datasets/CUB_200_2011/CUB_200_2011/CUB_200_2011') # CUB
    #parser.add_argument('--data_root', type=str, default='/l/users/20020067/Datasets/Stanford Dogs/Stanford_Dogs') # dogs
    #parser.add_argument('--data_root', type=str, default='/l/users/20020067/Datasets/Stanford Cars/Stanford Cars') # cars
    #parser.add_argument('--data_root', type=str, default='/l/users/20020067/Datasets/CRC_colorectal_cancer_histology') # CRC Medical
    #parser.add_argument('--data_root', type=str, default='/l/users/20020067/Datasets/FGVC-Aircraft-2013/fgvc-aircraft-2013b') # aircraft


    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")
    
    args = parser.parse_args()
    
    #args.data_root = '{}/{}'.format(args.data_root, args.dataset)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    if args.sam:
        args, model, classifier, num_classes = setup(args)
        wandb.watch(model)

        # Training
        train(args, model, classifier, num_classes)

    else:    
        args, model, num_classes = setup(args)
        wandb.watch(model)
        #torch.autograd.set_detect_anomaly(True)

        # Training
        train(args, model, num_classes=num_classes)



if __name__ == "__main__":
    main()
