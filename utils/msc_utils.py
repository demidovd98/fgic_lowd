import os
import random
import torch
import numpy as np
import torch.distributed as dist

from torchvision import models
from models.modeling import CONFIGS, VisionTransformer
from _extra.SAM.models.method import SAM
from _extra.SAM.models.classifier import Classifier 
from _extra.SAM.src.utils import load_network 

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

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

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    #
    #torch.cuda.manual_seed(args.seed)
    #

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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


def setup(args, logger):

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
        raise Exception("No dataset chosen") 

    args.data_root = '{}/{}'.format(args.data_root, dataset_path)


    if args.split is not None:
        print(f"[INFO] A {args.split} split is used")

    if args.vanilla:
        print("[INFO] A vanilla (unmodified) model is used")


    # Prepare model

    if args.model_type == "ViT":
        config = CONFIGS[args.model_name]
        if args.feature_fusion:
            config.feature_fusion=True
        config.num_token = args.num_token


    timm_model = False #True
    resnet50 = True #TrueVisionTransformer
    SAM_check = False #True

    if not timm_model:
        if resnet50:

            if SAM_check:

                if args.model_name == 'resnet34' or 'resnet18':
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
                network, feature_dim = load_network(backbone_name)
                model = SAM(network=network, backbone=backbone_name, projector_dim=projector_dim,
                                class_num=num_classes, pretrained=True, pretrained_path=pretrained_path)#.to(args.device)
                classifier = Classifier(proj_size, num_classes)#.to(args.device)   #2048/num of bilinear 2048*16
                
                # mb initialise classifier ?
                # classifier.classifier_layer.apply(init_weights)

                print("[INFO] A pre-trained ECCV ResNet-50 model is used")


            else:
                #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True) 

                #model = models.resnet18(pretrained=True) #, num_classes=200)
                #model = models.resnet34(pretrained=True) #, num_classes=200)
                model = models.resnet50(pretrained=True) #, num_classes=200)
                ##model = models.resnet50(pretrained=True, zero_init_residual=True) #, num_classes=200)
                #model = models.resnet101(pretrained=True) #, num_classes=200)
                #model = models.resnet152(pretrained=True) #, num_classes=200)

                model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
                
                #model.fc.apply(init_weights) #?
                model.fc.weight.data.normal_(0, 0.01)
                model.fc.bias.data.fill_(0.0)

                print("[INFO] A pre-trained ResNet-50 model is used")

        else:
            model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, vis=True, smoothing_value=args.smoothing_value, dataset=args.dataset)
            
            if args.pretrained_dir != "":
                print("[INFO] A pre-trained model is used")
                model.load_from(np.load(args.pretrained_dir))
            else:
                print("[INFO] A model will be trained from scratch")

    '''
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
    
    '''


    if SAM_check:
        model.to(args.device)
        classifier.to(args.device)

        print(model)
        print(classifier)
    
        print("backbone params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6 / 2))
        print("classifier params: {:.2f}M".format(sum(p.numel() for p in classifier.parameters()) / 1e6))
    
    else:
        model.to(args.device)
        num_params = count_parameters(model)

        print(model)

        save_model(args, model)

        #print(model)

        if args.model_type == "ViT": logger.info("{}".format(config))
        logger.info("Training parameters %s", args)
        logger.info("Total Parameter: \t%2.1fM" % num_params)
        print(num_params)


    if SAM_check:
        return args, model, classifier, num_classes
    else:
        return args, model, num_classes


