import logging

import torch

from torchvision import transforms, datasets
from .dataset import *
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from PIL import Image
from .autoaugment import AutoAugImageNetPolicy
import os

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == 'INat2017':
        train_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.RandomCrop((304, 304)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.CenterCrop((304, 304)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)

    elif args.dataset=="cotton" or args.dataset=="soyloc":
        train_transform=transforms.Compose([transforms.Resize((args.resize_size, args.resize_size),Image.BILINEAR),
                                    transforms.RandomCrop((args.img_size, args.img_size)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                           ])
        test_transform=transforms.Compose([
            transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                                    transforms.CenterCrop((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = eval(args.dataset)(root=args.data_root, is_train=True, transform=train_transform)
        testset = eval(args.dataset)(root=args.data_root, is_train=False, transform = test_transform)
    
    
    elif args.dataset == 'dogs':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = dogs(root=args.data_root,
                                train=True,
                                cropped=False,
                                transform=train_transform,
                                download=True
                                )
        testset = dogs(root=args.data_root,
                                train=False,
                                cropped=False,
                                transform=test_transform,
                                download=False
                                )
    
    
    elif args.dataset== "CUB":
        train_transform=transforms.Compose([
                                    transforms.Resize((args.resize_size, args.resize_size),Image.BILINEAR), # if no saliency
                                    #transforms.Resize((560, 560), Image.BILINEAR), #transFG 600 
                                    transforms.RandomCrop((args.img_size, args.img_size)), # if no saliency


                                    # transforms.RandomChoice(
                                    #     [
                                    #         #transforms.RandomCrop((128, 128)),
                                    #         transforms.RandomCrop((160, 160)),
                                    #         transforms.RandomCrop((192, 192)),
                                    #         transforms.RandomCrop((224, 224)),
                                    #     ]),
                                    # transforms.Resize((args.img_size, args.img_size),Image.BILINEAR), # my for bbox (# if saliency)


                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add (FFVT)
                                    #AutoAugImageNetPolicy(),
                                    
                                    transforms.RandomHorizontalFlip(), # !!! FLIPPING in dataset.py !!!

                                    transforms.ToTensor(),
                                    #transforms.Normalize([0.8416, 0.867, 0.8233], [0.2852, 0.246, 0.3262])])
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])

        test_transform=transforms.Compose([
                                    transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR), # if no saliency
                                    #transforms.Resize((560, 560), Image.BILINEAR),  #transFG 600
                                    transforms.CenterCrop((args.img_size, args.img_size)), # if no saliency

                                    #transforms.Resize((args.img_size, args.img_size),Image.BILINEAR), # my for bbox (# if saliency)
                                    
                                    # transforms.Resize(( int(args.img_size // 0.84) , int(args.img_size // 0.84) ),Image.BILINEAR), # fro img crop
                                    # transforms.CenterCrop((args.img_size, args.img_size)),


                                    transforms.ToTensor(),
                                    #transforms.Normalize([0.8416, 0.867, 0.8233], [0.2852, 0.246, 0.3262])])
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

        '''
        train_transform=transforms.Compose([
            
                                    #transforms.Resize((args.resize_size, args.resize_size),Image.BILINEAR),
                                    transforms.Resize((args.img_size, args.img_size),Image.BILINEAR),

                                    #transforms.RandomCrop((args.img_size, args.img_size)),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add
                                    transforms.RandomHorizontalFlip(),
                                    
                                    transforms.ToTensor(),
                                    #transforms.Normalize([0.8416, 0.867, 0.8233], [0.2852, 0.246, 0.3262])])
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                           ])
        test_transform=transforms.Compose([
            transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                                    transforms.CenterCrop((args.img_size, args.img_size)),
            #transforms.Resize((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    #transforms.Normalize([0.8416, 0.867, 0.8233], [0.2852, 0.246, 0.3262])])
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        '''

        trainset = eval(args.dataset)(root=args.data_root, is_train=True, transform=train_transform, split=args.split, vanilla=args.vanilla)
        testset = eval(args.dataset)(root=args.data_root, is_train=False, transform=test_transform, split=args.split, vanilla=args.vanilla)


    elif args.dataset == 'cars':
        trainset = CarsDataset(
                            root=args.data_root, #my
                            is_train=True, #my
                            transform=transforms.Compose([
                                    transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                                    transforms.RandomCrop((args.img_size, args.img_size)),
                                    AutoAugImageNetPolicy(),
                                    transforms.RandomHorizontalFlip(),  # !!! FLIPPING in dataset.py !!!
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]),
                            split=args.split
                            )

        testset = CarsDataset(
                            args.data_root, #my
                            is_train=False,
                            transform=transforms.Compose([
                                    transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                                    transforms.CenterCrop((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]),
                            split=args.split
                            )


    elif args.dataset== "air":
        train_transform=transforms.Compose([
                                    # transforms.Resize((args.resize_size, args.resize_size),Image.BILINEAR),
                                    # transforms.RandomCrop((args.img_size, args.img_size)),
                                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add
                                    # transforms.RandomHorizontalFlip(),
                                    # #transforms.RandomVerticalFlip(),
                                    # transforms.ToTensor(),
                                    # #transforms.Normalize([0.8416, 0.867, 0.8233], [0.2852, 0.246, 0.3262])])
                                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


                                    #transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
                                    transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                                    transforms.RandomCrop((args.img_size, args.img_size)),

                                    #transforms.Resize((450, 450), Image.BILINEAR),
                                    # transforms.RandomChoice(
                                    #     [
                                    #         transforms.RandomCrop((300, 300), Image.BILINEAR),
                                    #         transforms.RandomCrop((400, 400), Image.BILINEAR),
                                    #     ]),
                                    #transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),

                                    #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add
                                    AutoAugImageNetPolicy(),

                                    transforms.RandomHorizontalFlip(), # !!! FLIPPING in dataset.py !!!

                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

        test_transform=transforms.Compose([
                                    # transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                                    # transforms.CenterCrop((args.img_size, args.img_size)),
                                    # #transforms.Resize((args.img_size, args.img_size)),
                                    # transforms.ToTensor(),
                                    # #transforms.Normalize([0.8416, 0.867, 0.8233], [0.2852, 0.246, 0.3262])])
                                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    
                                    
                                    #transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
                                    transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                                    transforms.CenterCrop((args.img_size, args.img_size)),

                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

        trainset = FGVC_aircraft(root=args.data_root, is_train=True, transform=train_transform, split=args.split)
        testset = FGVC_aircraft(root=args.data_root, is_train=False, transform=test_transform, split=args.split)


    elif args.dataset== "CRC":
        train_transform=transforms.Compose([
                                    transforms.Resize((args.resize_size, args.resize_size),Image.BILINEAR), # if no saliency
                                    #transforms.Resize((560, 560), Image.BILINEAR), #transFG 600 
                                    transforms.RandomCrop((args.img_size, args.img_size)), # if no saliency
                                    
                                    #transforms.Resize((args.img_size, args.img_size),Image.BILINEAR), # my for bbox (# if saliency)

                                    transforms.RandomHorizontalFlip(), # !!! FLIPPING in dataset.py !!!
                                    transforms.RandomVerticalFlip(), # from air
                                    
                                    #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add (FFVT)
                                    #AutoAugImageNetPolicy(), # from cars


                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    #transforms.Normalize([0.650, 0.472, 0.584], [0.158, 0.164, 0.143]), # for CRC (our manual)
                                    ])

        test_transform=transforms.Compose([
                                    transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR), # if no saliency
                                    #transforms.Resize((560, 560), Image.BILINEAR),  #transFG 600
                                    transforms.CenterCrop((args.img_size, args.img_size)), # if no saliency

                                    #transforms.Resize((args.img_size, args.img_size),Image.BILINEAR), # my for bbox (# if saliency)
                                    
                                    # transforms.Resize(( int(args.img_size // 0.84) , int(args.img_size // 0.84) ),Image.BILINEAR), # fro img crop
                                    # transforms.CenterCrop((args.img_size, args.img_size)),


                                    transforms.ToTensor(),
                                    #transforms.Normalize([0.8416, 0.867, 0.8233], [0.2852, 0.246, 0.3262])])
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

        trainset = eval(args.dataset)(root=args.data_root, is_train=True, transform=train_transform, vanilla=args.vanilla, split=args.split)
        testset = eval(args.dataset)(root=args.data_root, is_train=False, transform=test_transform, vanilla=args.vanilla, split=args.split)



    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              
                              #num_workers=4,
                              num_workers=8, #20,

                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,

                             #num_workers=4,
                             num_workers=8, ##20,

                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
