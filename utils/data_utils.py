import logging

import torch

from torchvision import transforms, datasets
from .dataset import *
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from PIL import Image
from .autoaugment import AutoAugImageNetPolicy
import os


from timm.data.auto_augment import rand_augment_transform
from _extra.asym_siam.moco.loader import CropsTransform, GaussianBlur



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


    if args.aug_vanilla is not None:
        ## From asym_siam:
        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        if args.aug_vanilla == "aug_multicrop":
            #ratio_range=(0.14,1.0)
            ratio_range=(0.3,0.8)
        else:
            #ratio_range=(0.2,1.0)
            if args.aug_vanilla == "aug_asymmAugs":
                ratio_range=(0.5,0.8)
            else:
                ratio_range=(0.5,1.0)

        augmentation = [
            transforms.RandomResizedCrop(224, scale=ratio_range),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],  # not strengthened
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]


        """
        # --------------------------------------------------------------------------- #
        #                           Asymmetric Augmentations                          #
        # --------------------------------------------------------------------------- #
        asymmetric augmentation recipes are formed by stronger and weaker augmentation
        in source and target. Stronger augmentation introduces a higher variance, that
        hurts target but helps source, and vice versa for weaker augmentation.
        # --------------------------------------------------------------------------- #
        """
        if args.aug_vanilla == "aug_asymmAugs":
            augmentation_stronger = [
                transforms.RandomResizedCrop(224, scale=ratio_range),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],  # not strengthened
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                rand_augment_transform(
                    "rand-m10-n2-mstd0.5", {"translate_const": 100},
                ),
                transforms.ToTensor(),
                normalize,
            ]
            augmentation_weaker = [
                transforms.RandomResizedCrop(224),
                #transforms.RandomResizedCrop(224, scale=(0.5,0.8)), # (0.2, 1.0) try

                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        
        """
        # --------------------------------------------------------------------------- #
        #                                  MultiCrop                                  #
        # --------------------------------------------------------------------------- #
        Besides the two basic views needed for Siamese learning, MultiCrop takes
        additional views from each image per iteration. To alleviate the added
        computation cost, a common strategy is to have low-resolution crops
        (e.g., 96×96) instead of standard-resolution crops (224×224) as added views.
        As a side effect, inputting small crops can potentially increase the variance
        for an encoder due to the size and crop-distribution changes.
        # --------------------------------------------------------------------------- #
        """
        augmentation_mini = [
            #transforms.RandomResizedCrop(96, scale=(0.05, 0.14)),
            transforms.RandomResizedCrop(224, scale=ratio_range),

            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],  # not strengthened
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

        """
        # --------------------------------------------------------------------------- #
        #                                  ScaleMix                                   #
        # --------------------------------------------------------------------------- #
        ScaleMix generates new views of an image by mixing two views of potentially
        different scales together via binary masking. The masking strategy follows
        CutMix. where an entire region - denoted by a box with randomly sampled
        coordinates - is cropped and pasted. Unlike CutMix, ScaleMix only operates on
        views from the same image, and the output is a single view of standard size
        (224x224). This single view can be regarded as an efficient approximation of
        MultiCrop, without the need to process small crops separately.
        # --------------------------------------------------------------------------- #
        """
        transform_vanillaAugs = CropsTransform(
                #args,
                key_transform=transforms.Compose(augmentation_weaker)
                if (args.aug_vanilla == "aug_asymmAugs")
                else transforms.Compose(augmentation),
                query_mini_transform=transforms.Compose(augmentation_mini),
                query_transform=transforms.Compose(augmentation_stronger)
                if (args.aug_vanilla == "aug_asymmAugs")
                else transforms.Compose(augmentation),
                enable_scalemix=(args.aug_vanilla == "aug_scalemix"), #args.aug_scalemix,
                enable_multicrop=(args.aug_vanilla == "aug_multicrop"), # args.aug_multicrop,
                enable_asymm=(args.aug_vanilla == "aug_asymmAugs"), # args.aug_asymmAugs,
                enable_mean_encoding=False, #args.enable_mean_encoding,
            )
        ##






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
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
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


        if args.aug_vanilla is not None: #args.aug_basic or args.aug_scalemix or args.aug_multicrop or args.aug_asymmAugs:
            train_transform = transform_vanillaAugs

        else:
            train_transform=transforms.Compose([
                    transforms.Resize((args.resize_size, args.resize_size),Image.BILINEAR), # if no saliency
                    #transforms.Resize((560, 560), Image.BILINEAR), #transFG 600 

                    transforms.RandomCrop((args.img_size, args.img_size)), # if no saliency


                    # transforms.RandomChoice(
                    #     [
                    #         transforms.Compose([
                    #             transforms.RandomCrop((args.img_size, args.img_size)),
                    #         ]),

                    #         transforms.Compose([
                    #             transforms.RandomChoice(
                    #             [
                    #                 #transforms.RandomCrop((112, 112)),
                    #                 transforms.RandomCrop((128, 128)),
                    #                 transforms.RandomCrop((160, 160)),
                    #                 #transforms.RandomCrop((180, 180)),
                    #                 transforms.RandomCrop((192, 192)),
                    #                 #transforms.RandomCrop((200, 200)),
                    #             ]),
                    #             transforms.Resize((args.img_size, args.img_size),Image.BILINEAR), # my for bbox (# if saliency)
                    #         ]),
                    #     ]),


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

        trainset = eval(args.dataset)(args=args, is_train=True, transform=train_transform,
                                      #root=args.data_root, split=args.split, vanilla=args.vanilla, aug_type=args.aug_type, preprocess_full_ram=args.preprocess_full_ram
                                      )
        testset = eval(args.dataset)(args=args, is_train=False, transform=test_transform,
                                     #root=args.data_root, split=args.split, vanilla=args.vanilla, aug_type=args.aug_type, preprocess_full_ram=args.preprocess_full_ram
                                      )


    elif args.dataset == 'cars':
        trainset = CarsDataset(
                # os.path.join(args.data_root,'devkit/cars_train_annos.mat'),
                # os.path.join(args.data_root,'cars_train'),
                # os.path.join(args.data_root,'devkit/cars_meta.mat'),
                # cleaned=os.path.join(data_dir,'cleaned.dat'),
                
                args=args, # my
                #root=args.data_root, #my
                is_train=True, #my

                transform=transforms.Compose([
                        # transforms.Resize((600, 600), Image.BILINEAR),
                        # transforms.RandomCrop((448, 448)),
                        # transforms.RandomHorizontalFlip(),
                        # AutoAugImageNetPolicy(),
                        # transforms.ToTensor(),
                        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        
                        
                        #transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
                        transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                        
                        transforms.RandomCrop((args.img_size, args.img_size)),



                        # transforms.RandomChoice(
                        #     [
                        #         transforms.Compose([
                        #             transforms.RandomCrop((args.img_size, args.img_size)),
                        #         ]),

                        #         transforms.Compose([
                        #             transforms.RandomChoice(
                        #             [
                        #                 #transforms.RandomCrop((112, 112)),
                        #                 transforms.RandomCrop((128, 128)),
                        #                 transforms.RandomCrop((160, 160)),
                        #                 #transforms.RandomCrop((180, 180)),
                        #                 transforms.RandomCrop((192, 192)),
                        #                 #transforms.RandomCrop((200, 200)),
                        #             ]),
                        #             transforms.Resize((args.img_size, args.img_size),Image.BILINEAR), # my for bbox (# if saliency)
                        #         ]),
                        #     ]),



                        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add (from FFVT) mb instead of imagenet?
                        AutoAugImageNetPolicy(),

                        transforms.RandomHorizontalFlip(),  # !!! FLIPPING in dataset.py !!!

                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]),
                )

        testset = CarsDataset(
                # os.path.join(args.data_root,'cars_test_annos_withlabels.mat'),
                # os.path.join(args.data_root,'cars_test'),
                # os.path.join(args.data_root,'devkit/cars_meta.mat'),
                # cleaned=os.path.join(data_dir,'cleaned_test.dat'),

                args=args, # my
                #root = args.data_root, #my
                is_train=False, # my

                transform=transforms.Compose([
                        # transforms.Resize((600, 600), Image.BILINEAR),
                        # transforms.CenterCrop((448, 448)),
                        # transforms.ToTensor(),
                        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        
                        
                        #transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),
                        transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                        transforms.CenterCrop((args.img_size, args.img_size)),

                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]),
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

        trainset = FGVC_aircraft(args=args, is_train=True, transform=train_transform)
                                #root=args.data_root, is_train=True, transform=train_transform, split=args.split)
        testset = FGVC_aircraft(args=args, is_train=False, transform=train_transform)
                                #root=args.data_root, is_train=False, transform=test_transform, split=args.split)


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
                              num_workers=args.dataload_workers, #4, 12, 20

                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,

                             #num_workers=4,
                             num_workers=args.dataload_workers, #4, 12, 20,

                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
