import torch
import argparse
import torch.nn.functional as F
from models.models_adapted import resnet_my
from torchvision import transforms 
from utils.dataset import *


parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--name", #required=True,
		    default=None,
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
						"resnet101", "resnet152",
						"vgg16",
						"googlenet",
						"squeezenet1_1",
						"inception_v3",
						"densenet169",
						"wide_resnet50_2",
						],
		    default="resnet50",
		    help="Which specific model to use.")

parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
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

parser.add_argument("--dataload_workers", type=int, default=8,
		    help="Number of workers for data pre-processing")
    
parser.add_argument('--vanilla', action='store_true',
		    help="Whether to use the vanilla model")
parser.add_argument("--split", # required=True,
		    choices=["1i", "1p", "2", "3", "4", "5", "10", "15", "30", "50", "100"],
		    default=None,
		    help="Name of the split")

parser.add_argument('--sam', action='store_true',
		    help="Whether to use the SAM training setup")
parser.add_argument('--timm_model', action='store_true',
		    help="Whether to use pre-trained model from the timm library")        
parser.add_argument('--saliency', action='store_true',
		    help="Whether to use saliency information (foreground/nackground mask)")
parser.add_argument('--preprocess_full_ram', action='store_true',
		    help="Whether to preprocess full dataset and upload it to RAM before training")

parser.add_argument('--auto_scheduler', action='store_true',
		    help="Whether to use the SAM training setup")
parser.add_argument("--lr_ratio", default=None, type=float, # required=True,
		    help="Learning rate ratio for the last classification layer.")
parser.add_argument("--dist_coef", default=0.1, type=float, # required=True,
		    help="Coefficient of the distillation loss contribution.")

parser.add_argument("--aug_type", choices=["single_crop", "double_crop"],
		    default=None,
		    help="Which architecture to use.")
parser.add_argument("--aug_vanilla", choices=["aug_basic", "aug_scalemix", "aug_asymmAugs", "aug_multicrop"],
		    default=None,
		    help="Whether to use an extra augmentations on vanilla.")
parser.add_argument("--aug_crop", choices=["aug_basic", "aug_scalemix", "aug_asymmAugs", "aug_multicrop"],
		    default=None,
		    help="Whether to use an extra augmentations on our crop branch.")
parser.add_argument('--data_root', type=str, default='/l/users/cv-805/Datasets') # shared
parser.add_argument('--smoothing_value', type=float, default=0.0,
		    help="Label smoothing value\n")
parser.add_argument("--montecarlo_dropout", type=float, default=None,
		help="Traing Bayesian Montecarlo Dropout model") 
args = parser.parse_args()

def get_dataset(arg):
    test_transform=transforms.Compose([
	    transforms.Resize((arg.resize_size, arg.resize_size), Image.BILINEAR), # if no saliency
	    transforms.CenterCrop((arg.img_size, arg.img_size)), # if no saliency
	    transforms.ToTensor(),
	    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    ])
    test_dataset = CUB(args=arg, is_train=False, transform=test_transform)
    return test_dataset 

if __name__ == "__main__":

	## change to argument
	path2checkpoint = "output/CUB10_cnn_resnet50_vanillaFalse_lrRatNone_augTypedouble_crop_augCropNone_distCoef0.1_steps40000_checkpoint.bin"
	num_classes = 200
	args.data_root = args.data_root + "/CUB_200_2011/CUB_200_2011/CUB_200_2011"

	model = resnet_my.resnet50(montecarlo_dropout=args.montecarlo_dropout) 
	model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)

	checkpoint = torch.load(path2checkpoint)
	model.load_state_dict(checkpoint, strict=True)
	dataset = get_dataset(args)

	img, label = next(iter(dataset))
	img = img[None, ...]

	model.eval()
	preds = model.MCDInference(img)
    
	print([torch.argmax(i, dim=1).item() for i in preds], "label: " , label)
