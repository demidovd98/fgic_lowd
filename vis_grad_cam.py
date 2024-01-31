from pytorch_grad_cam import GradCAM
from PIL import Image
from torchvision.models import resnet50
import torch
import torchvision.transforms as transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
import numpy as np
import cv2
import torch.nn as nn



imgs = ["191.Red_headed_Woodpecker/Red_Headed_Woodpecker_0078_183427.jpg",
"191.Red_headed_Woodpecker/Red_Headed_Woodpecker_0098_182732.jpg",
"191.Red_headed_Woodpecker/Red_Headed_Woodpecker_0101_182538.jpg",
"192.Downy_Woodpecker/Downy_Woodpecker_0038_184418.jpg",
"192.Downy_Woodpecker/Downy_Woodpecker_0080_184240.jpg",
"192.Downy_Woodpecker/Downy_Woodpecker_0096_184532.jpg",
"193.Bewick_Wren/Bewick_Wren_0042_184878.jpg",
"193.Bewick_Wren/Bewick_Wren_0081_185080.jpg",
"193.Bewick_Wren/Bewick_Wren_0083_185190.jpg",
"194.Cactus_Wren/Cactus_Wren_0016_185582.jpg",
"194.Cactus_Wren/Cactus_Wren_0025_185696.jpg",
"194.Cactus_Wren/Cactus_Wren_0044_185492.jpg",
"195.Carolina_Wren/Carolina_Wren_0058_186409.jpg",
"195.Carolina_Wren/Carolina_Wren_0086_186431.jpg",
"195.Carolina_Wren/Carolina_Wren_0099_186237.jpg",
"196.House_Wren/House_Wren_0035_187708.jpg",
"196.House_Wren/House_Wren_0045_187374.jpg",
"196.House_Wren/House_Wren_0072_187899.jpg",
"197.Marsh_Wren/Marsh_Wren_0062_188158.jpg",
"197.Marsh_Wren/Marsh_Wren_0095_188371.jpg",
"197.Marsh_Wren/Marsh_Wren_0097_188214.jpg",
"198.Rock_Wren/Rock_Wren_0004_189046.jpg",
"198.Rock_Wren/Rock_Wren_0026_189181.jpg",
"198.Rock_Wren/Rock_Wren_0059_188941.jpg",
"199.Winter_Wren/Winter_Wren_0033_189635.jpg",
"199.Winter_Wren/Winter_Wren_0037_190123.jpg",
"199.Winter_Wren/Winter_Wren_0081_190049.jpg",
"200.Common_Yellowthroat/Common_Yellowthroat_0037_190698.jpg",
"200.Common_Yellowthroat/Common_Yellowthroat_0049_190708.jpg",
"200.Common_Yellowthroat/Common_Yellowthroat_0075_190900.jpg"]


transform = transforms.Compose([transforms.PILToTensor()])

model = resnet50(pretrained=False)
model.fc = nn.Linear(2048, 200)

vanilla=False

if vanilla:
    model.load_state_dict(torch.load("CUB10_cnn_resnet50_vanillaTrue_lrRat1_0_steps40000_checkpoint.bin"))
else:
    model.load_state_dict(torch.load("CUB10_cnn_resnet50_vanillaFalse_lrRatNone_augTypedouble_crop_augCropNone.bin"))

model.eval()
target_layers = [model.layer4]

with open("/l/users/cv-805/Datasets/CUB_200_2011/CUB_200_2011/CUB_200_2011/low_data/test.txt", "r") as rfile:
    imgs = rfile.read().split("\n")

print(imgs)
imgs = imgs[:-1]
print(imgs)

image_path = "/l/users/cv-805/Datasets/CUB_200_2011/CUB_200_2011/CUB_200_2011/images/"
for img_name in imgs[:-100]:
    img_data = img_name.split(" ")
    img_name = img_data[0]
    cls_id = int(img_name.split(".")[0]) - 1
    img_f = img_name.split("/")[-1]
    input_tensor = Image.open(f"{image_path}/{img_name}")
    img = cv2.resize(np.array(input_tensor), (224, 224))
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    cam = GradCAM(model=model, target_layers=target_layers)


    targets = [ClassifierOutputTarget(cls_id)]

    print(f'{input_tensor.shape}')
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)


    cam = np.uint8(255*grayscale_cam[0, :])
    cam = cv2.merge([cam, cam, cam])
    images = np.hstack((np.uint8(255*img), cam , visualization))
    print(images.shape)

    final = Image.fromarray(images)

    final.save(f"gradcam/vanilla/test/gradcam_{img_f}.png")



