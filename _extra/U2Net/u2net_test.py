import os
from re import X
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

#from data_loader import RescaleT
from .data_loader import RescaleT

#from data_loader import ToTensor
from .data_loader import ToTensor

#from data_loader import ToTensorLab
from .data_loader import ToTensorLab

#from data_loader import SalObjDataset
from .data_loader import SalObjDataset

#from model import U2NET # full size version 173.6 MB
from .model import U2NET # full size version 173.6 MB

#from model import U2NETP # small version u2net 4.7 MB
from .model import U2NETP # small version u2net 4.7 MB


# My:
#import cv2
import copy
from torchvision.utils import save_image
#


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')


#def mask_hw():
def mask_hw(full_ds=True, img_path=None, shape_hw=None):

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp


    if img_path is None:
        #image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
        #image_dir = os.path.join(os.getcwd(), 'images')
        image_dir = os.path.join(os.getcwd(), 'U2Net/images')

        img_name_list = glob.glob(image_dir + os.sep + '*')
        print("local image")
        print(img_name_list)


    else:
        if full_ds:
            img_name_list = img_path
            shape_hw_list = shape_hw
            #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            #print(img_name_list)
            #print(len(img_name_list))
            #print(len(shape_hw_list))

        else:
            img_name_list = glob.glob(img_path)
            #print("global image")
            print(img_path)


    '''
    #prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    #prediction_dir = os.path.join(os.getcwd(), 'output', model_name + '_results' + os.sep)
    prediction_dir = os.path.join(os.getcwd(), 'U2Net/output/test', model_name + '_results' + os.sep)
    '''
    
    #model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
    model_dir = os.path.join(os.getcwd(), 'U2Net/saved_models', model_name, model_name + '.pth')


    #print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,

                                        #num_workers=1
                                        num_workers=1 # 20                                        
                                        
                                        )

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        #print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        #print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------

    mask_out_np_list = []
    start_x_list = []
    start_y_list = []
    h_list = []
    w_list = []

    bad_mask_count = 0
    refined_mask_count = 0


    print_info = False

    for i_test, data_test in enumerate(test_salobj_dataloader):

        #print("U2N:", i_test, img_name_list[i_test].split(os.sep)[-1])
        if print_info: print("U2N:", i_test, img_name_list[i_test])

        #print(shape_hw_list[i_test])


        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        # if inputs_test.size(1) != 3:
        #     print("gray_image2________________________________________________")
        #     print(inputs_test.size())
        # #print(type(inputs_test))

        if full_ds:
            shape_hw_i = shape_hw_list[i_test]
        #else:
            #shape_hw = [333, 500]


        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)



        with torch.no_grad():
            d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)



        #print(pred.size())
        #print(type(pred))

        #with torch.set_printoptions(profile="full"):
        #print(pred)



        THRESHOLD = 0.8 # 0.5 # 0.8 # 0.5 #0.8 # for an original mask orig 0.9, try 0.7 - 0.8 ? (better not smaller than 0.7 cuz artifacts)
        THRESHOLD_resize = 0.2 # 0.1 # 0.2 # for a resized mask
        THRESHOLD_deep = 0.1 # for a non-detected mask

        #RESCALE = 255.0
        #LAYER = 2 # green?


        pred = pred[0, :, :]

        
        pred_cpu = pred.cpu()
        out_img = pred_cpu.detach().numpy()
        
        out_img_refine = copy.deepcopy(out_img)

        '''
        # pred_np = pred_cpu.detach().numpy()
        # print(type(pred_np))
        # print(pred_np.shape)
        # print(pred_np)

        # out_img = np.asarray(pred_np)
        #print(type(out_img))
        #print(out_img.shape)

        #with torch.set_printoptions(profile="full"):
        #print(out_img)


        # out_img /= RESCALE # already between [0, 1]
        '''
        
    
        # BACKGROUND REMOVAL

        
        out_img[out_img > THRESHOLD] = 1
        out_img[out_img <= THRESHOLD] = 0
        
        '''
        # for i in range(pred.size(0)):
        #     pred_temp = pred[i, :, :]

        # pred = torch.where( pred > THRESHOLD, torch.tensor(1.).cuda(), 
        #                                         torch.tensor(0.).cuda())
        # pred =  (pred * RESCALE)
        ######### CEHCK THAT IT ACTUALY WORKS !!!!!!!!!!!!!

        # for i in range(pred.size(0)):
        #     pred_temp = pred[i, :, :]
        # pred = torch.where( pred >= THRESHOLD, torch.tensor(RESCALE).cuda(), 
        #                                             torch.tensor(0.).cuda())



        #with np.printoptions(threshold=np.inf):
        #print(out_img.shape)
        #print(out_img)

        #out_img = np.transpose(out_img, (1, 2, 0))
        #print(out_img.shape)
        '''

        '''
        pred_cpu = pred.cpu()
        out_img = pred_cpu.detach().numpy()
        '''
        '''
        out_img = out_img[:, :, 0]
        '''
        
        out_img = (out_img * 255).astype(np.uint8)
        # if self.count < 5: print("mask 255", mask)
        out_img = Image.fromarray(out_img, mode='L')
        # if self.count < 5: print("mask tensor before", mask)
        
        
        '''
        # import time
        # from torchvision.utils import save_image
        # mask_tem = transforms.ToTensor()(mask)
        # save_image( mask_tem, "mask_before.png" )


        transform_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(), #(mode='1'),
            transforms.Resize((25,25), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
            transforms.ToTensor()])

        mask = transform_mask(out_img)

        from torchvision.utils import save_image
        #print(img.shape)
        #mask_img = transforms.ToPILImage()(mask)
        print("next mask")
        save_image( mask, "mask_after.png" )
        #time.sleep(4)
        '''

        '''
        # from torchvision.utils import save_image
        # #print(img.shape)
        # out_img_temp = transforms.ToTensor()(out_img)
        # print("next mask")
        # save_image( out_img_temp, "mask_after.png" )




        # out_img = out_img[:, :, 0]
        # out_img = np.stack([out_img] * 3, 2)
        # print(out_img.shape)

        # import matplotlib.pyplot as plt
        # plt.imshow(out_img)
        # plt.show()




        #mask = int(mask)
        #mask = np.stack([mask] * 3, 2)
        #mask = Image.fromarray(mask, mode='RGB')

        # mask = (mask * 255).astype(np.uint8)
        # # if self.count < 5: print("mask 255", mask)
        # mask = Image.fromarray(mask, mode='L')
        # # if self.count < 5: print("mask tensor before", mask)
        
        # # import time
        # from torchvision.utils import save_image
        # mask_tem = transforms.ToTensor()(mask)
        # save_image( mask_tem, "mask_before.png" )




        # print(out_img.shape)

        # from torchvision.utils import save_image
        # print(out_img.shape)
        # print("next img")
        # out_img = transforms.ToTensor()(out_img)
        # print(out_img.size())

        # save_image( out_img, "img1.png" )
        '''
        
        # BOUNDING BOX CREATION

        #out_layer = out_img[:,:,LAYER]

        # shape_hw = [333, 500]


        transform_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(mode='L'), #(mode='1'),
            #transforms.Resize((image.shape[1],image.shape[0]), Image.BILINEAR),
            transforms.Resize((shape_hw_i[0], shape_hw_i[1]), Image.BILINEAR), # shape_hw (0 - height, 1 - width)
            transforms.ToTensor(),
            ])

        #out_img = pred.cpu()
        out_img = transform_mask(out_img)
        #print(type(out_img))
        #print(out_img.shape)

        #save_image( mask, "img_bb.png" )
        #time.sleep(4)

        out_img = out_img[0, :, :]



        #mask_out = out_img.detach().numpy()
        mask_out = out_img
        mask_out = mask_out.cpu()

        # torch.set_printoptions(profile="full")
        # print(mask_out)        
        # print("------------------------------------------------")

        mask_out = torch.where( mask_out > THRESHOLD_resize, torch.tensor(0.), 
                                                torch.tensor(1.))

        # torch.set_printoptions(profile="full")
        # print(mask_out)
        
        mask_out_np = mask_out.detach().numpy()
        
        '''
        # with np.printoptions(threshold=np.inf):
        #     print(mask_out_np)

        # mask_out_np[mask_out_np > THRESHOLD] = 0
        # mask_out_np[mask_out_np <= THRESHOLD] = 1
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        # with np.printoptions(threshold=np.inf):
        #     print(mask_out_np)


        # from torchvision.utils import save_image
        # #print(out_img.shape)
        # print("next img")
        # out_img = transforms.ToTensor()(out_img)
        # print(out_img.size())
        # save_image( out_img, "img1.png" )
        '''



        out_layer = out_img
        out_layer = out_layer.cpu()

        #out_layer =  out_img.detach().numpy() # old
        out_layer = torch.where( out_layer > THRESHOLD_resize, torch.tensor(1.), 
                                                torch.tensor(0.))
        out_layer = out_layer.detach().numpy()

        #out_layer =  out_img

        #mask = transforms.ToTensor()(mask)

        #print(out_layer.size())
        #print(out_layer)

        x_starts = [np.where(out_layer[i]==1)[0][0] if len(np.where(out_layer[i]==1)[0])!=0 else out_layer.shape[0]+1 for i in range(out_layer.shape[0])]
        x_ends = [np.where(out_layer[i]==1)[0][-1] if len(np.where(out_layer[i]==1)[0])!=0 else 0 for i in range(out_layer.shape[0])]
        y_starts = [np.where(out_layer.T[i]==1)[0][0] if len(np.where(out_layer.T[i]==1)[0])!=0 else out_layer.T.shape[0]+1 for i in range(out_layer.T.shape[0])]
        y_ends = [np.where(out_layer.T[i]==1)[0][-1] if len(np.where(out_layer.T[i]==1)[0])!=0 else 0 for i in range(out_layer.T.shape[0])]
        
        startx = min(x_starts)
        endx = max(x_ends)
        starty = min(y_starts)
        endy = max(y_ends)
        start = (startx,starty)
        end = (endx,endy)


        ## for cases when U2N couldn't detect mask:
        # [DONE] TODO 1.1 if (end - start) < 30-50px -> decrease the THRESHOLD
        # [DONE] TODO 1.2 if (start>end) or end==0 ? -> decrease the THRESHOLD
        # [DONE] TODO 2.1 if no mask found anyway -> create center crop mask (x, y) +-10 %
        # [DONE] TODO 2.2 + restore h,w from (0,0) to (x, y) +-10 %
        w_temp = end[0] - start[0] 
        h_temp = end[1] - start[1]

        mask_px = np.count_nonzero(out_layer > 0.9) # (expected to be == 1.0)
        if print_info: print("Mask px old:", mask_px)

        #if (w_temp < 50) or (h_temp < 50) or (mask_px < 5000) : # can be combined
        if (end[0] <= start[0]) or (end[1] <= start[1]) or (mask_px < 5000) or (w_temp < 50) or (h_temp < 50) :
            # if (end[0] <= start[0]) or \
            #     (end[1] <= start[1]) or \
            #         (mask_px < 4000) or \
            #             (h_temp < 45) or \
            #             (w_temp < 45) :

            if print_info: print("[WARNING] Mask was not detected by U2N for image", img_name_list[i_test])
            if print_info: print("Trying to refine image and then detect mask again.")

            if print_info: print("Old x (start, end):", startx, endx)
            if print_info: print("Old y (start, end):", starty, endy)


            # img_dir = ("test/" + str(i_test))
            # if not os.path.exists(img_dir):
            #     os.makedirs(img_dir, exist_ok=True)   

            # img_name = ("test/" + str(i_test) + "/1mask_init" + str(i_test) + ".png")
            # img_temp = transforms.ToTensor()(out_img_refine)
            # save_image(img_temp, img_name)

            # img_name = ("test/" + str(i_test) + "/2mask_old" + str(i_test) + ".png")
            # img_temp = transforms.ToTensor()(mask_out_np)
            # save_image(img_temp, img_name)


            out_img_refine[out_img_refine > THRESHOLD_deep] = 1
            out_img_refine[out_img_refine <= THRESHOLD_deep] = 0


            out_img_refine = (out_img_refine * 255).astype(np.uint8)
            out_img_refine = Image.fromarray(out_img_refine, mode='L')

            transform_mask = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(mode='L'), #(mode='1'),
                #transforms.Resize((image.shape[1],image.shape[0]), Image.BILINEAR),
                transforms.Resize((shape_hw_i[0], shape_hw_i[1]), Image.BILINEAR), # shape_hw (0 - height, 1 - width)
                transforms.ToTensor(),
                ])

            out_img_refine = transform_mask(out_img_refine)

            out_img_refine = out_img_refine[0, :, :]

            out_layer_refine = out_img_refine
            out_layer_refine = out_layer_refine.cpu()
            out_layer_refine = torch.where( out_img_refine > THRESHOLD_resize, torch.tensor(1.), 
                                                    torch.tensor(0.))
            out_layer_refine = out_layer_refine.detach().numpy()


            x_starts = [np.where(out_layer_refine[i]==1)[0][0] if len(np.where(out_layer_refine[i]==1)[0])!=0 else out_layer_refine.shape[0]+1 for i in range(out_layer_refine.shape[0])]
            x_ends = [np.where(out_layer_refine[i]==1)[0][-1] if len(np.where(out_layer_refine[i]==1)[0])!=0 else 0 for i in range(out_layer_refine.shape[0])]
            y_starts = [np.where(out_layer_refine.T[i]==1)[0][0] if len(np.where(out_layer_refine.T[i]==1)[0])!=0 else out_layer_refine.T.shape[0]+1 for i in range(out_layer_refine.T.shape[0])]
            y_ends = [np.where(out_layer_refine.T[i]==1)[0][-1] if len(np.where(out_layer_refine.T[i]==1)[0])!=0 else 0 for i in range(out_layer_refine.T.shape[0])]
            
            startx = min(x_starts)
            endx = max(x_ends)
            starty = min(y_starts)
            endy = max(y_ends)
            start = (startx,starty)
            end = (endx,endy)

            if print_info: print("New x (start, end):", startx, endx)
            if print_info: print("New y (start, end):", starty, endy)

            w_temp = end[0] - start[0] 
            h_temp = end[1] - start[1]

            mask_px = np.count_nonzero(out_layer_refine > 0.9) # (expected to be == 1.0)
            if print_info: print("Mask px new:", mask_px)


            #if (w_temp < 50) or (h_temp < 50) or (mask_px < 5000) : # can be combined
            if (end[0] <= start[0]) or (end[1] <= start[1]) or (mask_px < 5000) or (w_temp < 50) or (h_temp < 50) :
                # if (end[0] <= start[0]) or \
                #     (end[1] <= start[1]) or \
                #         (mask_px < 4000) or \
                #             (h_temp < 45) or \
                #             (w_temp < 45) :

                if print_info: print("[WARNING] Mask was not deteted by U2N even after refining.")
                if print_info: print("Changing mask size (0, 0) to img size (", shape_hw_i[1], shape_hw_i[0], ") -10 p/c boundaries: ")

                if print_info: print("Old x (start, end):", startx, endx)
                startx = shape_hw_i[1] * 0.1
                endx = shape_hw_i[1] * 0.9 # w -> x
                if print_info: print("New x (start, end):", startx, endx)

                if print_info: print("Old y (start, end):", starty, endy)
                starty = shape_hw_i[0] * 0.1
                endy = shape_hw_i[0] * 0.9 # h -> y
                if print_info: print("New y (start, end):", starty, endy)

                start = (startx,starty)
                end = (endx,endy)

                mask_out_np = np.ones((int(shape_hw_i[0]), int(shape_hw_i[1])))
                mask_out_np[int(starty):int(endy), int(startx):int(endx)] = 0


                # img_name = ("test/" + str(i_test) + "/4mask_new2_" + str(i_test) + ".png")
                # img_temp = transforms.ToTensor()(mask_out_np)
                # save_image(img_temp, img_name)


                bad_mask_count+=1

            else:
                mask_out_refine = out_img_refine
                mask_out_refine = mask_out_refine.cpu()
                mask_out_refine = torch.where( mask_out_refine > THRESHOLD_resize, torch.tensor(0.), 
                                                        torch.tensor(1.))


                # img_name = ("test/" + str(i_test) + "/3mask_new1_" + str(i_test) + ".png")
                # #mask_tem = transforms.ToTensor()(mask)
                # save_image(mask_out_refine, img_name)


                mask_out_np = mask_out_refine.detach().numpy()

                refined_mask_count+=1


        w = end[0] - start[0]
        h = end[1] - start[1]

        # print(start)
        # print(end)
        # print(w, h)

        # out_img = out_img[:, :, 0]
        # out_img = np.stack([out_img] * 3, 2)
        # print(out_img.shape)

        # import matplotlib.pyplot as plt
        # plt.imshow(out_img)
        # plt.show()


        # print(out_img.size())

        # save results to test_results folder
        # if not os.path.exists(prediction_dir):
        #     os.makedirs(prediction_dir, exist_ok=True)
        # save_output(img_name_list[i_test], mask_out, prediction_dir)
        # print(out_img.size())

        # print(type(mask_out))
        # print(mask_out.shape)

        del d1,d2,d3,d4,d5,d6,d7

        # print(img_path)
        # print(mask_out, start[0], start[1], h, w)
        # print(type(mask_out_np))
        # print(mask_out_np)

        if print_info: print(start[0], start[1], h, w)

        mask_out_np_list.append(mask_out_np)
        start_x_list.append(start[0])
        start_y_list.append(start[1])
        h_list.append(h)
        w_list.append(w)

        if ((i_test % 1000) == 0) : print(i_test)



    print("Refined masks total:", refined_mask_count)
    print("Bad masks total:", bad_mask_count)

    #return
    return mask_out_np_list, start_x_list, start_y_list, h_list, w_list

if __name__ == "__main__":
    #main()
    mask_hw()
