import torch
import torch.nn as nn
import numpy as np
import sys
import cv2
import os
import math
from itertools import product
from PIL import Image
from datetime import datetime


sys.path.append("COMMON")
from unet import UNet_2D
from DatasetReal import *
import data_utils
import opti_utils


def evaluate_losses_on_data(unet, data_loader, loss_function):
    data_loss_log = []
    for srcs, tgts in data_loader:
        outputs = unet(srcs.cuda())
        loss = loss_function(outputs, tgts.cuda())
        data_loss_log.append(loss.item())
    return data_loss_log

def evaluate_arf_type_indep_loss_on_data(unet, data_loader):
    loss = 0
    cnt = 0
    for srcs, tgts in data_loader:
        outputs = unet(srcs.cuda())
        outputs = outputs.cpu().detach().numpy().copy()
        tgts = tgts.cpu().numpy().copy()
        loss += loss_util.arf_type_indep_loss_function(outputs, tgts)
        cnt += srcs.shape[0]
    return 0.1*(loss/cnt)

def patch_output_imgs(unet,src_patch_imgs):
    out_imgs = []
    src_patch_imgs = data_utils.numpy_images_to_norm_torch_data(src_patch_imgs, PATCH_SIZE, src=True) # normalize
    out_imgs = unet(src_patch_imgs)
    out_imgs = data_utils.norm_torch_data_to_numpy_images(out_imgs) # de-normalize
    #cv2.imshow("an out patch", out_imgs[0]) # for debugg
    #cv2.waitKey(0)                          # for debugg
    return out_imgs

def save_full_imgs(unet, src_imgs, tgt_imgs, OUT_PATH, epoch, filename):
    for i in range(len(src_imgs)):
        src_img = src_imgs[i]
        tgt_img = tgt_imgs[i]
        out_path = OUT_PATH +"/" + str(epoch).zfill(4) + "_" + filename + "_" + str(i).zfill(4) + ".png"
        estimate_AnnualRingField(src_img, tgt_img, unet, out_path, kmean_num=KMEAN_NUM)

def load_real_dataset(PATH, n, noise=False, train=True, end_batch=False):
    src_imgs, tgt_imgs = data_utils.get_image_data(PATH)
    if n<len(src_imgs):
        src_imgs = src_imgs[0:n]
        tgt_imgs = tgt_imgs[0:n]
    print("Loaded data from:", PATH, len(src_imgs), "image pairs\n")
    dataset = RealWoodData(src_imgs, tgt_imgs, PATCH_SIZE, n, noise=noise, train=train)
    if end_batch: loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE_END, shuffle=True)
    else: loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader, dataset

def estimate_AnnualRingField(src_img, unet, save=False, export_file_path='', kmean_num=-1):

    H, W, _   = src_img.shape
    img       = np.zeros((H,W), dtype=np.float32)
    img_count = np.zeros((H,W), dtype=np.float32)

    #gen gaussian mask (PATCH_SIZE x PATCH_SIZE)
    mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    cx, cy = (PATCH_SIZE-1)/2, (PATCH_SIZE-1)/2
    sigma2  = 8.0 * 8.0
    for y in range(PATCH_SIZE):
        for x in range(PATCH_SIZE):
            mask[y,x] = math.exp( -((x-cx) ** 2 + (y-cy)**2) / sigma2 )

    for yi, xi in product(range(0,H,PATCH_SIZE//2), range(0,W,PATCH_SIZE//2)):
        y  = yi if yi < H - PATCH_SIZE else H - PATCH_SIZE
        x  = xi if xi < W - PATCH_SIZE else W - PATCH_SIZE
        in_patch = src_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE,:]

        in_patch = data_utils.numpy_image_to_norm_torch_data(in_patch, PATCH_SIZE, src=True, lst_out=True) # normalize
        out_patch = unet(in_patch.cuda())[0]
        out_patch = data_utils.norm_torch_data_to_numpy_image(out_patch) # de-normalize

        out_patch = out_patch.astype(np.float32).reshape(64,64)
        img[y:y+PATCH_SIZE,x:x+PATCH_SIZE] += mask * out_patch
        img_count[y:y+PATCH_SIZE,x:x+PATCH_SIZE] += mask


    img = img / img_count
    img = np.clip(img, 0, 255)
    img = np.uint8( img )

    if save: cv2.imwrite(export_file_path, img)

    #cv2.imshow("img", img) #for debugg
    #cv2.waitKey(0)         #for debugg

    return img

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyper parameters
# Recommended target training data size and number of epoch parameters
TARGET_TRAINING_DATA_SIZE = 5120
NUM_EPOCHS = 2000
# Parameters to run through code fast for debeugging - uncomment for actual training
TARGET_TRAINING_DATA_SIZE = 32 #for debugging
NUM_EPOCHS = 20 #for debugging
#Other parameters
BATCH_SIZE = 256
BATCH_SIZE_END = 128
LEARNING_RATE = 0.005
LEARNING_RATE_END = 0.0005
FINE_NETLOCK_START = 0
FINE_NETLOCK_END = 79
PATCH_SIZE = 64
LAMBDA = 0.001  # Adjust this regularization strength as needed
REGULARIZER = nn.MSELoss()  

# Dataset path
DATASET_NAME = 'ImagePairs'
DATA_PATH = DATASET_NAME + "\\"

# Save location of current experiment outputs
OUT_PATH = "unet_output"
if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)


def train_model():

    # start time
    start_time = datetime.now()
    elapsed_days, elapsed_hrs, elapsed_mins, elapsed_secs = opti_utils.get_elapsed_time(start_time)

    # dataset
    loader_train, dataset_train = load_real_dataset(DATA_PATH + "training_data", TARGET_TRAINING_DATA_SIZE, noise=True, train=True)
    loader_test, dataset_test =  load_real_dataset(DATA_PATH + "test_data",  int(0.1*TARGET_TRAINING_DATA_SIZE), noise=True, train=False)
    
    imgs_real, _ = data_utils.get_image_data(DATA_PATH + "training_data", grayscale=False)
    org_train_real_len = len(imgs_real)
    imgs_real, _ = data_utils.get_image_data(DATA_PATH + "test_data", grayscale=False)
    org_test__real_len = len(imgs_real)
    aug_train_real_len = sum(len(batch[0]) for batch in loader_train)
    aug_test_real_len  = sum(len(batch[0]) for batch in loader_test)

    # collect a number of patches for displaying results (output image)
    src_test, tgt_test = dataset_test.get_items(10)
    grid_img_patch = data_utils.ImageGrid(OUT_PATH, "gridImgPatches", src_test, tgt_test)

    # setup plot
    PLT_YLIM = [0.0, 0.30] # y-axis range of plot
    PLT_SUBTITLE = "Training results"
    PLT_SUBTITLE+= ". Size: " +       str(org_train_real_len) + ' ('+ str(org_test__real_len) + ')'
    PLT_SUBTITLE+= ". Augmented: " + str(aug_train_real_len) + ' ('+ str(aug_test_real_len) + ')'

    # initialize model and optimizer
    unet = UNet_2D(in_dim=3, out_dim=1).to(DEVICE)  # Instantiate your model class and move it to the device
    optimizer = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE)

    # print network parameters
    for i, (name, para) in enumerate(unet.named_parameters()):
        print("-"*20)
        print(i, ":", f"name: {name}")
        #print("values: ")
        #print(para)


    loss_function = torch.nn.L1Loss(reduction="mean")
    
    train_loss_log = []
    train_img_loss_log = []
    
    
    for epoch in range(NUM_EPOCHS):

        ########## Training mode ##########
        unet.train()
        epoch_loss_log = []
        epoch_img_loss_log = []
        epoch_mag_loss_log = []
        epoch_dir_loss_log = []
        epoch_fft_loss_log = []
        count = 0

        for srcs, tgts in loader_train:

            outputs = unet(srcs.cuda())

            # Calculate the loss
            iloss = loss_function(outputs, tgts.cuda())
            epoch_img_loss_log.append(iloss.item())
            loss = iloss
            if LAMBDA > 0:
                for param in unet.parameters():
                    loss += LAMBDA * REGULARIZER(param, torch.zeros_like(param))
            epoch_loss_log.append(loss.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            print('>', end='', flush=True)
        print('>')

        ########## Evaluation mode ##########
        unet.eval()

        # Numberical evaluation
        epoch_test_real_loss_log = evaluate_losses_on_data(unet, loader_test, loss_function)

        train_loss_log.append(     sum(epoch_loss_log)/len(epoch_loss_log))
        train_img_loss_log.append( sum(epoch_img_loss_log)/len(epoch_img_loss_log))
        
        # save intermediate output images
        #if (epoch!=0 and epoch%100==0) or (epoch==NUM_EPOCHS-1):

        #########update learning rate if dynamic############
        if LEARNING_RATE_END>0:
            lr_step = (LEARNING_RATE-LEARNING_RATE_END)/(NUM_EPOCHS)
            lr_new = LEARNING_RATE - epoch*lr_step
            #print('Updated learning rate to', lr_new)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_new
        
        if (epoch/NUM_EPOCHS)>=0.89 and (epoch/NUM_EPOCHS)<=0.91:

            print("Epoch for potentially changing bactch size and freezing layers:", epoch)
        
            if BATCH_SIZE_END>0:
                loader_train, dataset_train = load_real_dataset(DATA_PATH + "training_data", TARGET_TRAINING_DATA_SIZE, noise=True, train=True, end_batch=True)

                print('Updated batch size to', BATCH_SIZE_END)
                print() 

            if FINE_NETLOCK_START>=0 and FINE_NETLOCK_END>0:

                # lock some parts of the network
                for i, (name, para) in enumerate(unet.named_parameters()):
                    if i>= FINE_NETLOCK_START and i<=FINE_NETLOCK_END:
                        para.requires_grad = False
                        print(i, ":", f"name: {name}", "locked")
                    else:
                        if not para.requires_grad: print("error. should not be locked")
                        print(i, ":", f"name: {name}")
                    #print("values: ")
                    #print(para)

                print('Froze layers', FINE_NETLOCK_START, 'to', FINE_NETLOCK_END)
                print() 


            test_out_imgs_patch = patch_output_imgs(unet,src_test)
            grid_img_patch.add_column(test_out_imgs_patch)

            #time
            elapsed_days, elapsed_hrs, elapsed_mins, elapsed_secs = opti_utils.get_elapsed_time(start_time)

        #plot
        opti_utils.epoch_export_plot(OUT_PATH, train_loss_log, train_img_loss_log, PLT_SUBTITLE, elapsed_days, elapsed_hrs, elapsed_mins, NUM_EPOCHS, PLT_YLIM)


    #save model
    torch.save(unet.state_dict(),OUT_PATH+"//unet_trained_model.pt")
    print("Model saved\n")

def main():

    #check cuda
    print("\nCuda is available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Current device name:", torch.cuda.get_device_name(0),"\n")

    train_model()
    
if __name__ == '__main__' :
    main()
