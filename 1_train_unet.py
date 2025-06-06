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
import argparse

from DatasetReal import *

sys.path.append("COMMON")
from unet import UNet_2D
import mokume_util as util
import image_util
import general_util
import data_util
import loss_util


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
    src_patch_imgs = image_util.numpy_images_to_norm_torch_data(src_patch_imgs, PATCH_SIZE, src=True) # normalize
    out_imgs = unet(src_patch_imgs)
    out_imgs = image_util.norm_torch_data_to_numpy_images(out_imgs) # de-normalize
    #cv2.imshow("an out patch", out_imgs[0]) # for debugg
    #cv2.waitKey(0)                          # for debugg
    return out_imgs

def save_full_imgs(unet, src_imgs, tgt_imgs, OUT_PATH, epoch, filename):
    for i in range(len(src_imgs)):
        src_img = src_imgs[i]
        tgt_img = tgt_imgs[i]
        out_path = OUT_PATH +"/" + str(epoch).zfill(4) + "_" + filename + "_" + str(i).zfill(4) + ".png"
        estimate_AnnualRingField(src_img, tgt_img, unet, out_path, kmean_num=KMEAN_NUM)

def load_real_dataset(PATH, n, smix=0.0, augmented=True, noise=False, train=True, grayscale=False, kmean_num=-1, end_batch=False):
    src_imgs, tgt_imgs = image_util.get_image_data(PATH)
    if n<len(src_imgs):
        src_imgs = src_imgs[0:n]
        tgt_imgs = tgt_imgs[0:n]
    print("Loaded data from:", PATH, len(src_imgs), "image pairs\n")
    dataset = RealWoodData(src_imgs, tgt_imgs, PATCH_SIZE, n, smix=smix, sgen=SYNT_WOOD_GENERATOR, noise=noise, train=train, grayscale=grayscale, kmean_num=kmean_num)
    if end_batch: loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE_END, shuffle=True)
    else: loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader, dataset

def load_synt_dataset(target_size, SYNT_WOOD_GENERATOR, train=True):
    dataset = SyntheticWoodData(target_size, PATCH_SIZE, SYNT_WOOD_GENERATOR, train=train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
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

        if kmean_num>0: in_patch = image_util.kmeans_filter(in_patch,k=kmean_num)

        in_patch = image_util.numpy_image_to_norm_torch_data(in_patch, PATCH_SIZE, src=True, lst_out=True) # normalize
        out_patch = unet(in_patch.cuda())[0]
        out_patch = image_util.norm_torch_data_to_numpy_image(out_patch) # de-normalize

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
PATCH_SIZE = 64
loss_functions = ['MSE', 'L1']

# Add command line arguments with flags
parser = argparse.ArgumentParser()
parser.add_argument('-lf',         type=int,   default= 0,    help='Loss function') #0=mse, 1=L1
parser.add_argument('-iloss',      type=int,   default= 1,    help='Image loss') #0=off, 1=on
parser.add_argument('-mloss',      type=int,   default= 0,    help='Mean image gradient magnitude loss')
parser.add_argument('-dloss',      type=int,   default= 0,    help='Image gradient direction loss')
parser.add_argument('-fftloss',    type=int,   default= 0,    help='FFT loss')
parser.add_argument('-ttds',       type=int,   default= 32,   help='Target training data size')
parser.add_argument('-ns',         type=int,   default= 5,    help='Number of epoches for synthetic data')
parser.add_argument('-nr',         type=int,   default= 5,    help='Number of epoches for real data')
parser.add_argument('-lock_st',    type=int,   default=-1,    help='Finetuning learning locked from this layer')
parser.add_argument('-lock_en',    type=int,   default=-1,    help='Finetuning learning locked until this layer')
parser.add_argument('-batch_size', type=int,   default= 8,    help='Batch size')
parser.add_argument('-batch_size1', type=int,   default= -1,    help='Batch size for last 1/10th of epoches')
parser.add_argument('-lr',         type=float, default= 0.02, help='Learning rate')
parser.add_argument('-lr1',        type=float, default= -1  , help='Learning rate end')
parser.add_argument('-finetune',   type=int,   default= 0,    help='Finetuning based on pretrained model (model trained on synthetic data)')
parser.add_argument('-ds',         type=str,   default='all', help='dataset name')
parser.add_argument('-arf',        type=int,   default=1,     help='0 is cos based type, 1 is e^cos based type, 2 is the magnitude image')
parser.add_argument('-kmean',      type=int,   default=-1,    help='2=2 colors, 3=3 colors. -1=deactivated')
parser.add_argument('-gray',       type=int,   default=0,     help='1: grayscale activated')
parser.add_argument('-smix',       type=float, default=0.0,   help='Ratio of synthethic data mixed into real data')
parser.add_argument('-run',        type=int,   default=0,     help='Run model (as opposed to train model)')


# Parse the command line arguments
args = parser.parse_args()

# Access the argument values
LOSSFUNC = loss_functions[args.lf]
IMAGE_LOSS = bool(args.iloss)
MAG_LOSS = bool(args.mloss)
DIR_LOSS = bool(args.dloss)
FFT_LOSS = bool(args.fftloss)
TARGET_TRAINING_DATA_SIZE = args.ttds
NUM_EPOCHS_SYNT = args.ns
NUM_EPOCHS_REAL = args.nr
FINE_NETLOCK_START = args.lock_st
FINE_NETLOCK_END = args.lock_en
BATCH_SIZE = args.batch_size
BATCH_SIZE_END = args.batch_size1
LEARNING_RATE = args.lr
LEARNING_RATE_END = args.lr1
FINE_TUNING = bool(args.finetune)
ARF_TYPE = args.arf
KMEAN_NUM = args.kmean
GRAYSCALE = bool(args.gray)
SYNT_MIX = args.smix #ratio of synthetic data mixed into real data. Tried this approach beore but it did not yield an improvement of the results.
RUN_MODEL = bool(args.run)

lambda_reg = 0.001  # Adjust this regularization strength as needed
regularizer = nn.MSELoss()  

DATASET_NAME = 'my_unet_data_' + args.ds
if ARF_TYPE!=0: DATASET_NAME  += '_ARFtype' + str(ARF_TYPE+1)

BASE_PATH = os.getcwd().split('git')[0] + "Dropbox\\"
DATA_PATH = BASE_PATH + DATASET_NAME + "\\"
print("\nDataset path:", DATA_PATH)

SYNT_WOOD_GENERATOR = SyntheticWoodImage(PATCH_SIZE, DATA_PATH, arf_type=ARF_TYPE, grayscale=GRAYSCALE, kmean_num=KMEAN_NUM, run_fast=RUN_MODEL) #Why does the synthetic image generator need the datapath as input? --> It samples wood colors from existing data
print("\nInitiated the synthetic wood generator") 

# Creating the filename with which to save the current experiment
name_settings = DATASET_NAME
if KMEAN_NUM>0: name_settings += "_kmean" + str(KMEAN_NUM)
if GRAYSCALE: name_settings += "_gray"
name_settings += '_' + LOSSFUNC
if not IMAGE_LOSS: name_settings += '_noImageLoss'
if MAG_LOSS: name_settings += '_magLoss'
if DIR_LOSS: name_settings += '_dirLoss'
if FFT_LOSS: name_settings += '_fftLoss'
name_settings += '_batch' + str(BATCH_SIZE)
if BATCH_SIZE_END>0: name_settings += 'to' + str(BATCH_SIZE_END)
name_settings += '_lr' + str(LEARNING_RATE)
if LEARNING_RATE_END>0: name_settings += 'to' + str(LEARNING_RATE_END)
if SYNT_MIX>0.0: name_settings += "_smix" + str(SYNT_MIX)
if FINE_NETLOCK_START>=0 or FINE_NETLOCK_END>=0: name_settings += '_lockedLayers' + str(FINE_NETLOCK_START) + "-" + str(FINE_NETLOCK_END)
if FINE_TUNING: name_settings += "_finetuning"
print("\nCurrent name:", name_settings)

# Save location of current experiment outputs
OUT_PATH = "output_" + name_settings
if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)

print('\n-Hyperparameters-')
print('Number of epochs:', NUM_EPOCHS_SYNT)
print('Learning rate:', LEARNING_RATE)
print('Loss Function:', LOSSFUNC)

# start time and initiate values measuing elapsed time
start_time = datetime.now()
elapsed_days, elapsed_hrs, elapsed_mins = general_util.get_and_save_elapsed_time(OUT_PATH, start_time, print_=True)
print("el day", elapsed_days)


# species wise full size test images real data
species_names = ['B', 'BW', 'CH', 'CN', 'H', 'IC', 'K', 'KR', 'MP', 'MZ', 'N', 'NR', 'P', 'RO', 'S', 'SG', 'TC']
species_test_src_imgs = []
species_test_tgt_imgs = []
for name in species_names:
    path = BASE_PATH + "my_unet_data_" + name
    if ARF_TYPE!=0: path  += '_ARFtype' + str(ARF_TYPE+1)
    path += "//test_data"
    simgs,  timgs = image_util.get_image_data(path, n=6, grayscale=GRAYSCALE)
    species_test_src_imgs.append(simgs)
    species_test_tgt_imgs.append(timgs)



def train_model():

    # dataset - synthetic data
    loader_train_synt, dataset_train_synt = load_synt_dataset(TARGET_TRAINING_DATA_SIZE, SYNT_WOOD_GENERATOR, train=True)
    loader_test_synt, dataset_test_synt =   load_synt_dataset(int(0.1*TARGET_TRAINING_DATA_SIZE), SYNT_WOOD_GENERATOR, train=False)

    # dataset - real data
    loader_train_real, dataset_train_real = load_real_dataset(DATA_PATH + "training_data", TARGET_TRAINING_DATA_SIZE, smix=SYNT_MIX, noise=True, train=True, grayscale=GRAYSCALE, kmean_num=KMEAN_NUM)
    loader_test_real, dataset_test_real =  load_real_dataset(DATA_PATH + "test_data",  int(0.1*TARGET_TRAINING_DATA_SIZE), smix=0.0, noise=True, train=False, grayscale=GRAYSCALE, kmean_num=KMEAN_NUM)

    start_time = datetime.now()
    elapsed_days, elapsed_hrs, elapsed_mins = general_util.get_and_save_elapsed_time(OUT_PATH, start_time, print_=True)
    
    print(DATA_PATH)

    imgs_real, _ = image_util.get_image_data(DATA_PATH + "training_data", grayscale=GRAYSCALE)
    org_train_real_len = len(imgs_real)
    imgs_real, _ = image_util.get_image_data(DATA_PATH + "test_data", grayscale=GRAYSCALE)
    org_test__real_len = len(imgs_real)
    aug_train_real_len = sum(len(batch[0]) for batch in loader_train_real)
    aug_test_real_len  = sum(len(batch[0]) for batch in loader_test_real)

    # dataset - speciewise real data
    speciswise_loaders = []
    for name in species_names:
        path = BASE_PATH + "my_unet_data_" + name
        if ARF_TYPE!=0: path  += '_ARFtype' + str(ARF_TYPE+1)
        path += "//test_data"
        spe_loader, spe_dataset =  load_real_dataset(path,  int(0.1*TARGET_TRAINING_DATA_SIZE), smix=0.0, noise=True, train=False, grayscale=GRAYSCALE, kmean_num=KMEAN_NUM)
        speciswise_loaders.append(spe_loader)

    # collect a number of patches for displaying results (output image)
    test_path_folder_path = OUT_PATH + "\\test_patches\\"

    src_real, tgt_real = dataset_test_real.get_items(10)
    src_synt, tgt_synt = dataset_test_synt.get_items(10)
    test_src_img_patches = src_real + src_synt
    test_tgt_img_patches = tgt_real + tgt_synt
    #save images to that you can use same ones if you continue training later
    if not os.path.isdir(test_path_folder_path): os.mkdir(test_path_folder_path)
    for i,img in enumerate(test_src_img_patches):
        cv2.imwrite(test_path_folder_path + str(i).zfill(2)+".png", img)

    #initiate grid images (output images)
    #grid_img_full = image_util.ImageGrid(test_src_imgs, test_tgt_imgs, OUT_PATH, "gridImgFullSize")
    #grid_img_full.add_column(test_src_imgs) #to be overwritten by output
    grid_img_patch = image_util.ImageGrid(OUT_PATH, "gridImgPatches", test_src_img_patches, test_tgt_img_patches)
    specieswise_grid_imgs_full = []
    for name,simgs,timgs in zip(species_names, species_test_src_imgs, species_test_tgt_imgs):
        print(name, len(simgs), len(timgs))
        gimg = image_util.ImageGrid(OUT_PATH, "gridImgFullSize_"+name, simgs, timgs)
        gimg.add_column(simgs) #to be overwritten by output
        specieswise_grid_imgs_full.append(gimg)


    # setup plot
    if LOSSFUNC=='MSE':  PLT_YLIM = [0.0, 0.18] # y-axis range of plot
    elif LOSSFUNC=='L1': PLT_YLIM = [0.0, 0.30] # y-axis range of plot
    if MAG_LOSS: PLT_YLIM[1] += 0.05
    if DIR_LOSS: PLT_YLIM[1] += 0.05

    PLT_SUBTITLE = "Training results"
    PLT_SUBTITLE+= ". Size: " +       str(org_train_real_len) + ' ('+ str(org_test__real_len) + ')'
    PLT_SUBTITLE+= ". Augmented: " + str(aug_train_real_len) + ' ('+ str(aug_test_real_len) + ')'

    # initialize model and optimizer
    unet = UNet_2D(in_dim=3, out_dim=1).to(DEVICE)  # Instantiate your model class and move it to the device
    if FINE_TUNING:
        if ARF_TYPE==0:
            #pre_path = os.getcwd().split('git')[0] + "Dropbox\\UnetAnnualRingDetectionModel\\20230904_pretrained\\unet_trained_model_synthetic.pt"
            #pre_path = os.getcwd().split('git')[0] + "Dropbox\\my_unet_output_sharing\\output_my_unet_data_all_MSE_batch256_lr0.01_synthetic\\unet_trained_model_synthetic.pt"
            pre_path = os.getcwd().split('git')[0] + "Dropbox\\my_unet_output_sharing\\output_my_unet_data_all_L1_batch256to128_lr0.005to0.0005_lockedLayers0-79\\unet_trained_model_final.pt"
        elif ARF_TYPE==1:
            #pre_path = os.getcwd().split('git')[0] + "Dropbox\\my_unet_output_sharing\\output_my_unet_data_all_ARFtype2_L1_batch256_lr0.01_smix0.8_synthetic\\unet_trained_model.pt"
            pre_path = os.getcwd().split('git')[0] + "Dropbox\\my_unet_output_sharing\\output_my_unet_data_all_ARFtype2_L1_batch256to128_lr0.005to0.0005_lockedLayers0-79\\unet_trained_model_final.pt"
            
        elif ARF_TYPE==2:    
            print("Not pretrained model to load for ARF3")
        unet.load_state_dict(torch.load(pre_path))
        print("\nLoaded pretrained model from", pre_path, "\n")
    optimizer = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE)

    # print network parameters
    for i, (name, para) in enumerate(unet.named_parameters()):
        print("-"*20)
        print(i, ":", f"name: {name}")
        #print("values: ")
        #print(para)


    if LOSSFUNC=='MSE':  loss_function = torch.nn.MSELoss(reduction="mean")
    elif LOSSFUNC=='L1': loss_function = torch.nn.L1Loss(reduction="mean")
    fft_loss_function = loss_util.FFTLoss()

    #test_real_arf_indep_loss = 0.10

    train_loss_log = []
    train_img_loss_log = []
    train_mag_loss_log = []
    train_dir_loss_log = []
    train_fft_loss_log = []
    test_real_loss_log = []
    test_synt_loss_log = []
    test_real_speciswise_loss_logs = []
    test_arfin_loss_log = []

    for name in species_names: test_real_speciswise_loss_logs.append([])

    NUM_PRE_EPOCHS = 0

    if NUM_EPOCHS_SYNT>0: print("--- Training on SYNTHETIC data ---")

    for epoch in range(NUM_EPOCHS_SYNT):

        ########## Training mode ##########

        unet.train()
        epoch_loss_log = []
        epoch_img_loss_log = []
        epoch_mag_loss_log = []
        epoch_dir_loss_log = []
        epoch_fft_loss_log = []
        count = 0

        for srcs, tgts in loader_train_synt:

            outputs = unet(srcs.cuda())

            # Calculate the loss
            iloss = 0
            mloss = 0
            dloss = 0
            floss = 0

            if IMAGE_LOSS:
                iloss = loss_function(outputs, tgts.cuda())
                epoch_img_loss_log.append(iloss.item())
            if MAG_LOSS:
                tgt_mag_imgs = loss_util.get_gradient_magnitude_images(tgts, acos=True)
                out_mag_imgs = loss_util.get_gradient_magnitude_images(outputs, acos=True)
                mloss = 10.0*loss_function(out_mag_imgs,tgt_mag_imgs.cuda())
                epoch_mag_loss_log.append(mloss.item())
            if DIR_LOSS:
                dloss = 0.5*loss_util.image_gradient_direction_loss(outputs,tgts.cuda())
                epoch_dir_loss_log.append(dloss.item())
            if FFT_LOSS:
                floss = 0.02 * fft_loss_function(outputs, tgts.cuda())
                epoch_fft_loss_log.append(floss.item())

            loss = iloss + mloss + dloss + floss
            epoch_loss_log.append(loss.item())



            if lambda_reg > 0:
                for param in unet.parameters():
                    loss += lambda_reg * regularizer(param, torch.zeros_like(param))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            print('>', end='', flush=True)



        ########## Evaluation mode ##########
        unet.eval()

        # numerical evaluation
        epoch_test_real_loss_log = evaluate_losses_on_data(unet, loader_test_real, loss_function)
        epoch_test_synt_loss_log = evaluate_losses_on_data(unet, loader_test_synt, loss_function)
        train_loss_log.append(     general_util.average(epoch_loss_log)           )
        train_img_loss_log.append(     general_util.average(epoch_img_loss_log)           )
        train_mag_loss_log.append(     general_util.average(epoch_mag_loss_log)           )
        train_dir_loss_log.append(     general_util.average(epoch_dir_loss_log)           )
        train_fft_loss_log.append(     general_util.average(epoch_fft_loss_log)           )
        test_real_loss_log.append( general_util.average(epoch_test_real_loss_log) )
        test_synt_loss_log.append( general_util.average(epoch_test_synt_loss_log) )
        #test_arfin_loss_log.append(test_real_arf_indep_loss)

        n_last = 10
        train_loss_last_ave = general_util.calculate_average_of_last_n_items_in_list(train_loss_log,n_last,round_dec=4)
        train_img_loss_last_ave = general_util.calculate_average_of_last_n_items_in_list(train_img_loss_log,n_last,round_dec=4)
        train_mag_loss_last_ave = general_util.calculate_average_of_last_n_items_in_list(train_mag_loss_log,n_last,round_dec=4)
        train_dir_loss_last_ave = general_util.calculate_average_of_last_n_items_in_list(train_dir_loss_log,n_last,round_dec=4)
        train_fft_loss_last_ave = general_util.calculate_average_of_last_n_items_in_list(train_fft_loss_log,n_last,round_dec=4)
        test_real_loss_last_ave = general_util.calculate_average_of_last_n_items_in_list(test_real_loss_log,n_last,round_dec=4)
        test_synt_loss_last_ave = general_util.calculate_average_of_last_n_items_in_list(test_synt_loss_log,n_last,round_dec=4)
        print("epoch:", epoch+1, "train_loss:",      "{:.4f}".format(train_loss_last_ave),
                                 "train_img_loss:",      "{:.4f}".format(train_img_loss_last_ave),
                                 "train_mag_loss:",      "{:.4f}".format(train_mag_loss_last_ave),
                                 "train_dir_loss:",      "{:.4f}".format(train_dir_loss_last_ave),
                                 "train_fft_loss:",      "{:.4f}".format(train_fft_loss_last_ave),
                                 "test_real_loss:",  "{:.4f}".format(test_real_loss_last_ave),
                                 "test_synt_loss:",  "{:.4f}".format(test_synt_loss_last_ave))

        # same as above for each species
        print("Speciswise real test losses:")
        spe_last_aves = []
        for i in range(len(species_names)):
            vals = evaluate_losses_on_data(unet, speciswise_loaders[i], loss_function)
            vals = general_util.average(vals)
            test_real_speciswise_loss_logs[i].append(vals)
            val_ave = general_util.calculate_average_of_last_n_items_in_list(test_real_speciswise_loss_logs[i],n_last,round_dec=4)
            #if len(species_names[i])==1: print(species_names[i]+' ', val_ave)
            #else: print(species_names[i], val_ave)
            spe_last_aves.append(val_ave)



        # save intermediate output images
        if (epoch!=0 and epoch%100==0) or (epoch==NUM_EPOCHS_SYNT-1):

            print("Saving losses and image output for test data.")

            data_util.save_losses_to_textfile(OUT_PATH, train_loss_last_ave, test_real_loss_last_ave, test_synt_loss_last_ave, species_names, spe_last_aves)
            data_util.save_loss_logs_to_npz(OUT_PATH, train_loss_log, test_real_loss_log, test_synt_loss_log, test_real_speciswise_loss_logs)

            # update grid images
            #test_out_imgs = [estimate_AnnualRingField(img, unet) for img in test_src_imgs]
            #grid_img_full.replace_last_column(test_out_imgs)
            test_out_imgs_patch = patch_output_imgs(unet,test_src_img_patches)
            grid_img_patch.add_column(test_out_imgs_patch)

            #save arf-comparable loss
            #test_real_arf_indep_loss = evaluate_arf_type_indep_loss_on_data(unet, loader_test_real)
            #data_util.save_arf_independant_loss_to_textfile(OUT_PATH, test_real_arf_indep_loss)
            #print("\n ARF rep. indep. test loss:", test_real_arf_indep_loss, "\n")
            #test_arfin_loss_log[-1] = test_real_arf_indep_loss


            if epoch==NUM_EPOCHS_SYNT-1: #last

                if NUM_EPOCHS_SYNT>=100:
                    for simgs,gimg in zip(species_test_src_imgs, specieswise_grid_imgs_full):
                        test_out_imgs = [estimate_AnnualRingField(img, unet, kmean_num=KMEAN_NUM) for img in simgs]
                        gimg.replace_last_column(test_out_imgs)

            #time
            elapsed_days, elapsed_hrs, elapsed_mins = general_util.get_and_save_elapsed_time(OUT_PATH, start_time, print_=True)

            torch.save(unet.state_dict(),OUT_PATH+"//unet_trained_model_synthetic.pt")
            print("Current model trained based on synthetic data only saved\n")


        #plot
        general_util.epoch_export_plot(OUT_PATH, train_loss_log, [train_img_loss_log, train_mag_loss_log, train_dir_loss_log, train_fft_loss_log], test_real_loss_log, test_synt_loss_log, test_arfin_loss_log,
                                       PLT_SUBTITLE, elapsed_days, elapsed_hrs, elapsed_mins, NUM_PRE_EPOCHS+NUM_EPOCHS_SYNT+NUM_EPOCHS_REAL,
                                       PLT_YLIM, LOSSFUNC, name_settings, NUM_EPOCHS_SYNT, swlls = test_real_speciswise_loss_logs, swns=species_names)

    if NUM_EPOCHS_SYNT>0:
        torch.save(unet.state_dict(),OUT_PATH+"//unet_trained_model_synthetic.pt")
        print("Model trained based on synthetic data only saved\n")
        grid_img_patch.add_vertical_line()

    if NUM_EPOCHS_REAL>0: print("--- Training on REAL data ---")

    # put a vertical line in output image to visualize switch from synthetic to real data
    #grid_img_full.add_vertical_line()

    # lock layers if any layers are specified to be locked during finetuning
    if FINE_TUNING and (FINE_NETLOCK_START>=0 or FINE_NETLOCK_END>=0):
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

    elapsed_days, elapsed_hrs, elapsed_mins = general_util.get_and_save_elapsed_time(OUT_PATH, start_time, print_=True)

    for epoch in range(NUM_EPOCHS_REAL):

        ########## Training mode ##########
        unet.train()
        epoch_loss_log = []
        epoch_img_loss_log = []
        epoch_mag_loss_log = []
        epoch_dir_loss_log = []
        epoch_fft_loss_log = []
        count = 0

        for srcs, tgts in loader_train_real:

            outputs = unet(srcs.cuda())

            # Calculate the loss
            iloss = 0
            mloss = 0
            dloss = 0
            floss = 0

            if IMAGE_LOSS:
                iloss = loss_function(outputs, tgts.cuda())
                epoch_img_loss_log.append(iloss.item())
            if MAG_LOSS:
                tgt_mag_imgs = loss_util.get_gradient_magnitude_images(tgts, acos=True)
                out_mag_imgs = loss_util.get_gradient_magnitude_images(outputs, acos=True)
                mloss = 2.0*loss_function(out_mag_imgs,tgt_mag_imgs.cuda())
                epoch_mag_loss_log.append(mloss.item())
            if DIR_LOSS:
                dloss = 0.1*loss_util.image_gradient_direction_loss(outputs,tgts.cuda())
                epoch_dir_loss_log.append(dloss.item())
            if FFT_LOSS:
                floss = 0.02 * fft_loss_function(outputs, tgts.cuda())
                epoch_fft_loss_log.append(floss.item())


            loss = iloss + mloss + dloss + floss
            epoch_loss_log.append(loss.item())

            if lambda_reg > 0:
                for param in unet.parameters():
                    loss += lambda_reg * regularizer(param, torch.zeros_like(param))

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
        epoch_test_real_loss_log = evaluate_losses_on_data(unet, loader_test_real, loss_function)
        epoch_test_synt_loss_log = evaluate_losses_on_data(unet, loader_test_synt, loss_function)

        train_loss_log.append(     general_util.average(epoch_loss_log)           )
        train_img_loss_log.append(     general_util.average(epoch_img_loss_log)           )
        train_mag_loss_log.append(     general_util.average(epoch_mag_loss_log)           )
        train_dir_loss_log.append(     general_util.average(epoch_dir_loss_log)           )
        train_fft_loss_log.append(     general_util.average(epoch_fft_loss_log)           )
        test_real_loss_log.append( general_util.average(epoch_test_real_loss_log) )
        test_synt_loss_log.append( general_util.average(epoch_test_synt_loss_log) )
        #test_arfin_loss_log.append(test_real_arf_indep_loss)

        n_last = 10
        train_loss_last_ave = general_util.calculate_average_of_last_n_items_in_list(train_loss_log,n_last,round_dec=4)
        train_img_loss_last_ave = general_util.calculate_average_of_last_n_items_in_list(train_img_loss_log,n_last,round_dec=4)
        train_mag_loss_last_ave = general_util.calculate_average_of_last_n_items_in_list(train_mag_loss_log,n_last,round_dec=4)
        train_dir_loss_last_ave = general_util.calculate_average_of_last_n_items_in_list(train_dir_loss_log,n_last,round_dec=4)
        train_fft_loss_last_ave = general_util.calculate_average_of_last_n_items_in_list(train_fft_loss_log,n_last,round_dec=4)
        test_real_loss_last_ave = general_util.calculate_average_of_last_n_items_in_list(test_real_loss_log,n_last,round_dec=4)
        test_synt_loss_last_ave = general_util.calculate_average_of_last_n_items_in_list(test_synt_loss_log,n_last,round_dec=4)
        print("epoch:", epoch+1, "train_loss:",      "{:.4f}".format(train_loss_last_ave),
                                 "train_img_loss:",      "{:.4f}".format(train_img_loss_last_ave),
                                 "train_mag_loss:",      "{:.4f}".format(train_mag_loss_last_ave),
                                 "train_dir_loss:",      "{:.4f}".format(train_dir_loss_last_ave),
                                 "train_fft_loss:",      "{:.4f}".format(train_fft_loss_last_ave),
                                 "test_real_loss:",  "{:.4f}".format(test_real_loss_last_ave),
                                 "test_synt_loss:",  "{:.4f}".format(test_synt_loss_last_ave))

        # same as above for each species
        #print("Speciswise real test losses:")
        spe_last_aves = []
        for i in range(len(species_names)):
            vals = evaluate_losses_on_data(unet, speciswise_loaders[i], loss_function)
            vals = general_util.average(vals)
            test_real_speciswise_loss_logs[i].append(vals)
            val_ave = general_util.calculate_average_of_last_n_items_in_list(test_real_speciswise_loss_logs[i],n_last,round_dec=4)
            #if len(species_names[i])==1: print(species_names[i]+' ', val_ave)
            #else: print(species_names[i], val_ave)
            spe_last_aves.append(val_ave)


        # save intermediate output images
        if (epoch!=0 and epoch%100==0) or (epoch==NUM_EPOCHS_REAL-1):
        #if epoch!=0 and epoch%2==0:


            #########update learning rate if dynamic############
            if LEARNING_RATE_END>0:
                lr_step = (LEARNING_RATE-LEARNING_RATE_END)/(NUM_EPOCHS_REAL)
                lr_new = LEARNING_RATE - epoch*lr_step
                print()
                print('Updated learning rate to', lr_new)
                print()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_new
            
            if (epoch/NUM_EPOCHS_REAL)>=0.89 and (epoch/NUM_EPOCHS_REAL)<=0.91:

                print("Epoch for potentially changing bactch size and freezing layers:", epoch)
            
                if BATCH_SIZE_END>0:
                    loader_train_real, dataset_train_real = load_real_dataset(DATA_PATH + "training_data", TARGET_TRAINING_DATA_SIZE, smix=SYNT_MIX, noise=True, train=True, grayscale=GRAYSCALE, kmean_num=KMEAN_NUM, end_batch=True)

                    print('Updated batch size to', BATCH_SIZE_END)
                    print() 

                if (not FINE_TUNING) and FINE_NETLOCK_START>=0 and FINE_NETLOCK_END>0:# and (epoch/NUM_EPOCHS_REAL)>=0.89 and (epoch/NUM_EPOCHS_REAL)<=0.91:

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


            print("Saving losses and image output for test data.")

            data_util.save_losses_to_textfile(OUT_PATH, train_loss_last_ave, test_real_loss_last_ave, test_synt_loss_last_ave, species_names, spe_last_aves)
            data_util.save_loss_logs_to_npz(OUT_PATH, train_loss_log, test_real_loss_log, test_synt_loss_log, test_real_speciswise_loss_logs)

            # update grid image
            #test_out_imgs = [estimate_AnnualRingField(img, unet, kmean_num=KMEAN_NUM) for img in test_src_imgs]
            #grid_img_full.replace_last_column(test_out_imgs)

            test_out_imgs_patch = patch_output_imgs(unet,test_src_img_patches)
            grid_img_patch.add_column(test_out_imgs_patch)

            #save arf-comparable loss
            #test_real_arf_indep_loss = evaluate_arf_type_indep_loss_on_data(unet, loader_test_real)
            #data_util.save_arf_independant_loss_to_textfile(OUT_PATH, test_real_arf_indep_loss)
            #print("\n ARF rep. indep. test loss:", test_real_arf_indep_loss, "\n")
            #test_arfin_loss_log[-1] = test_real_arf_indep_loss

            if epoch==NUM_EPOCHS_REAL-1: #last

                if NUM_EPOCHS_REAL>=100:
                    for simgs,gimg in zip(species_test_src_imgs, specieswise_grid_imgs_full):
                        test_out_imgs = [estimate_AnnualRingField(img, unet, kmean_num=KMEAN_NUM) for img in simgs]
                        gimg.replace_last_column(test_out_imgs)

            #time
            elapsed_days, elapsed_hrs, elapsed_mins = general_util.get_and_save_elapsed_time(OUT_PATH, start_time, print_=True)

            torch.save(unet.state_dict(),OUT_PATH+"//unet_trained_model.pt")
            print("Current model trained based on real (and possibly synthetic) data saved\n")


        #plot
        general_util.epoch_export_plot(OUT_PATH, train_loss_log, [train_img_loss_log, train_mag_loss_log, train_dir_loss_log, train_fft_loss_log], test_real_loss_log, test_synt_loss_log, test_arfin_loss_log,
                                       PLT_SUBTITLE, elapsed_days, elapsed_hrs, elapsed_mins, NUM_PRE_EPOCHS+NUM_EPOCHS_SYNT+NUM_EPOCHS_REAL,
                                       PLT_YLIM, LOSSFUNC, name_settings, NUM_EPOCHS_SYNT, swlls=test_real_speciswise_loss_logs, swns=species_names)


    if NUM_EPOCHS_REAL>0:
        #save model
        torch.save(unet.state_dict(),OUT_PATH+"//unet_trained_model_final.pt")
        print("Model saved\n")



def run_model():

    # dataset - synthetic data
    loader_train_synt, dataset_train_synt = load_synt_dataset(TARGET_TRAINING_DATA_SIZE, SYNT_WOOD_GENERATOR, train=True)
    loader_test_synt, dataset_test_synt =   load_synt_dataset(int(0.1*TARGET_TRAINING_DATA_SIZE), SYNT_WOOD_GENERATOR, train=False)

    # dataset - real data
    loader_train_real, dataset_train_real = load_real_dataset(DATA_PATH + "training_data", TARGET_TRAINING_DATA_SIZE, smix=SYNT_MIX, noise=True, train=True, grayscale=GRAYSCALE, kmean_num=KMEAN_NUM)
    loader_test_real, dataset_test_real =  load_real_dataset(DATA_PATH + "test_data",  int(0.1*TARGET_TRAINING_DATA_SIZE), smix=0.0, noise=True, train=False, grayscale=GRAYSCALE, kmean_num=KMEAN_NUM)


    print("Applying model")

    # initiate model
    unet = UNet_2D(in_dim=3, out_dim=1).to(DEVICE)

    # load model
    #MODEL_PATH = os.getcwd().split('git')[0] + "Dropbox//UnetAnnualRingDetectionModel//20230904_best//"

    #MODEL_PATH = os.getcwd().split('git')[0] + "//Dropbox//my_unet_output_sharing//output_my_unet_data_all_MSE_batch256_lr0.005//"
    #MODEL_PATH = os.getcwd().split('git')[0] + "//Dropbox//my_unet_output_sharing//output_my_unet_data_all_L1_batch256_lr0.005//"
    #MODEL_PATH = os.getcwd().split('git')[0] + "//Dropbox//my_unet_output_sharing//output_my_unet_data_all_ARFtype2_MSE_batch256_lr0.01//"
    #MODEL_PATH = os.getcwd().split('git')[0] + "//Dropbox//my_unet_output_sharing//output_my_unet_data_all_ARFtype2_L1_batch256_lr0.01//"

    MODEL_PATH = os.getcwd().split('git')[0] + "Dropbox//my_unet_output_sharing//" + OUT_PATH + "//"
    print("Model path:", MODEL_PATH)

    unet.load_state_dict(torch.load(MODEL_PATH+"unet_trained_model_final.pt"))
    print("Model loaded from", MODEL_PATH)

    unet.eval()

    APPLY_MODEL = True
    GEN_TEST = False
    EVAL_MODEL = False
    SAVE_REPRESENTATIVE_OUTPUTS = False
    LARGE_PLATES = False

    if SAVE_REPRESENTATIVE_OUTPUTS:
        species_names = ['B', 'BW', 'CH', 'CN', 'H', 'IC', 'K', 'KR', 'MP', 'MZ', 'N', 'NR', 'P', 'RO', 'S', 'SG', 'TC']
        species_long_names = ['Beech', 'Black walnut', 'Cherry', 'Chestnut', 'Hinoki', 'Icho', 'Keyaki', 'Kurumi', 'Maple', 'Mizume', 'Nire', 'Nara', 'Platanus', 'Red oak', 'Sakura', 'Sugi', 'Tochi']

        for sn in species_names:

            print(sn)

            # load data
            load_data_path = os.getcwd().split('git')[0] + 'Dropbox//my_unet_data_' + sn + '_ARFtype2//test_data//'
            export_data_path = 'rep_output//'

            file_names = os.listdir(load_data_path)

            ##  split source and target images
            src_file_names = []
            tgt_file_names = []
            for nm in file_names:
                if len(nm.split("_"))==1: src_file_names.append(nm)
                else: tgt_file_names.append(nm)

            simgs = []
            timgs = []
            oimgs = []
            error_log = []
            for sfin, tfin in zip(src_file_names, tgt_file_names):
                simg = cv2.imread(load_data_path + sfin)
                simgs.append(simg)
                timg = cv2.imread(load_data_path + tfin)
                timg = cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY)
                timgs.append(timg)
                oimg = estimate_AnnualRingField(simg, unet, save=False)
                oimgs.append(oimg)
                error_log.append(np.mean(np.abs(oimg-timg)))
            error_log = np.array(error_log)
            median_index = np.argmin(np.abs(error_log - np.median(error_log)))
            print(error_log, error_log[median_index])
            simg = simgs[median_index]
            timg = timgs[median_index]
            oimg = oimgs[median_index]
            timg = cv2.cvtColor(timg, cv2.COLOR_GRAY2BGR)
            oimg = cv2.cvtColor(oimg, cv2.COLOR_GRAY2BGR)
            border_width = 5
            white_border = np.full((simg.shape[0], border_width, 3), 255, dtype=np.uint8)
            stoimg = np.hstack([simg, white_border, timg, white_border, oimg])
            #stoimg = np.hstack([simg,timg,oimg])
            exp_path = export_data_path + sn + '.png'
            cv2.imwrite(exp_path, stoimg)
            print('Saved', exp_path)
            
    if APPLY_MODEL:
        # load data
        if LARGE_PLATES:
            exp_folder = os.getcwd().split('git')[0] + "Dropbox//diffwood_output//large_samples//"
            sample_folder = exp_folder
        if DATASET_NAME!="my_unet_data_all" and DATASET_NAME!="my_unet_data_all_ARFtype2": # differnt dataset = generalization test
            exp_folder = DATA_PATH + "outputs//"
            if not os.path.isdir(exp_folder): os.makedirs(exp_folder)
            sample_folder = os.getcwd().split('git')[0] + "Dropbox//2023_WoodDataset_Shared_Folder//samples//"
            GEN_TEST = True
        else: 
            exp_folder = os.getcwd().split('git')[0] + "Dropbox//diffwood_output//"
            sample_folder = os.getcwd().split('git')[0] + "Dropbox//2023_WoodDataset_Shared_Folder//samples//"
        
        #test_data_folder_names_ = os.listdir(exp_folder)
        ## remove files other than folders
        #test_data_folder_names = []
        #for tdfn in test_data_folder_names_:
        #    if len(tdfn.split("."))==1 and tdfn!="large_samples":
        #        test_data_folder_names.append(tdfn)
        #test_data_folder_names = ["BW06"]
        #test_data_folder_names = test_data_folder_names[:2]
        
        test_data = os.listdir(DATA_PATH + "test_data")
        test_data_folder_names = []
        for item in test_data:
            if len(item.split('_'))>1: continue #_arf.png files
            sample_name = item[:-5]
            if sample_name not in test_data_folder_names:
                test_data_folder_names.append(sample_name)
        print("Test data folder names", test_data_folder_names)

        file_names = ["A", "B", "C", "D", "E", "F"]
        for fon in test_data_folder_names:
            print("---", fon, "---")
            folder_path = sample_folder + fon + "//"
            exp_path = exp_folder + fon + "//"
            if not os.path.isdir(exp_path): os.makedirs(exp_path)
            for fin in file_names:
                data_path = folder_path + fin + "_col.png"
                exp_data_path = exp_path + fin + "_arf"
                if ARF_TYPE==1: exp_data_path += "2"
                if FFT_LOSS:    exp_data_path += "-fft"
                exp_data_path += "-unet.png"
                print("Data path", data_path)
                simg = cv2.imread(data_path)
                if not LARGE_PLATES: simg = cv2.resize(simg, (256, 256), interpolation=cv2.INTER_CUBIC) #<--cancel out if large samples
                cv2.imshow("img",simg)
                cv2.waitKey(1)
                oimg = estimate_AnnualRingField(simg, unet, save=True, export_file_path=exp_data_path)
                cv2.imshow("img",oimg)
                cv2.waitKey(1)
                print("Saved output image in", exp_data_path)
    
    if EVAL_MODEL:

        n_batches = 2  # Choose the number of batches to process

        mse_values = []
        mae_values = []

        for _ in range(n_batches):
            for srcs, tgts in loader_test_real:
                # Forward pass through the model
                outputs = unet(srcs.cuda())

                print("Evaluating model on", outputs.size(), "image outputs")
                
                # Calculate SSIM, PSNR, MSE, and MAE for each image pair
                for output, target in zip(outputs, tgts):

                    #output = output.cpu().detach().numpy()
                    #target = target.cpu().numpy()

                    output = output.squeeze().cpu().detach().numpy()
                    target = target.squeeze().cpu().numpy()

                    #print("Output shape:", output.shape, "Data type:", output.dtype)
                    #print("Target shape:", target.shape, "Data type:", target.dtype)

                    # Calculate MSE
                    mse_value = np.mean(np.power(output - target, 2))
                    mse_values.append(mse_value)

                    # Calculate MAE
                    mae_value = np.mean(np.abs(output - target))
                    mae_values.append(mae_value)

        # Average SSIM, PSNR, MSE, and MAE values across all batches
        avg_mse = np.mean(mse_values)
        avg_mae = np.mean(mae_values)
        print("Average MSE:", avg_mse)
        print("Average MAE:", avg_mae)

        data_util.save_other_losses_to_textfile(MODEL_PATH, avg_mse, avg_mae, avg_psnr, avg_ssim)
        print("Saved other losses to text file.")



def main():

    #check cuda
    print("\nCuda is available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Current device name:", torch.cuda.get_device_name(0),"\n")

    if RUN_MODEL:
        run_model()
    else:
        train_model()
    
if __name__ == '__main__' :
    main()
