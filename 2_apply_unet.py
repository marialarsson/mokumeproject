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

    run_model()
    
if __name__ == '__main__' :
    main()




