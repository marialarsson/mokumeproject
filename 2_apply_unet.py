import torch
import numpy as np
import sys
import cv2
import os
import math
from itertools import product
import argparse

sys.path.append("COMMON")
from unet import UNet_2D
import data_utils


def estimate_AnnualRingField(src_img, unet, save=False, export_file_path=''):

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

    #cv2.imshow("img", img) 
    #cv2.waitKey(0)         

    return img

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 64

def run_model():

    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-sample', type=str,   default="CN03", help='Chosen sample')
    args = parser.parse_args()
    
    # Sample name and location, path to folder
    SAMPLE_NAME = args.sample
    print("Sample", SAMPLE_NAME)
    DATA_FOLDER_PATH = "Samples\\"
    IN_OUT_PATH = DATA_FOLDER_PATH + SAMPLE_NAME + "\\"

    # initiate model
    unet = UNet_2D(in_dim=3, out_dim=1).to(DEVICE)
    unet.load_state_dict(torch.load("unet_trained_model.pt"))
    unet.eval()
     
    # loop cube faces
    file_names = ["A", "B", "C", "D", "E", "F"]
    for ltr in file_names:
        in_data_path = IN_OUT_PATH + ltr + "_col.png"
        out_data_path = IN_OUT_PATH + ltr + "_arl-unet.png"
        src_img = cv2.imread(in_data_path)
        src_img = cv2.resize(src_img, (256, 256), interpolation=cv2.INTER_CUBIC) 
        cv2.imshow("img",src_img)
        cv2.waitKey(1)
        out_img = estimate_AnnualRingField(src_img, unet, save=True, export_file_path=out_data_path)
        cv2.imshow("img",out_img)
        cv2.waitKey(1)
        print("Saved output image in", out_data_path)
    
def main():

    #check cuda
    print("\nCuda is available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Current device name:", torch.cuda.get_device_name(0),"\n")

    run_model()
    
if __name__ == '__main__' :
    main()




