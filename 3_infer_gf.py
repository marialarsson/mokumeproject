import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import sys
from datetime import datetime
import argparse
from tqdm import tqdm
import cv2

# Classes
from _ProceduralParameters import ProceduralParameters
from _DataInstance import DataInstance

# Functions and utilities
sys.path.append("COMMON")
from procedural_wood_function import *
import data_utils
import loss_utils
import opti_utils

# Constants and global settings
torch.pi = torch.acos(torch.zeros(1)).item() * 2
torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(sci_mode=False)

def main():

    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-sample', type=str,   default="CN03", help='Chosen sample')
    args = parser.parse_args()
    
    # Sample name and location
    SAMPLE_NAME = args.sample
    print("Sample", SAMPLE_NAME)
    DATA_FOLDER_PATH = "Samples\\"

    # Optimization parameters
    ITER_NUM = 200
    LEARNING_RATE = 0.2
    LAMBDA = 0.001 

    # Setup
    start_time = datetime.now()
    dim = 256

    # Load target images
    target_img_folder_path = DATA_FOLDER_PATH + SAMPLE_NAME +"//"
    rgb_imgs, arl_imgs, _ = data_utils.get_cube_side_images(target_img_folder_path, H=dim, W=dim)
    
    # Initialize data instance class and coordinates of each face of the cube
    target_data = DataInstance(dim, dim, dim, TARGET=True)
    target_data.update_rgb_imgs_from_numpy(rgb_imgs)
    target_data.create_white_balanced_rgb_imgs()
    target_data.update_average_wb_rgb_color()
    target_data.update_arf_imgs_from_numpy(arl_imgs)
    output_data = DataInstance(dim, dim, dim, OUTPUT=True)
    out_img_coords = data_utils.generate_cuboid_coordinates(dim,dim,dim)
    target_data.get_contours(out_img_coords)

    # Initialize lists for optimization
    loss_log = []
    isoContour_loss_log = []
    regularization_log = []
    best_i = 0
    min_loss = torch.tensor(99999.9)

    # Initialize parmameter class
    params = ProceduralParameters()

    # Make list of pith axis for initla discontinous search
    Xs = []       
    dir = torch.tensor([0.0, 1.0, 0.0])
    dir += 0.01 * (torch.rand(3)-0.5)
    dir = dir/np.linalg.norm(dir)
    for px in range(-3,2,4):
        for py in range(-3,2,4):
            off = torch.tensor([0.5*px, 0.0, 0.5*py])
            off = torch.tensor(off)
            X = torch.cat((off, dir), dim=0)
            Xs.append(X)

    # Vertical lines at discontinous search points for displaying in plot
    VL0s = [(index + 1) for index in range(len(Xs))]

    # Optimization loop 
    for i in tqdm(range(ITER_NUM), desc=SAMPLE_NAME):

        if i<len(Xs): # If initial discontinous grid search stage
            X = Xs[i]
            X.requires_grad_()
            optimizer = Adam([X], lr=LEARNING_RATE)
        elif i==len(Xs): # Else if first iteration after discontious grid search
            X = torch.from_numpy(best_X) # Reinstate the best initial pith axis
            X.requires_grad_()
            optimizer = Adam([X], lr=LEARNING_RATE)
        
        # Update parameters
        params.update_init_pith_parameters(X)

        # Apply procedural funciton
        img_gtfs = []

        for j,px_coords in enumerate(out_img_coords):

            px_coords = px_coords.view(-1,3)

            img_gtf = procedural_wood_function_for_initialization(params, px_coords, A=256, B=256, return_reshaped=True)
            img_gtfs.append(img_gtf)
        
        output_data.update_gtf_imgs_from_torch(img_gtfs)
        output_data.update_gtf_map_imgs(with_contours=False)

        # Compute the iso contour loss
        isoContour_loss = 0
        loss_imgs = []

        for j in range(6):

            tgt_pxs = target_data.contour_pixels[j]
            tgt_pos = target_data.contour_positions[j]

            loss_value, loss_img_loc = loss_utils.iso_contour_loss(tgt_pxs, tgt_pos, params, dim, dim)
            isoContour_loss += loss_value
            loss_imgs.append(loss_img_loc)

        loss = isoContour_loss
        output_data.update_loss_imgs_from_np(loss_imgs)

        # Add regularization term
        regularization_term = LAMBDA * (X ** 2).sum()
        regularization_log.append(float(regularization_term.detach()))
        loss += regularization_term
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        ####################################################################

        # if better
        if loss<min_loss:
            best_X = X.detach().numpy()
            min_loss = loss.detach()
            best_i = i
        
        # Append loss logs
        isoContour_loss_log.append(float(isoContour_loss))
        loss_log.append(float(loss))

        # Plot optimization progress
        out_display_height = 256
        loss_list = [loss_log, isoContour_loss_log]
        loss_lbls = ["Total", "IsoContour loss"]
        plt_img = data_utils.get_plot_image(loss_list, loss_lbls, regularization_log, best_i, min_loss, ITER_NUM, H=out_display_height, VL0s=VL0s, simple=False)

        imgs = [output_data.unfolded_gtf_map_img]
        txts = ['']
        map_imgs = [output_data.unfolded_loss_img]
        map_txts = ['']
        map_cmaps = ['cool']
        img = data_utils.assemble_images(imgs, txts, map_imgs, map_txts, map_cmaps, out_display_height, simple=False)
        #img = np.hstack([img,plt_img])
        cv2.imshow("Growth field optimization", img)
        #cv2.waitKey(1)
        
        
    dd, dh, dm, ds = opti_utils.get_elapsed_time(start_time)
    print("Computation time:", dm, "min", ds, "s")

    # Save optimized X
    #file_name_pith_parameters = DATA_FOLDER_PATH + "pith.npy"
    #np.save(file_name_pith_parameters, X)
    #print("Saved pith parameters in", file_name_pith_parameters)


if __name__ == "__main__":

    print(torch.__version__)

    main()
