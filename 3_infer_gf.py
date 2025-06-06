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
    PITH_ITER_NUM = 20 # number of iterations for optimization of growth field PITH AXIS
    DIST_ITER_NUM = 20 # number of iterations for optimization of growth field DISTORTIONS
    ITER_NUM = PITH_ITER_NUM + DIST_ITER_NUM
    LEARNING_RATE = 0.2
    PITH_LAMBDA = 0.0002 
    DIST_LAMBDA = 0.1
    PITH_STAGE = True

    # Resolution of R
    HEIGHT_NUM = 8
    AZIMUTH_NUM = 16
    RADIUS_NUM = 16

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
    target_data.update_arl_imgs_from_numpy(arl_imgs)
    target_data.update_average_arl_color()
    output_data = DataInstance(dim, dim, dim, OUTPUT=True)
    out_img_coords = data_utils.generate_cuboid_coordinates(dim,dim,dim)
    target_data.get_contours(out_img_coords)

    # Initialize lists for optimization
    loss_log = []
    isoContour_loss_log = []
    locImage_loss_log = []
    colImage_loss_log = []
    regularization_log = []
    best_i = 0
    min_loss = 99999.9
    grad_des_optim = False

    # Initialize parmameter class
    params = ProceduralParameters()

    # Make list of pith axis for initla discontinous search
    OVs = []       
    V = torch.tensor([0.0, 1.0, 0.0])
    V += 0.01 * (torch.rand(3)-0.5)
    V = V/np.linalg.norm(V)
    for px in range(-3,2,4):
        for py in range(-3,2,4):
            O = torch.tensor([0.5*px, 0.0, 0.5*py])
            O = torch.tensor(O)
            OVs.append([O,V])
    params.update_init_pith_parameters(OVs[0][0],OVs[0][1])
    VL0s = [(index + 1) for index in range(len(OVs))]     # Vertical lines at discontinous search points for displaying in plot


    """
    # Check for knot
    RENDER_KNOT = False
    COL_IMAGE_LOSS = False
    ltrs = ['A','B','C','D','E','F']
    # read knot center points - get their position in the image, translate to the 3D position
    knot_pts = []
    for j in range(6):
        file_name = out_path  + ltrs[j] + "_ann.png"
        if os.path.exists(file_name):
            ann_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            mask = (ann_img >= 180) & (ann_img <= 220)
            indices = np.argwhere(mask)
            for pixel_index in indices:                
                x = pixel_index[0]
                y = pixel_index[1]
                knot_pt = out_img_coords[j][y][x]
                knot_pts.append(knot_pt)
        else:
            break
    if len(knot_pts)==2: 
        RENDER_KNOT = True
        COL_IMAGE_LOSS = True
        if EVALUATION_MODE and EVALUATION_TYPE==4: COL_IMAGE_LOSS=False
        print("Knot identified")
        # set dir/org
        knot_org = torch.tensor(0.5*(knot_pts[0] + knot_pts[1]))
        knot_dir = torch.tensor(knot_pts[0] - knot_pts[1])
        knot_dir = knot_dir/knot_dir.norm()
        params.init_knot_parameters(knot_org, knot_dir)
        # init knot deforms
        knot_deformations = torch.zeros(8)
        best_knot_deformations = torch.zeros_like(knot_deformations)
        params.update_knot_deform_parameters(knot_deformations)
        knot_deformations.requires_grad_()
        # init knot simple colors
        simple_colors = torch.zeros(7)
        params.update_simple_colors(simple_colors)
        simple_colors.requires_grad_()"""
    
    KNOT = False
    

    # Optimization loop 
    for i in tqdm(range(ITER_NUM), desc=SAMPLE_NAME):

        
        # If initial discontinous grid search stage
        if i<len(OVs): 
            O,V = OVs[i]

        # Else if first iteration of pith after discontious grid search
        elif i==len(OVs): 
            grad_des_optim = True
            O,V = best_OV # Reinstate the best initial pith axis
            O = torch.from_numpy(O)
            V = torch.from_numpy(V)
            O.requires_grad_()
            V.requires_grad_()
            optimizer = Adam([O,V], lr=LEARNING_RATE)

        # Else if first iteration of distrotions
        elif i==PITH_ITER_NUM: 
            PITH_STAGE = False
            O.requires_grad_(False)
            V.requires_grad_(False)
            del optimizer
            # Initiate R
            R = torch.zeros(HEIGHT_NUM,AZIMUTH_NUM,RADIUS_NUM)
            height_range, spoke_range, ring_range = data_utils.get_ranges(params, out_img_coords, dim)        
            params.init_refined_procedual_parameters(HEIGHT_NUM, height_range, AZIMUTH_NUM, spoke_range, RADIUS_NUM, ring_range)
            params.update_spoke_rads(R)
            # Initialize arl bar
            params.update_average_arl_color(torch.tensor(target_data.average_arl_color/255.0))
            M = torch.zeros(256) #M is the annual ring localization 1D greymap
            params.update_base_arl_color_bar(length=M.size()[0])
            R.requires_grad_()
            M.requires_grad_()
            #if KNOT: optimizer = Adam([R, M, RK, CM], lr=LEARNING_RATE)
            optimizer = Adam([R, M], lr=LEARNING_RATE)

        # Update parameters
        if PITH_STAGE:
            params.update_init_pith_parameters(O,V)
        else:
            params.update_spoke_rads(R)
            params.update_arl_color_bar(M)
            #params.update_knot_deform_parameters(RK)

        # Apply procedural funciton
        img_gtfs = []
        img_arls = []
        #img_cols = []

        axes = [2,1,0,1,0,2]
        for j,px_coords in enumerate(out_img_coords):

            ax = axes[j]
            px_coords = px_coords.view(-1,3)

            # growth field
            #if PITH_STAGE:
            img_gtf = procedural_wood_function_for_initialization(params, px_coords, A=dim, B=dim, return_reshaped=True)
            #else:
            #    img_gtf = procedural_wood_function_for_refinement(params, px_coords, A=dim, B=dim, return_reshaped=True, show_knot=KNOT)
            img_gtfs.append(img_gtf)

            #if not PITH_STAGE:
            #    #annual ring localization image
            #    img_arl, _ = procedural_wood_function_refined_and_with_rings(params, px_coords, side_index=j, surface_normal_axis=ax, A=dim, B=dim, return_reshaped=True, show_knot=KNOT)
            #    img_arls.append(img_arl)

            #color map image
            #img_col = procedural_wood_function_knot_only(params, px_coords, side_index=j, side_axis=ax, A=A, B=B, return_reshaped=True)
            #img_cols.append(img_col)
        
        output_data.update_gtf_imgs_from_torch(img_gtfs)
        output_data.update_gtf_map_imgs(with_contours=False)
        #output_data.update_arl_imgs_from_torch(img_arls)
        #output_data.update_rgb_imgs_from_torch(img_cols)

        # Compute the iso contour loss
        isoContour_loss = 0
        locImage_loss = 0
        colImage_loss = 0
        isoContour_loss_imgs = []
        #locImage_loss_imgs = []
        #colImage_loss_imgs = []

        for j in range(6):

            tgt_pxs = target_data.contour_pixels[j]
            tgt_pos = target_data.contour_positions[j]

            #isoContour loss
            #loss_value, loss_img_loc = loss_utils.iso_contour_loss(tgt_pxs, tgt_pos, params, dim, dim, init_stage=PITH_STAGE, show_knot=KNOT)
            loss_value, loss_img_loc = loss_utils.iso_contour_loss(tgt_pxs, tgt_pos, params, dim, dim, init_stage=True, show_knot=KNOT)

            isoContour_loss += 10*loss_value
            isoContour_loss_imgs.append(loss_img_loc)

            #annual ring localization image loss
            #loss_value, loss_img_loc = loss_utils.image_loss(output_data.arl_imgs_torch[j], target_data.arl_imgs_torch[j])
            #locImage_loss += loss_value
            #locImage_loss_imgs.append(loss_img_loc)

            #color image loss
            #loss_value, loss_img_loc = loss_utils.image_loss(output_data.rgb_imgs_torch[j], target_data.rgb_imgs_torch[j])
            #colImage_loss += loss_value
            #colImage_loss_imgs.append(loss_img_loc)
            
        if PITH_STAGE:  loss = isoContour_loss
        else:           loss = isoContour_loss #+ locImage_loss #+colImage_loss

        output_data.update_loss_imgs_from_np(isoContour_loss_imgs)
        #output_data.update_loss_imgs_from_np(locImage_loss_imgs, index=1)
        #output_data.update_loss_imgs_from_np(colImage_loss_imgs, index=2)

        # Add regularization term
        regularization_term = 0
        #if PITH_STAGE:
        #    regularization_term += PITH_LAMBDA * ( (O ** 2).sum() + (V ** 2).sum())
        #else:
        #    regularization_term += 0.1 * DIST_LAMBDA*torch.pow(M,2).mean()
        #    regularization_term += DIST_LAMBDA * opti_utils.regularization_of_deformations(R)
        #    #if KNOT: 
        #    #    regularization_term += DIST_LAMBDA*torch.pow(RK,2).mean()
        #    #    regularization_term += DIST_LAMBDA*torch.pow(CM,2).mean()
        loss += regularization_term
        
        if grad_des_optim:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        ####################################################################

        # If lower loss
        if loss.item() < min_loss:
            best_OV = [O.detach().numpy(), V.detach().numpy()]
            min_loss = loss.detach()
            best_i = i
        
        # Append loss logs
        isoContour_loss_log.append(float(isoContour_loss))
        #locImage_loss_log.append(float(locImage_loss))
        #colImage_loss_log.append(float(colImage_loss))
        #regularization_log.append(float(regularization_term.detach()))
        loss_log.append(float(loss))

        # Show intermediate output images and plot optimization progress
        out_display_height = 256
        # Plot losses
        loss_list = [loss_log, isoContour_loss_log, regularization_log]
        loss_lbls = ["Total", "IsoContour loss", "Regularization"]
        plt_img = data_utils.get_plot_image(loss_list, loss_lbls, regularization_log, best_i, min_loss, ITER_NUM, H=out_display_height, VL0s=VL0s, simple=False)
        # Compose images
        imgs = [target_data.unfolded_rgb_img, target_data.unfolded_arl_img, output_data.unfolded_gtf_map_img]
        txts = ['', '', '']
        map_imgs = [output_data.unfolded_loss_img]
        map_txts = ['']
        map_cmaps = ['cool']
        img = data_utils.assemble_images(imgs, txts, map_imgs, map_txts, map_cmaps, out_display_height)
        img = np.hstack([img,plt_img])
        cv2.imshow("Growth field optimization", img)
        cv2.waitKey(1)
        
        
    dd, dh, dm, ds = opti_utils.get_elapsed_time(start_time)
    print("Computation time:", dm, "min", ds, "s")

    # Save optimized X
    #file_name_pith_parameters = DATA_FOLDER_PATH + "pith.npy"
    #np.save(file_name_pith_parameters, X)
    #print("Saved pith parameters in", file_name_pith_parameters)


if __name__ == "__main__":

    print(torch.__version__)

    main()
