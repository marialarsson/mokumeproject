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
import imageio
import pickle
import os

# Functions and utilities
sys.path.append("COMMON")
from procedural_wood_function import *
import data_utils
import loss_utils
import opti_utils

# Classes
from ProceduralParameters import ProceduralParameters
from DataInstance import DataInstance

# Constants and global settings
torch.pi = torch.acos(torch.zeros(1)).item() * 2
torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(sci_mode=False)

def main():

    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', type=str,   default="CN03", help='Chosen sample')
    parser.add_argument('-show', type=int, default=1, help='Show optimization progress or not')
    args = parser.parse_args()
    
    # Sample name and location
    SAMPLE_NAME = args.id
    SHOW_OPTIM_PROG = bool(args.show)
    print("Sample", SAMPLE_NAME)
    DATA_FOLDER_PATH = "Samples\\"

    # Optimization parameters
    #PITH_ITER_NUM = 100 # number of iterations for optimization of growth field PITH AXIS
    PITH_ITER_NUM = 10 # for fast debugging
    #DIST_ITER_NUM = 100 # number of iterations for optimization of growth field DISTORTIONS
    DIST_ITER_NUM = 10 # for fast debugging
    #COL_ITER_NUM = 50
    COL_ITER_NUM = 10 # for fast degugging
    ITER_NUM = PITH_ITER_NUM + DIST_ITER_NUM + COL_ITER_NUM
    LEARNING_RATE = 0.02
    LAMBDA = 0.02
    PITH_STAGE = True
    ARL_STAGE = False

    # Resolution of R
    HEIGHT_NUM = 8
    AZIMUTH_NUM = 16
    RADIUS_NUM = 16

    # Other
    SAVE_GIF = True

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
    arlImage_loss_log = []
    colImage_loss_log = []
    regularization_log = []
    best_i = 0
    min_loss = 99999.9
    CONT_OPTIM = False

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
    VL0s.append(PITH_ITER_NUM - 1)
    VL0s.append(PITH_ITER_NUM + DIST_ITER_NUM - 1)
    

    # Check for knot
    KNOT = False
    ltrs = ['A','B','C','D','E','F']
    # read knot center points - get their position in the image, translate to the 3D position
    knot_pts = []
    for j in range(6):
        file_name = target_img_folder_path  + ltrs[j] + "_knot.png"
        if os.path.exists(file_name):
            knot_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            mask = (knot_img >= 180) & (knot_img <= 220)
            indices = np.argwhere(mask)
            for pixel_index in indices:                
                x = pixel_index[0]
                y = pixel_index[1]
                knot_pt = out_img_coords[j][y][x]
                knot_pts.append(knot_pt)
    if len(knot_pts)==2: 
        KNOT = True
        print("Knot identified")
        Ok = torch.tensor(0.5*(knot_pts[0] + knot_pts[1]))
        Vk = torch.tensor(knot_pts[0] - knot_pts[1])
        Vk = Vk/Vk.norm()
        
    # Optimization loop 
    img_frames = []
    for i in tqdm(range(ITER_NUM), desc=SAMPLE_NAME):

        # If initial discontinous grid search stage
        if i<len(OVs): 
            O,V = OVs[i]

        # Else if first iteration of pith after discontious grid search
        elif i==len(OVs): 
            CONT_OPTIM = True
            O,V = best_OV # Reinstate the best initial pith axis
            O = torch.from_numpy(O)
            V = torch.from_numpy(V)
            O.requires_grad_()
            V.requires_grad_()
            optimizer = Adam([O,V], lr=LEARNING_RATE)

        # Else if first iteration of distrotions
        elif i==PITH_ITER_NUM: 
            PITH_STAGE = False
            ARL_STAGE = True
            #
            min_loss = 9999.99
            O.requires_grad_(False)
            V.requires_grad_(False)
            params.update_init_pith_parameters(O,V)
            del optimizer
            # Initiate R
            R = torch.zeros(HEIGHT_NUM,AZIMUTH_NUM,RADIUS_NUM).requires_grad_()
            height_range, spoke_range, ring_range = data_utils.get_ranges(params, out_img_coords, dim)        
            params.init_refined_procedual_parameters(HEIGHT_NUM, height_range, AZIMUTH_NUM, spoke_range, RADIUS_NUM, ring_range)
            params.update_spoke_rads(R)
            # init knot
            if KNOT:
                params.init_knot_parameters(Ok, Vk)
                knot_deformations = torch.zeros(8).requires_grad_()
                params.update_knot_deform_parameters(knot_deformations)
            # Initialize arl bar
            params.update_average_arl_color(torch.tensor(target_data.average_arl_color/255.0))
            M = torch.zeros(128).requires_grad_() #M is the annual ring localization 1D greymap
            params.update_base_arl_color_bar(length=M.size()[0])
            #
            #if KNOT: optimizer = Adam([R, M, RK, CM], lr=LEARNING_RATE)
            parameter_list = [R, M]
            if KNOT: parameter_list.append(knot_deformations)
            optimizer = Adam(parameter_list, lr=2*LEARNING_RATE)

        elif i==PITH_ITER_NUM + DIST_ITER_NUM: 
            PITH_STAGE = False
            ARL_STAGE = False
            min_loss = 9999.99
            R.requires_grad_(False)
            M.requires_grad_(False)
            if KNOT: knot_deformations.requires_grad_(False)
            del optimizer
            params.update_spoke_rads(R)
            params.update_arl_color_bar(M)
            if KNOT: params.update_knot_deform_parameters(knot_deformations)
            # Initial col bar
            CM = torch.zeros(128,3).requires_grad_() # color map
            face_cols = torch.zeros(6,3).requires_grad_()
            mean_col = torch.tensor(target_data.average_wb_rgb_color)/255.0
            base_col_bar = mean_col.unsqueeze(0).expand(128, -1)
            params.update_base_color_bar(base_col_bar)
            parameter_list = [CM, face_cols]
            if KNOT: 
                knot_col_bar = torch.zeros(32,3).requires_grad_()
                knot_col_ani = torch.zeros(1).requires_grad_()
                params.update_knot_colors(knot_col_bar, knot_col_ani)
                parameter_list.extend([knot_col_bar, knot_col_ani])
            optimizer = Adam(parameter_list, lr=LEARNING_RATE)

        # Update parameters
        if PITH_STAGE:
            params.update_init_pith_parameters(O,V)
        elif ARL_STAGE:
            params.update_spoke_rads(R)
            params.update_arl_color_bar(M)
            if KNOT: params.update_knot_deform_parameters(knot_deformations)
        else:
            params.update_color_bar(CM, face_cols)
            if KNOT: params.update_knot_colors(knot_col_bar, knot_col_ani)

        # Apply procedural funciton
        img_gtfs = []
        img_arls = []
        img_cols = []

        axes = [2,1,0,1,0,2]
        for j,px_coords in enumerate(out_img_coords):

            ax = axes[j]
            px_coords = px_coords.view(-1,3)

            # growth field
            if PITH_STAGE:  img_gtf = procedural_wood_function_for_initialization(params, px_coords, A=dim, B=dim, return_reshaped=True)
            else:           img_gtf = procedural_wood_function_for_refinement(params, px_coords, A=dim, B=dim, return_reshaped=True, show_knot=KNOT)
            img_gtfs.append(img_gtf)

            if ARL_STAGE:
                #annual ring localization image
                img_arl, _ = procedural_wood_function_refined_and_with_1dmap(params, px_coords, side_index=j, surface_normal_axis=ax, A=dim, B=dim, return_reshaped=True, show_knot=KNOT, color_map=False)
                img_arls.append(img_arl)

            if not PITH_STAGE and not ARL_STAGE:
                #color map image
                img_col, _ = procedural_wood_function_refined_and_with_1dmap(params, px_coords, side_index=j, surface_normal_axis=ax, A=dim, B=dim, return_reshaped=True, show_knot=KNOT, color_map=True)
                #img_col = procedural_wood_function_refined_and_colors_and_details(params, px_coords, side_index=j, side_axis=ax, A=dim, B=dim, show_fiber=False, show_pore=False, show_knot=KNOT, color_map=True, return_reshaped=True)
                img_cols.append(img_col)
        
        output_data.update_gtf_imgs_from_torch(img_gtfs)
        output_data.update_gtf_map_imgs(with_contours=False)
        output_data.update_arl_imgs_from_torch(img_arls)
        output_data.update_rgb_imgs_from_torch(img_cols)
                        

        # Compute the iso contour loss
        isoContour_loss = 0
        arlImage_loss = 0
        colImage_loss = 0
        isoContour_loss_imgs = []
        arlImage_loss_imgs = []
        colImage_loss_imgs = []

        for j in range(6):

            tgt_pxs = target_data.contour_pixels[j]
            tgt_pos = target_data.contour_positions[j]

            if PITH_STAGE or ARL_STAGE:
                #isoContour loss
                loss_value, loss_img_loc = loss_utils.iso_contour_loss(tgt_pxs, tgt_pos, params, dim, dim, init_stage=PITH_STAGE, show_knot=KNOT)
                isoContour_loss += 10*loss_value
                isoContour_loss_imgs.append(loss_img_loc)
            
            if ARL_STAGE:
                #annual ring localization image loss
                loss_value, loss_img_loc = loss_utils.image_loss(output_data.arl_imgs_torch[j], target_data.arl_imgs_torch[j])
                arlImage_loss += loss_value
                arlImage_loss_imgs.append(loss_img_loc)

            if not PITH_STAGE and not ARL_STAGE:
                #color image loss
                loss_value, loss_img_loc = loss_utils.image_loss(output_data.rgb_imgs_torch[j], target_data.rgb_imgs_torch[j])
                colImage_loss += loss_value
                colImage_loss_imgs.append(loss_img_loc)
            
        if PITH_STAGE:  loss = isoContour_loss
        elif ARL_STAGE: loss = isoContour_loss + arlImage_loss
        else:           loss = colImage_loss

        output_data.update_loss_imgs_from_np(isoContour_loss_imgs)
        output_data.update_loss_imgs_from_np(arlImage_loss_imgs, index=1)
        output_data.update_loss_imgs_from_np(colImage_loss_imgs, index=2)

        # Add regularization term
        regularization_term = 0
        if PITH_STAGE:
            regularization_term += 0.01 * LAMBDA * ( (O ** 2).sum() + (V ** 2).sum())
        elif ARL_STAGE:
            regularization_term += LAMBDA*torch.pow(M,2).mean()
            regularization_term += LAMBDA * opti_utils.regularization_of_deformations(R)
        else:
            regularization_term += LAMBDA*torch.pow(CM,2).mean()
            regularization_term += 10 * LAMBDA*torch.pow(face_cols,2).mean()

        ###COL REG
        #    #if KNOT: 
        #    #    regularization_term += LAMBDA*torch.pow(RK,2).mean()
        #    #    regularization_term += LAMBDA*torch.pow(CM,2).mean()
        loss += regularization_term
        
        if CONT_OPTIM:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        ####################################################################

        # If lower loss
        if loss.item() < min_loss:
            if i<len(OVs): best_OV = [O.detach().numpy(), V.detach().numpy()]
            min_loss = loss.detach()
            best_i = i
        
        # Append loss logs
        isoContour_loss_log.append(float(isoContour_loss))
        arlImage_loss_log.append(float(arlImage_loss))
        colImage_loss_log.append(float(colImage_loss))
        regularization_log.append(float(regularization_term.detach()))
        loss_log.append(float(loss))

        if SHOW_OPTIM_PROG:

            # Show intermediate output images and plot optimization progress
            out_display_height = 256
            map_cmaps = ['cool']

            # Top row: inputs and combined plot
            loss_list = [loss_log]
            loss_lbls = ["Total"]
            plt_img = data_utils.get_plot_image(loss_list, loss_lbls, regularization_log, best_i, min_loss, ITER_NUM, H=out_display_height, VL0s=VL0s)
            imgs = [target_data.unfolded_rgb_img, target_data.unfolded_arl_img]
            txts = ['Input RGB imgs', 'U-Net generated ARL imgs']
            img = data_utils.assemble_images(imgs, txts, [], [], map_cmaps, out_display_height)
            img0 = np.hstack([img,plt_img])

            # 2nd row: contour loss
            loss_list = [isoContour_loss_log]
            loss_lbls = ["IsoContour Loss"]
            plt_img = data_utils.get_plot_image(loss_list, loss_lbls, [], best_i, isoContour_loss_log[best_i], ITER_NUM, H=out_display_height, VL0s=VL0s)
            imgs = [output_data.unfolded_gtf_map_img]
            if PITH_STAGE: txts = ['Output GF (optmizing O and V)']
            elif ARL_STAGE: txts = ['Output GF (optimizing R)']
            else: txts = ['Output GF']
            map_imgs = [output_data.unfolded_loss_img]
            map_txts = ['IsoContour Loss']
            img = data_utils.assemble_images(imgs, txts, map_imgs, map_txts, map_cmaps, out_display_height)
            img1 = np.hstack([img,plt_img])
            if not PITH_STAGE and not ARL_STAGE: img1 = np.clip(220 + 0.2*img1, 0, 255).astype(np.uint8) #ligher

            # 3rd row: grey image loss
            loss_list = [arlImage_loss_log]
            loss_lbls = ["ARL Image Loss"]
            plt_img = data_utils.get_plot_image(loss_list, loss_lbls, [], best_i, arlImage_loss_log[best_i], ITER_NUM, H=out_display_height, VL0s=VL0s)
            imgs = [output_data.unfolded_arl_img]
            txts = ['Output ARL']
            if ARL_STAGE: txts = ['Output ARL (optmizing M)']
            map_imgs = [output_data.unfolded_loss_img1]
            map_txts = ['ARL Image Loss']
            img = data_utils.assemble_images(imgs, txts, map_imgs, map_txts, map_cmaps, out_display_height)
            img2 = np.hstack([img,plt_img])
            if not ARL_STAGE: img2 = np.clip(220 + 0.2*img2, 0, 255).astype(np.uint8) #ligher

            # 4th row: col image loss
            loss_list = [colImage_loss_log]
            loss_lbls = ["RGB Image Loss"]
            plt_img = data_utils.get_plot_image(loss_list, loss_lbls, [], best_i, colImage_loss_log[best_i], ITER_NUM, H=out_display_height, VL0s=VL0s)
            imgs = [output_data.unfolded_rgb_img]
            txts = ['Output RGB']
            if not PITH_STAGE and not ARL_STAGE: txts = ['Output RGB (optmizing col map)']
            map_imgs = [output_data.unfolded_loss_img2]
            map_txts = ['RGB Image Loss']
            img = data_utils.assemble_images(imgs, txts, map_imgs, map_txts, map_cmaps, out_display_height)
            img3 = np.hstack([img,plt_img])
            if PITH_STAGE or ARL_STAGE: img3 = np.clip(220 + 0.2*img3, 0, 255).astype(np.uint8) 

            # Compose verically and show
            img = np.vstack([img0,img1,img2,img3])
            cv2.imshow("Growth field optimization and color bar initialization", img)
            cv2.waitKey(1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_frames.append(img)
        
    dd, dh, dm, ds = opti_utils.get_elapsed_time(start_time)
    print("Computation time:", dm, "min", ds, "s")

    # Save optimized X
    #file_name_pith_parameters = DATA_FOLDER_PATH + "pith.npy"
    #np.save(file_name_pith_parameters, X)
    #print("Saved pith parameters in", file_name_pith_parameters)

    # save output
    
    

    #save GF and ARL volumes

    cube_gtf_img = np.zeros((dim,dim,dim), dtype=np.float16)    
    cube_arl_img = np.zeros((dim,dim,dim), dtype=np.float16)
    cube_col_img = np.zeros((dim,dim,dim,3), dtype=np.float16)

    axes = [2,1,0,1,0,2]
    for j in tqdm(range(dim), desc="Building cubes"):
        A,B = dim, dim
        ax = axes[0]
        px_coords = out_img_coords[0]
        px_coords = px_coords.reshape(-1,3)  #.view(-1,3)
        z_val = -0.5 + j*(1.0/(dim-1))
        px_coords[:, 2] = z_val
        img_gtf = procedural_wood_function_for_refinement(params, px_coords, A=dim, B=dim, return_reshaped=True, show_knot=KNOT)
        img_arl, _ = procedural_wood_function_refined_and_with_1dmap(params, px_coords, side_index=0, surface_normal_axis=ax, A=dim, B=dim, return_reshaped=True, show_knot=KNOT, color_map=False)
        img_col, _ = procedural_wood_function_refined_and_with_1dmap(params, px_coords, side_index=0, surface_normal_axis=ax, A=dim, B=dim, return_reshaped=True, show_knot=KNOT, color_map=True)
                
        img_gtf = img_gtf.detach().numpy().astype(np.float16)
        img_arl = img_arl.detach().numpy().astype(np.float16)
        img_col = img_col.detach().numpy().astype(np.float16)

        img_gtf = cv2.rotate(img_gtf, cv2.ROTATE_90_CLOCKWISE)
        img_arl = cv2.rotate(img_arl, cv2.ROTATE_90_CLOCKWISE)
        img_col = cv2.rotate(img_col, cv2.ROTATE_90_CLOCKWISE)

        cube_gtf_img[:, :, j] = img_gtf
        cube_arl_img[:, :, j] = img_arl
        cube_col_img[:, :, j] = img_col

    file_name = target_img_folder_path + 'gf_cube.npz'
    cube_gtf_img = (cube_gtf_img - cube_gtf_img.min()) / (cube_gtf_img.max() - cube_gtf_img.min()) #normalizing 
    np.savez_compressed(file_name, cube_gtf_img)
    print("Saved", file_name)

    file_name = target_img_folder_path + 'arl_cube.npz'
    np.savez_compressed(file_name, cube_arl_img)
    print("Saved", file_name)

    file_name = target_img_folder_path + 'col_cube.npz'
    np.savez_compressed(file_name, cube_col_img)
    print("Saved", file_name)

    #save color volume



    # iso-values of annual ring locaitons
    peak_centers = data_utils.get_peak_centers_from_1d_gray_colormap(params.arl_color_bar.detach().numpy(),params)
    peak_centers = torch.from_numpy(peak_centers).to(dtype=torch.float32)
    params.update_ring_distances(peak_centers)
    params.update_median_ring_dist()

    #save procedural parameters (class instance)
    params.detach_tensors()
    file_name = target_img_folder_path + 'gf_params.pkl'
    with open(file_name, 'wb') as f: pickle.dump(params, f)
    print("Saved", file_name)

    if SAVE_GIF and len(img_frames)>1: 
        file_name = "Optimization_process_3_infer_gf.gif"
        imageio.mimsave(file_name, img_frames)
        print("Saved", file_name)


if __name__ == "__main__":

    print(torch.__version__)

    main()
