import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import sys
import cv2
import os
from datetime import datetime
import argparse
import imageio
import itertools
import copy
import pickle


from _ProceduralParameters import ProceduralParameters
from _DataInstance import DataInstance
from procedural_wood_function import *

sys.path.append("COMMON")
import opti_utils
import data_utils
import image_utils
import loss_utils

torch.pi = torch.acos(torch.zeros(1)).item() * 2
torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(sci_mode=False)
torch.set_default_dtype(torch.float32)
torch.cuda.empty_cache()  # Clear cached memory
torch.cuda.reset_peak_memory_stats() 

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
    ITER_NUM = 100 # number of iterations for optimization of growth field PITH AXIS
    ITER_NUM = 10 # for fast debugging
    LEARNING_RATE = 0.02
    LAMBDA = 0.002
    
    # Resolution of R
    HEIGHT_NUM = 8
    AZIMUTH_NUM = 16
    RADIUS_NUM = 16

    # Initiate loss function
    style_loss_module = loss_utils.VGGStyleLoss()
    
    # Setup
    start_time = datetime.now()
    dim = 256

    # Load target images
    target_img_folder_path = DATA_FOLDER_PATH + SAMPLE_NAME +"//"
    rgb_imgs, arl_imgs, _ = data_utils.get_cube_side_images(target_img_folder_path, H=dim, W=dim)
    
    # Initialize data instance class and coordinates of each face of the cube
    target_data = DataInstance(dim, dim, dim, TARGET=True)
    target_data.update_rgb_imgs_from_numpy(rgb_imgs)
    #target_data.create_white_balanced_rgb_imgs()
    #target_data.update_average_wb_rgb_color()
    output_data = DataInstance(dim, dim, dim, OUTPUT=True)
    out_img_coords = data_utils.generate_cuboid_coordinates(dim,dim,dim)
    target_data.get_contours(out_img_coords)

    # Initialize lists for optimization
    loss_log = []
    style_loss_log = []
    best_i = 0
    min_loss = 99999.9

    # Initialize parmameter class
    params = #... import by piclke
        
    # Optimization loop 
    img_frames = []
    for i in tqdm(range(ITER_NUM), desc=SAMPLE_NAME):


        RENDER_KNOT = False
        RENDER_FIBERS = True
        RENDER_PORES = False
        RING_POROUS = False
        RENDER_RAYS = False

        
        extra_ew_lw_cols    = torch.zeros_like(ew_lw_cols).requires_grad_()
        extra_lw_end_start  = torch.zeros_like(lw_end_start).requires_grad_()
        lw_end_start_linear = torch.zeros(2).requires_grad_()
        parameters_list = [extra_ew_lw_cols, extra_lw_end_start, lw_end_start_linear]
        optimizer = Adam(parameters_list, lr=LEARNING_RATE)
        autograd_optimization=True
                
        #fibers grid search
        FIBER_ITERS=0
        if RENDER_FIBERS:
            params.init_fiber_parameters()
            fiber_sizes = torch.arange(0.0, 1.1, 0.2, dtype=torch.float32) 
            fiber_discont_grid = fiber_sizes
            best_dis_fiber_params = fiber_discont_grid[0]
            FIBER_ITERS = fiber_discont_grid.size()[0]
        
        #pores grid search
        PORE_ITERS = 0
        if RENDER_PORES:
            params.init_pore_parameters()
            pore_cell_dim_ads = torch.arange(0.1, 0.8, 0.3, dtype=torch.float32)
            pore_cell_dims_hs = torch.arange(0.2, 1.0, 0.5, dtype=torch.float32)
            if RING_POROUS:
                pore_gen_occ_params = torch.tensor([0.0, 0.5])
                pore_occ_ring_ratio = torch.tensor([0.4, 0.8])
                pore_lw_occ_params = torch.tensor([1.0])
            else:
                pore_gen_occ_params = torch.tensor([0.4, 0.7])
                pore_occ_ring_ratio = torch.tensor([0.1, 0.4])
                pore_lw_occ_params = torch.tensor([0.2])
            pore_discont_grid = list(itertools.product(pore_cell_dim_ads, pore_cell_dims_hs, pore_gen_occ_params, pore_occ_ring_ratio, pore_lw_occ_params))
            pore_discont_grid = torch.tensor(pore_discont_grid)
            pore_discont_grid = torch.cat((torch.zeros(1, pore_discont_grid.size()[1]), pore_discont_grid), dim=0)
            best_dis_pore_params = pore_discont_grid[0]
            PORE_ITERS = pore_discont_grid.size()[0]
        
        #rays
        RAY_ITERS = 0
        if RENDER_RAYS:
            params.init_ray_parameters()
            ray_cell_dim_as = torch.arange(0.1, 1.0, 0.4, dtype=torch.float32)
            ray_cell_dims_ds = torch.arange(0.1, 1.0, 0.4, dtype=torch.float32)
            ray_cell_dims_hs = torch.arange(0.1, 1.0, 0.4, dtype=torch.float32)
            ray_discont_grid = list(itertools.product(ray_cell_dim_as, ray_cell_dims_ds, ray_cell_dims_hs))
            ray_discont_grid = torch.tensor(ray_discont_grid)
            ray_discont_grid = torch.cat((ray_discont_grid, torch.ones(ray_discont_grid.size()[0], 1)), dim=1)
            ray_discont_grid = torch.cat((torch.zeros(1, ray_discont_grid.size()[1]), ray_discont_grid), dim=0)
            best_dis_ray_params = ray_discont_grid[0]
            RAY_ITERS = ray_discont_grid.size()[0]

        # combine
        GRID_ITERS = FIBER_ITERS + PORE_ITERS + RAY_ITERS
        VL0s = (np.arange(FIBER_ITERS) + PRE_ITERS).tolist()
        VL1s = (np.arange(PORE_ITERS) + PRE_ITERS + FIBER_ITERS).tolist()
        VL2s = (np.arange(RAY_ITERS) + PRE_ITERS + FIBER_ITERS + PORE_ITERS).tolist()

        ITER_NUM = ITER_NUM_ + PRE_ITERS + GRID_ITERS
        #print("Number of iterations in total:", ITER_NUM)

        fiber_search = False
        pore_search = False
        ray_search = False
                
        for i in range(ITER_NUM):

            #grid search stage
            if i>=PRE_ITERS and i<=(PRE_ITERS + GRID_ITERS):

                autograd_optimization=False

                if i>=PRE_ITERS and i<(PRE_ITERS+FIBER_ITERS): # within fiber grid search
                    if i==PRE_ITERS:
                        fiber_search = True
                        min_loss = 9999 
                    dis_fiber_params = fiber_discont_grid[i-PRE_ITERS]
                    params.update_discontinous_fiber_parameters(dis_fiber_params)

                elif i>=(PRE_ITERS+FIBER_ITERS) and i<(PRE_ITERS+FIBER_ITERS+PORE_ITERS):  # within pore grid search
                    if i ==(PRE_ITERS+FIBER_ITERS): #first
                        pore_search = True
                        fiber_search = False
                        min_loss = 9999 
                    if RENDER_FIBERS: params.update_discontinous_fiber_parameters(best_dis_fiber_params)
                    dis_pore_params = pore_discont_grid[i-PRE_ITERS-FIBER_ITERS]
                    params.update_discontinous_pore_parameters(dis_pore_params)
                
                elif i>=(PRE_ITERS+FIBER_ITERS+PORE_ITERS) and i<(PRE_ITERS+GRID_ITERS):  # within ray grid search
                    if i==(PRE_ITERS+FIBER_ITERS+PORE_ITERS): #first
                        ray_search = True
                        fiber_search = False
                        pore_search = False
                        min_loss = 9999 
                    if RENDER_FIBERS: params.update_discontinous_fiber_parameters(best_dis_fiber_params)
                    if RENDER_PORES: params.update_discontinous_pore_parameters(best_dis_pore_params)
                    dis_ray_params = ray_discont_grid[i-PRE_ITERS-FIBER_ITERS-PORE_ITERS]
                    params.update_discontinous_ray_parameters(dis_ray_params)

                elif i==(PRE_ITERS + GRID_ITERS): #exiting grid search
                    fiber_search = False
                    pore_search = False
                    ray_search = False
                    if COLOR_MAP:   parameters_list = [col_bar]
                    else:           parameters_list = [extra_ew_lw_cols, extra_lw_end_start, lw_end_start_linear]
                    #if RENDER_KNOT: parameters_list.append(extra_knot_col_params)
                    if RENDER_FIBERS: 
                        params.update_discontinous_fiber_parameters(best_dis_fiber_params)
                        #print("reinstating best discont fiber params", best_dis_fiber_params)
                        fiber_continous_params = torch.zeros(2).requires_grad_()
                        parameters_list.append(fiber_continous_params)
                    if RENDER_PORES: 
                        params.update_discontinous_pore_parameters(best_dis_pore_params)
                        #print("reinstating best discont pore params", best_dis_pore_params)
                        pore_continous_params = torch.zeros(9).requires_grad_()
                        parameters_list.append(pore_continous_params)
                    if RENDER_RAYS: 
                        params.update_discontinous_ray_parameters(best_dis_ray_params)
                        #print("reinstating best discont ray params", best_dis_ray_params)
                        ray_continous_params = torch.zeros(11).requires_grad_()
                        parameters_list.append(ray_continous_params)
                    
                    optimizer = Adam(parameters_list, lr=LEARNING_RATE)
                    autograd_optimization=True
            
            # adjust pore grid size
            # if occurance rate is somewhat high (>0.5) and porerad is somewhat small (<0.3) - decrease the grid size
            if RENDER_PORES and not pore_search and i>PRE_ITERS:
                if (params.pore_rad/params.pore_cell_dim_ad)<0.3 and (params.pore_occurance_ratio>0.8 or params.pore_latewood_occ_dist>0.2):
                    new_ad = 0.97*params.pore_cell_dim_ad
                    #print("*****************Pore grid cell decreased*****************")
                    #print("ad", params.pore_cell_dim_ad, "-->", new_ad)
                    params.pore_cell_dim_ad = new_ad
                elif (params.pore_rad/params.pore_cell_dim_ad)>0.48 or ((params.pore_rad/params.pore_cell_dim_ad)>0.4 and params.pore_occurance_ratio<0.4 and params.pore_latewood_occ_dist<0.15):
                    new_ad = 1.03*params.pore_cell_dim_ad
                    #print("*****************Pore grid cell increased*****************")
                    #print("ad", params.pore_cell_dim_ad, "-->", new_ad)
                    params.pore_cell_dim_ad = new_ad
            
            # adjust ray grid size
            if RENDER_RAYS and not ray_search and i>PRE_ITERS:
                # increasing grid size if too large
                if (params.ray_length/params.ray_cell_dim_d)>0.48:
                    new_d = 1.1*params.ray_cell_dim_d
                    #print("*****************Ray grid cell length increased*****************")
                    #print("d", params.ray_cell_dim_d, "-->", new_d)
                    params.ray_cell_dim_d = new_d
                if (params.ray_width/params.ray_cell_dim_a)>0.48:
                    new_a = 1.1*params.ray_cell_dim_a
                    #print("*****************Ray grid cell width increased*****************")
                    #print("a", params.ray_cell_dim_a, "-->", new_a)
                    params.ray_cell_dim_a = new_a
                if (params.ray_height/params.ray_cell_dim_h)>0.48:
                    new_h = 1.1*params.ray_cell_dim_h
                    #print("*****************Ray grid cell height increased*****************")
                    #print("h", params.ray_cell_dim_h, "-->", new_h)
                    params.ray_cell_dim_h = new_h

              
            # update parameters
            if i<PRE_ITERS or i>=(PRE_ITERS + GRID_ITERS): # always except during the grid search
                if COLOR_MAP:   params.update_color_bar(col_bar, side_cols, col_bar_weight=0.1)
                else:           params.update_detailed_annual_ring_colors(extra_ew_lw_cols, extra_lw_end_start, lw_end_start_linear)
                #if RENDER_KNOT: params.update_detailed_knot_colors(extra_knot_col_params)
            if i>=(PRE_ITERS + GRID_ITERS): # after grid search
                if RENDER_FIBERS: params.update_continous_fiber_parameters(fiber_continous_params)
                if RENDER_PORES:  params.update_continous_pore_parameters(pore_continous_params)
                if RENDER_RAYS:   params.update_continous_ray_parameters(ray_continous_params)
                
                
            # Apply procedural function to each of the 6 faces
            img_rgb_outputs = []
            axes = [2,1,0,1,0,2]
            for j in range(len(out_img_coords)):

                A,B = ABs[j]
                ax = axes[j]
                px_coords = out_img_coords[j]

                if LARGE_PLATE:
                    f=2
                    px_coords = px_coords[::f, ::f, :] #taking every f-th element
                    A=int(A/f+0.6)
                    B=int(B/f+0.6)
                
                px_coords = px_coords.reshape(-1,3)  #.view(-1,3)

                if i<PRE_ITERS+GRID_ITERS:
                    RF=fiber_search
                    RP=pore_search
                    RR=ray_search
                    if pore_search or ray_search: RF=True
                    if ray_search and RENDER_PORES: RP = True

                else:
                    RF=RENDER_FIBERS
                    RP=RENDER_PORES
                    RR=RENDER_RAYS

                img_rgb_out = procedural_wood_function_refined_and_colors_and_details(params, px_coords, side_index=j, side_axis=ax, A=A, B=B, show_fiber=RF, show_pore=RP, show_ray=RR, show_knot=RENDER_KNOT, color_map=COLOR_MAP, return_reshaped=True)
                
                img_rgb_outputs.append(img_rgb_out)
            
            output_data.update_rgb_imgs_from_torch(img_rgb_outputs)
            output_data.unwhitebalance_rgb_imgs(target_data.channel_means_np)

            # Compute loss
            im_loss = 0
            fft_mse_loss = 0
            style_loss_value = 0
            rot_loss_value = 0
            wass_loss_value = 0
            hist_loss_value = 0
            blur_loss_value = 0
            mse_col_loss_value = 0

            loss_imgs = []

            for j in range(6):

                A,B = ABs[j]

                loss_img = np.zeros([B,A], dtype=np.float32)

                # image loss
                loss_value, loss_img_loc = loss_utils.image_loss(output_data.rgb_imgs_torch[j], target_data.wb_rgb_imgs_torch[j]) #white balanced images
                #loss_value, loss_img_loc = loss_utils.image_loss(output_data.rgb_imgs_torch[j], target_data.rgb_imgs_torch[j])
                im_loss += loss_value
                loss_img += loss_img_loc
                
                # fft loss
                if FFT_LOSS: fft_mse_loss += 0.005*fft_loss_module(target_data.wb_rgb_imgs_torch[j], output_data.rgb_imgs_torch[j])
                #fft_mse_loss += 0.005*fft_loss_module(target_data.rgb_imgs_torch[j], output_data.rgb_imgs_torch[j])

                #style loss
                if STYLE_LOSS: style_loss_value += 300*style_loss_module(target_data.wb_rgb_imgs_torch[j], output_data.rgb_imgs_torch[j])
                #if STYLE_LOSS: style_loss_value += 300*style_loss_module(target_data.rgb_imgs_torch[j], output_data.rgb_imgs_torch[j])

                if ROT_LOSS: rot_loss_value += 0.03*rot_loss_module(output_data.rgb_imgs_torch[j], target_data.wb_rgb_imgs_torch[j])

                #relaxed relaxed wasserstein difference
                if WASS_LOSS: wass_loss_value += 0.5*wass_loss_module(output_data.rgb_imgs_torch[j], target_data.wb_rgb_imgs_torch[j]) #2
                
                #histogram loss
                if HIST_LOSS: hist_loss_value += 20*hist_loss_module(output_data.rgb_imgs_torch[j], target_data.wb_rgb_imgs_torch[j])

                if BLUR_LOSS: blur_loss_value += 0.2*blur_loss_module(output_data.rgb_imgs_torch[j], target_data.wb_rgb_imgs_torch[j])

                if MSE_COLOR_LOSS: 
                    mean1 = output_data.rgb_imgs_torch[j].mean(dim=(0, 1))
                    mean2 = target_data.wb_rgb_imgs_torch[j].mean(dim=(0, 1))
                    channel_differences = torch.abs(mean1 - mean2)
                    average_difference = channel_differences.mean()                   
                    mse_col_loss_value += 0.5*average_difference
                
                loss_imgs.append(loss_img)

            loss = 0
            if IMAGE_LOSS:  loss += im_loss
            if FFT_LOSS:    loss += fft_mse_loss
            if STYLE_LOSS:  loss += style_loss_value
            if WASS_LOSS:   loss += wass_loss_value
            if HIST_LOSS:   loss += hist_loss_value
            if BLUR_LOSS:   loss += blur_loss_value
            if ROT_LOSS:    loss += rot_loss_value
            if MSE_COLOR_LOSS: loss += mse_col_loss_value

            output_data.update_loss_imgs_from_np(loss_imgs)

            # Add regularization term
            regularization_term = 0
            if COLOR_MAP:
                regularization_term += LAMBDA * (col_bar ** 2).mean()
            else:
                regularization_term += LAMBDA * (extra_ew_lw_cols ** 2).mean()
                regularization_term += LAMBDA * (extra_lw_end_start ** 2).mean()
                regularization_term += LAMBDA * (lw_end_start_linear ** 2).mean()
            #if RENDER_KNOT: regularization_term += LAMBDA * (knot_col_params ** 2).mean()
            if RENDER_FIBERS and i>PRE_ITERS+GRID_ITERS: regularization_term += LAMBDA * (fiber_continous_params ** 2).mean()
            if RENDER_PORES and i>PRE_ITERS+GRID_ITERS:  regularization_term += LAMBDA * (pore_continous_params ** 2).mean()
            if RENDER_RAYS and i>PRE_ITERS+GRID_ITERS:   regularization_term += LAMBDA * (ray_continous_params ** 2).mean()
            loss += regularization_term

            if autograd_optimization:
                try: 
                    optimizer.zero_grad()  
                    loss.backward()
                    optimizer.step()
                except:
                    print("Stopping early because of broken optimization")
                    break

            #if not fiber_search and not pore_search: print("X", X)
                
            # update best image if loss is better
            if i==0 or loss<min_loss:
                best_output_data.update_rgb_imgs_from_torch(output_data.rgb_imgs_torch)
                best_output_data.update_loss_imgs_from_np(output_data.loss_imgs_np)
                min_loss = loss.detach()
                best_i = i
                if autograd_optimization: 
                    if COLOR_MAP:
                        best_col_bar = col_bar.detach().clone()
                    else:
                        best_extra_ew_lw_cols = extra_ew_lw_cols.detach().clone()
                        best_extra_lw_end_start = extra_lw_end_start.detach().clone()
                        best_lw_end_start_linear = lw_end_start_linear.detach().clone()
                elif fiber_search:  best_dis_fiber_params = fiber_discont_grid[i-PRE_ITERS]
                elif pore_search:   best_dis_pore_params = pore_discont_grid[i-PRE_ITERS-FIBER_ITERS]
                elif ray_search:    best_dis_ray_params = ray_discont_grid[i-PRE_ITERS-FIBER_ITERS-PORE_ITERS]
                else: print("error")
                
            im_loss_log.append(float(im_loss))
            fft_loss_log.append(float(fft_mse_loss))
            style_loss_log.append(float(style_loss_value))
            rot_loss_log.append(float(rot_loss_value))
            wass_loss_log.append(float(wass_loss_value))
            hist_loss_log.append(float(hist_loss_value))
            blur_loss_log.append(float(blur_loss_value))
            mse_col_loss_log.append(float(mse_col_loss_value))
            reg_term_log.append(float(regularization_term))
            loss_log.append(float(loss))

            # Show image   
            if SHOW:   
                #imgs = [target_data.unfolded_wb_rgb_img, output_data.unfolded_rgb_img]
                #txts = [sample_name + ", Target", "Output"]
                imgs = [output_data.unfolded_unwb_rgb_img]
                txts = [""]
                map_imgs = []
                map_txts = []
                map_cmaps = []
                img = image_utils.assemble_images(imgs, txts, map_imgs, map_txts, map_cmaps, out_display_height, simple=False)
                loss_list = [loss_log, style_loss_log, rot_loss_log, wass_loss_log, blur_loss_log, hist_loss_log, im_loss_log, fft_loss_log, mse_col_loss_log]
                loss_lbls = ["Total", "Style loss", "Relaxed OT loss", "SWD loss", "Blur L1 img loss", "Historgam loss", "L1 image loss", "FFT loss", "MSE RGR loss"]
                plt_img = data_utils.get_plot_image(loss_list, loss_lbls, reg_term_log, best_i, min_loss, ITER_NUM, out_display_height,  VL0s=VL0s, VL1s=VL1s, VL2s=VL2s, simple=False, ymax=-1, id=CODEID)
                img = np.hstack([img,plt_img])
                cv2.imshow('Output', img)
                cv2.waitKey(1)
                gif_sequence_images.append(imageio.core.util.Array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

                if SAVE_FRAMES:
                    folder_path = out_path + "\\proc_final_output_images\\frame_" + str(i) + "\\"
                    if not os.path.exists(folder_path): os.mkdir(folder_path)
                    ltrs = ["A","B","C","D","E","F"]
                    for j in range(6):
                        file_name = folder_path + ltrs[j] + "_col.png"
                        img = output_data.unwb_rgb_imgs_np[j]
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(file_name, img)
                        #print("Saved", file_name)
            

        dd, dh, dm, ds = opti_utils.get_elapsed_time(start_time)
        print("Computation time:", dm, "min", ds, "s")

        if ITER_NUM>19 and SAVE_OUTPUT:
            saved_files = ''
            fend = '_'
            fend += '-' + str(HEIGHT_NUM) + 'x' + str(SPOKE_NUM) + 'x' + str(RING_NUM)
            ##
            params.detach_tensors()
            file_name = out_path + 'final_params' + fend + '.pkl'
            with open(file_name, 'wb') as f: pickle.dump(params, f)
            saved_files += file_name + ', '
            ##
            file_name = out_path + "result_stage5_wb" + fend + ".png"
            img = cv2.cvtColor(best_output_data.unfolded_rgb_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(file_name,img)
            saved_files += file_name + ', '
            ##
            file_name = out_path + "result_stage5" + fend + ".png"
            best_output_data.unwhitebalance_rgb_imgs(target_data.channel_means_np)
            img = cv2.cvtColor(best_output_data.unfolded_unwb_rgb_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(file_name,img)
            saved_files += file_name
            #
            print("Saved", saved_files)


    if len(gif_sequence_images)>0 and SHOW and N_MAX==1:
        imageio.mimsave("output//" + out_filename + ".gif", gif_sequence_images)
        

if __name__ == "__main__":

    main()


