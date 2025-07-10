import numpy as np
import cv2
import argparse
import os
import sys
sys.path.append("COMMON")
import data_utils

def draw_rotated_text(text, font, font_scale, thickness, color=(0, 0, 0)):
    # Get size of the text box
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Create a transparent image (with white background)
    text_img = 255 * np.ones((h+baseline, w, 3), dtype=np.uint8)

    # Draw the text onto this image
    cv2.putText(text_img, text, (0, h), font, font_scale, color, thickness)

    # Rotate the image
    rotated = cv2.rotate(text_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return rotated

def get_side_and_cut_imgs_of_cube(cube_vol, dim=256, unwb=False, ref_imgs=[]):
    if cube_vol.ndim == 3: cube_vol = np.stack([cube_vol]*3, axis=-1) # grey to col vol
    #
    sides_and_cuts_imgs = []
    # A
    img_data = cube_vol[:,:,-1,:]
    img_data = np.rot90(img_data, k=1)
    sides_and_cuts_imgs.append(img_data)
    # B
    img_data = cube_vol[:,-1,:,:]
    img_data = np.rot90(img_data, k=1)
    img_data = np.flip(img_data, axis=0) #flip vertically
    sides_and_cuts_imgs.append(img_data)
    # C
    img_data = cube_vol[0,:,:,:]
    img_data = np.flip(img_data, axis=0) #flip vertically
    sides_and_cuts_imgs.append(img_data)
    # D
    img_data = cube_vol[:,0,:,:]
    img_data = np.rot90(img_data, k=1)
    sides_and_cuts_imgs.append(img_data)
    # E
    img_data = cube_vol[-1,:,:,:]
    img_data = np.rot90(img_data, k=2)
    sides_and_cuts_imgs.append(img_data)
    # F
    img_data = cube_vol[:,:,0,:]
    img_data = np.rot90(img_data, k=1)
    img_data = np.flip(img_data, axis=1) #flip horistonally
    sides_and_cuts_imgs.append(img_data)
    #
    # CUTS
    out_img_cutA_coords, out_img_cutD_coords = data_utils.generate_cuboid_coordinates_cutsAD(dim)
    px_coords_cut1 = out_img_cutA_coords.reshape(-1,3) 
    px_coords_cut2 = out_img_cutD_coords.reshape(-1,3) 
    #cut 1
    px_coords_cut1 = px_coords_cut1.numpy()
    px_indexes_cut = 255*(px_coords_cut1+0.5)
    px_indexes_cut = px_indexes_cut.astype(np.uint8)
    img = cube_vol[px_indexes_cut[:, 0], px_indexes_cut[:, 1], px_indexes_cut[:, 2]]
    img = img.reshape(256, 256, 3)
    img = cv2.flip(img, 0)
    sides_and_cuts_imgs.append(img)
    #chan_means = 0.66667*target_data.channel_means_np[2] + 0.33333*target_data.channel_means_np[1] #C and B
    #img = img * 2.0 * chan_means

    #cut 2
    px_coords_cut2 = px_coords_cut2.numpy()
    px_indexes_cut = 255*(px_coords_cut2+0.5)
    px_indexes_cut = px_indexes_cut.astype(np.uint8)
    img = cube_vol[px_indexes_cut[:, 0], px_indexes_cut[:, 1], px_indexes_cut[:, 2]]
    img = img.reshape(256, 256, 3)
    img = cv2.flip(img, 0)
    sides_and_cuts_imgs.append(img)

    #format color  
    for i,img in enumerate(sides_and_cuts_imgs):
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[:, :, ::-1]
        sides_and_cuts_imgs[i] = img

    if unwb: #undo the normailiztion/whitebalancing of the images
        
        # gather means of ground truth images
        org_means = []
        for img in ref_imgs: 
            org_means.append(img.mean(axis=(0, 1))/255.0)

        # undo white-balancing based on above means
        #exterior surfaces
        for i in range(6):
            img = sides_and_cuts_imgs[i]
            channel_means = org_means[i]
            img = img * 2.0 * channel_means 
            sides_and_cuts_imgs[i] = img
        #cuts
        channel_means = 0.66667*org_means[2] + 0.33333*org_means[1] #C and B
        sides_and_cuts_imgs[6] *= 2.0 * channel_means 
        channel_means = 0.66667*org_means[2] + 0.33333*org_means[0] #C and A
        sides_and_cuts_imgs[7] *= 2.0 * channel_means 

           
    #format type  
    for i,img in enumerate(sides_and_cuts_imgs):
        if np.issubdtype(img.dtype, np.floating): img = img * 255
        img = np.clip(img,0,255).astype(np.uint8)
        sides_and_cuts_imgs[i] = img

        
    # return
    return sides_and_cuts_imgs

def get_slice_img_of_cube(cube_vol, index, unwb=False, ref_img=None):

    if cube_vol.ndim == 3: cube_vol = np.stack([cube_vol]*3, axis=-1) # grey to col vol

    img = cube_vol[:,:,-1-index,:]
    img = np.rot90(img, k=1)
    
    #format color
    img = img[:, :, ::-1]

    if unwb: #undo the normailiztion/whitebalancing of the images
        channel_means = ref_img.mean(axis=(0, 1))/255.0
        img *= 2.0 * channel_means 
    
    #format type
    if np.issubdtype(img.dtype, np.floating): img = img * 255
    img = np.clip(img,0,255).astype(np.uint8)


    return img

def main():

    # Add command line arguments with flags
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', type=str, default='CN03', help='Wood sample ID (specices code and number)')
    parser.add_argument('-w', type=int, default=1000, help='Max width of output display image')
    

    args = parser.parse_args()
    SAMPLE_ID = args.id
    MAX_IMG_WIDTH = args.w
    FOLDER_PATH = "Samples\\" + SAMPLE_ID + "\\"

    ltrs = ['A', 'B', 'C', 'D', 'E', 'F']


    #read external photographs
    gt_imgs = []
    arl_unet_imgs = []
    for i in range(6):
        # gt
        file_path = FOLDER_PATH + ltrs[i] + "_col.png"
        img = cv2.imread(file_path)
        gt_imgs.append(img)
        # arl-unet
        file_path = FOLDER_PATH + ltrs[i] + "_arl-unet.png"
        img = cv2.imread(file_path)
        arl_unet_imgs.append(img)

    #cut photos
    img0 = np.zeros_like(gt_imgs[0])
    file_path = FOLDER_PATH + "cut1_col.png"
    if os.path.exists(file_path): img = cv2.imread(file_path)
    else: img = img0
    gt_imgs.append(img)
    file_path = FOLDER_PATH + "cut2_col.png"
    if os.path.exists(file_path): img = cv2.imread(file_path)
    else: img = img0
    gt_imgs.append(img)
    gt_imgs.append(img0) #vol
    #
    arl_unet_imgs.append(img0)
    arl_unet_imgs.append(img0)
    arl_unet_imgs.append(img0)
    
    #read external images of saved cube data
    #GF
    file_path = FOLDER_PATH + "gf_cube.npz"
    data = np.load(file_path)
    gf_cube = data['arr_0']
    gf_imgs = get_side_and_cut_imgs_of_cube(gf_cube)
    gf_imgs.append(img0)

    #ARL
    file_path = FOLDER_PATH + "arl_cube.npz"
    data = np.load(file_path)
    arl_cube = data['arr_0']
    arl_imgs = get_side_and_cut_imgs_of_cube(arl_cube)
    arl_imgs.append(img0)

    #COL
    file_path = FOLDER_PATH + "col_cube.npz"
    data = np.load(file_path)
    col_cube = data['arr_0']
    col_imgs = get_side_and_cut_imgs_of_cube(col_cube, unwb=True, ref_imgs=gt_imgs)
    col_imgs.append(img0)
    
    #PROC
    file_path = FOLDER_PATH + "col_cube.npz"
    data = np.load(file_path)
    proc_cube = data['arr_0']
    proc_imgs = get_side_and_cut_imgs_of_cube(proc_cube, unwb=True, ref_imgs=gt_imgs)
    proc_imgs.append(img0)

    #NCA
    file_path = FOLDER_PATH + "nca_cube.npz"
    data = np.load(file_path)
    nca_cube = data['arr_0']
    nca_imgs = get_side_and_cut_imgs_of_cube(nca_cube, unwb=True, ref_imgs=gt_imgs)
    nca_imgs.append(img0)
    
    depth_i = 0
    while True:

        slice_img = get_slice_img_of_cube(gf_cube,depth_i)
        gf_imgs[-1] = slice_img

        slice_img = get_slice_img_of_cube(arl_cube,depth_i)
        arl_imgs[-1] = slice_img

        slice_img = get_slice_img_of_cube(col_cube,depth_i, unwb=True, ref_img=gt_imgs[0])
        col_imgs[-1] = slice_img

        slice_img = get_slice_img_of_cube(proc_cube,depth_i, unwb=True, ref_img=gt_imgs[0])
        proc_imgs[-1] = slice_img

        slice_img = get_slice_img_of_cube(nca_cube,depth_i, unwb=True, ref_img=gt_imgs[0])
        nca_imgs[-1] = slice_img

        #compose image
        grid_imgs = [gt_imgs, arl_unet_imgs, gf_imgs, arl_imgs, col_imgs, nca_imgs]
        grid_imgs_text = ['GT', 'ARL (U-Net)', 'GF', 'ARL (Proc)', 'COL', 'NCA']
        row_imgs = []
        for imgs in grid_imgs:
            row_imgs.append(np.hstack(np.array(imgs)))
        out_img = np.vstack(np.array(row_imgs))

        # space and text parameters
        added_space = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2

        # add top space and text
        top_row_space = 255*np.ones((added_space, out_img.shape[1], 3), dtype=np.uint8)
        out_img = np.vstack([top_row_space, out_img])
        for i in range(9):
            if i<6:    text = ltrs[i]
            elif i==6: text = "Slanted cut 1"
            elif i==7: text = "Slanted cut 2"
            elif i==8: text = "Slice: " + str(depth_i+1) + "/256"
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_width, text_height = text_size
            text_x = int(gt_imgs[0].shape[1]*(i+0.5) - 0.5*text_width)
            text_y = added_space - 10
            cv2.putText(out_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        # add side space and text
        side_row_space = 255*np.ones((out_img.shape[0], added_space, 3), dtype=np.uint8)
        out_img = np.hstack([side_row_space, out_img])
        for i,text in enumerate(grid_imgs_text):
            text_img = draw_rotated_text(text, font, font_scale, thickness)
            text_y = added_space + int(gt_imgs[0].shape[0]*(i+0.5) - 0.5*text_img.shape[0])
            text_x = 10
            h, w = text_img.shape[:2]
            out_img[text_y:text_y+h, text_x:text_x+w] = text_img
        
        # adjust output image width
        if out_img.shape[1]>MAX_IMG_WIDTH:
            new_w = MAX_IMG_WIDTH
            ratio = MAX_IMG_WIDTH / out_img.shape[1]
            new_h = int( out_img.shape[0] * ratio )
            out_img = cv2.resize(out_img, (new_w,new_h))

        # show output image
        cv2.imshow("img", out_img)
        key = cv2.waitKey(1)

        if cv2.getWindowProperty("img", cv2.WND_PROP_VISIBLE) < 1: break # stop if window is closed
        if key == 27: break # stop if esc

        depth_i += 1
        depth_i = depth_i%256
        

    cv2.destroyAllWindows()
                        
if __name__ == "__main__":
    main()


