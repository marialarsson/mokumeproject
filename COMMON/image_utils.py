import numpy as np
import cv2
import math
from PIL import Image
import os
import torch
import random
from matplotlib import pyplot as plt

from procedural_wood_function import *


torch.autograd.set_detect_anomaly(True)

def floatImg_to_intImg(img, scale_255=False, clip_on=True):
    if scale_255: img = (255.0*img)
    if clip_on: img = img.clip(0.0,255)
    img = img.astype(np.uint8)
    return img


def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]

    return color_range.reshape(256, 1, 3)

def assemble_images(imgs, txts, map_imgs, map_txts, map_cmaps, H, map_conts=[], simple=False):

    all_imgs = []

    for img,txt in zip(imgs,txts):
        h = img.shape[0]
        w = img.shape[1]
        if h!=H:
            w = int(w*H/h)
            img = cv2.resize(img, (w, H), interpolation=cv2.INTER_CUBIC)
        if len(img.shape)<3 or img.shape[2]==1: # greyscale image
            if img.dtype != np.int8:
                img = cv2.cvtColor(floatImg_to_intImg(img),cv2.COLOR_GRAY2RGB)
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(img, txt, (5,25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0))
        all_imgs.append(img)

    
    if not simple:
        for i, (img, txt, mcmap) in enumerate(zip(map_imgs, map_txts, map_cmaps)):
            h = img.shape[0]
            w = img.shape[1]
            if h!=H:
                w = int(w*H/h)
                img = cv2.resize(img, (w, H), interpolation=cv2.INTER_CUBIC)
            if img.dtype == np.float32: img = cv2.cvtColor(floatImg_to_intImg(img),cv2.COLOR_GRAY2RGB)
            
            #if len(map_conts)>0 and map_conts[i]:
            #    cont_img = img.copy()
            #    nc = 255.0/15.0
            #    cont_img = nc*(cont_img%nc)
            #    cont_img = cont_img.astype(np.uint8)
            #    #cv2.imshow("ci", cont_img)
            #    #cv2.waitKey(0)
            #    _, thresh = cv2.threshold(cont_img, 127, 255, cv2.THRESH_BINARY)
            #    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img = cv2.applyColorMap(img, get_mpl_colormap(mcmap))
            #if len(map_conts)>0 and map_conts[i]: img = cv2.drawContours(img, contours, -1, (0, 0, 0), thickness=1)
            cv2.putText(img, txt, (5,25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0))
            all_imgs.append(img)

    return np.hstack(all_imgs)


def mix_colors(color1, color2, alpha):
    """
    Mix two RGB colors using linear interpolation.

    Parameters:
        color1 (numpy.ndarray): First RGB color vector (3D numpy array).
        color2 (numpy.ndarray): Second RGB color vector (3D numpy array).
        alpha (float): Blending factor, ranging from 0.0 to 1.0.

    Returns:
        numpy.ndarray: Blended RGB color vector.
    """
    return (1.0 - alpha) * color1 + alpha * color2

def get_xy_indices_of_image(H,W):
    px_pos = np.indices((H, W)).astype(np.float32)
    px_pos = px_pos.reshape(2, -1).T
    px_pos = np.subtract(px_pos, np.array([0.5*H, 0.5*W]))
    px_pos[:,0] = px_pos[:,0]/min(H,W)
    #px_pos[:,1] = -px_pos[:,1]/min(H,W) #flip y axis
    px_pos[:,1] = px_pos[:,1]/min(H,W)
    px_pos = torch.from_numpy(px_pos)
    px_pos = torch.cat((px_pos, torch.zeros(px_pos.shape[0], 1)), dim=1) # add z=0 to 2D coords
    return px_pos


def xyz_coords(A,B,C,scale_factor,insert_position):
    px_pos = np.indices((A, B), dtype=np.float32)
    px_pos[0] -= 0.5*A
    px_pos[1] -= 0.5*B
    px_pos = np.transpose(px_pos, (1, 2, 0))

    if insert_position==0: px_pos = np.stack((px_pos[:, :, 1], px_pos[:, :, 0]), axis=-1)

    constant_array = np.full((px_pos.shape[0], px_pos.shape[1], 1), -0.5*C)

    if   insert_position == 0: 
        px_pos = np.concatenate((constant_array, px_pos), axis=2)

    elif insert_position == 1: 
        px_pos = np.concatenate((px_pos[:, :, :1], constant_array, px_pos[:, :, 1:]), axis=2)
    elif insert_position == 2: 
        px_pos = np.concatenate((px_pos, constant_array), axis=2)


    px_pos /= scale_factor
    return torch.from_numpy(px_pos)

def generate_cuboid_coordinates_cutsAD(hwd):

    #cut 1 (A)
    px_pos_C = xyz_coords(hwd,hwd,hwd,hwd,0)
    px_pos_C[:, :, 1] = -px_pos_C[:, :, 1] #reverse y
    px_pos_E1 = px_pos_C.clone()    
    px_pos_E1[:, :, 0] = -px_pos_E1[:, :, 0]
    px_pos_E1[:, :, 2] = -px_pos_E1[:, :, 2] #reverse z   
    # 
    px_pos_E1[:, :, 0] = px_pos_E1[:, :, 1]
    px_pos_E1[:, :, 0] += 0.5 #range -0.5 to 0.5 --> 0.0 to 1.0
    px_pos_E1[:, :, 0] *= 0.575
    px_pos_E1[:, :, 0] -= 0.375 

    #cut 2 (D)
    px_pos_C = xyz_coords(hwd,hwd,hwd,hwd,0)
    px_pos_C[:, :, 1] = -px_pos_C[:, :, 1] #reverse y
    px_pos_E = px_pos_C.clone()    
    px_pos_E[:, :, 0] = -px_pos_E[:, :, 0]
    px_pos_E[:, :, 2] = -px_pos_E[:, :, 2] #reverse z   
    # 
    px_pos_E[:, :, 0] = px_pos_E[:, :, 2]
    px_pos_E[:, :, 0] += 0.5 #range -0.5 to 0.5 --> 0.0 to 1.0
    px_pos_E[:, :, 0] *= 0.575
    px_pos_E[:, :, 0] -= 0.375 
    #px_pos_E[:, :, 0] -= 0.20

    return px_pos_E1, px_pos_E


def generate_cuboid_coordinates_teaserCut(W, H, D):

    scale_factor = 256


    #hB
    px_pos_B = xyz_coords(W,D,H,scale_factor,1)
    px_pos_B[:, :, 1] = -px_pos_B[:, :, 1] #reverse y

    px_pos_B[:,:,0] *= 0.5
    px_pos_B[:,:,0] -= W*0.25/scale_factor
    
    # tilt
    px_pos_B[:, :, 1] = px_pos_B[:, :, 0] #put X in Y
    px_pos_B[:, :, 1] *= W/H

    """
    #halfB

    px_pos_hB = np.indices((W, D)).astype(np.float32)
    px_pos_hB[0] *= 0.5
    px_pos_hB[0] -= 0.5*W
    px_pos_hB[1] -= 0.5*D
    px_pos_hB = np.transpose(px_pos_hB, (1, 2, 0))

    constant_array = np.full((px_pos_hB.shape[0], px_pos_hB.shape[1], 1), -0.5*H)
    px_pos_hB = np.concatenate((constant_array, px_pos_hB), axis=2)
    px_pos_hB /= scale_factor    
    px_pos_hB[:, :, 1] = -px_pos_hB[:, :, 1] #reverse y

    px_pos_hB = torch.from_numpy(px_pos_hB)
    """

    return px_pos_B

def generate_cuboid_coordinates(W, H, D):

    #max_hwd = 256
    max_hwd = max(max(W, H),D)

    #AF

    px_pos_A = xyz_coords(W,H,D,max_hwd,2)
    px_pos_A[:, :, 1] = -px_pos_A[:, :, 1] #reverse y
    px_pos_A[:, :, 2] = -px_pos_A[:, :, 2] #reverse z

    px_pos_F = px_pos_A.clone()
    px_pos_F[:, :, 0] = -px_pos_F[:, :, 0] #reverse x
    px_pos_F[:, :, 2] = -px_pos_F[:, :, 2] #reverse z

    #BD
    px_pos_B = xyz_coords(W,D,H,max_hwd,1)
    px_pos_B[:, :, 1] = -px_pos_B[:, :, 1] #reverse y

    px_pos_D = px_pos_B.clone()
    px_pos_D[:, :, 1] = -px_pos_D[:, :, 1] #reverse y
    px_pos_D[:, :, 2] = -px_pos_D[:, :, 2] #reverse z

    #CE
    
    px_pos_C = xyz_coords(D,H,W,max_hwd,0)
    px_pos_C[:, :, 1] = -px_pos_C[:, :, 1] #reverse y

    px_pos_E = px_pos_C.clone()    
    px_pos_E[:, :, 0] = -px_pos_E[:, :, 0]
    px_pos_E[:, :, 2] = -px_pos_E[:, :, 2] #reverse z

    return [px_pos_A, px_pos_B, px_pos_C, px_pos_D, px_pos_E, px_pos_F]

def sp_noise(image,prob,mixval,exp_size=0):
    output = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                col = np.random.rand(output[i][j].shape[0])*255
                col = mix_colors(image[i][j],col,mixval)
                col = col.astype(int)
                # Extract the neighborhood region
                min_row = max(0, i - exp_size // 2)
                max_row = min(output.shape[0], i + exp_size // 2 + 1)
                min_col = max(0, j - exp_size // 2)
                max_col = min(output.shape[1], j + exp_size // 2 + 1)
                neighborhood = output[min_row:max_row, min_col:max_col]
                neighborhood[:, :] = col
                output[i][j] = col
    return output

def filters(img, f_type_n=0, fsize=3):

    f_type = ["blur", "gaussian", "median"][f_type_n]
    '''
    ### Filtering ###
    img: image
    f_type: {blur: blur, gaussian: gaussian, median: median}
    '''
    if f_type == "blur":
        image=img.copy()
        #fsize = 3
        return cv2.blur(image,(fsize,fsize))

    elif f_type == "gaussian":
        image=img.copy()
        #fsize = 3
        return cv2.GaussianBlur(image, (fsize, fsize), 0)

    elif f_type == "median":
        image=img.copy()
        #fsize = 3
        return cv2.medianBlur(image, fsize)

def colorjitter(img, cj_type_n=0):

    cj_type= ["b", "s", "c"][cj_type_n]

    '''
    ### Different Color Jitter ###
    img: image
    cj_type: {b: brightness, s: saturation, c: constast}
    '''
    if cj_type == "b":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    elif cj_type == "s":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    elif cj_type == "c":
        brightness = 10
        contrast = random.randint(40, 100)
        dummy = np.int16(img)
        dummy = dummy * (contrast/127+1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        return img


def norm_torch_data_to_numpy_image(data):
    img = np.uint8(data.cpu().detach().numpy().copy()*255.0)
    img = img.transpose((1, 2, 0))
    img = np.clip(img, 0, 255)
    return img

def numpy_image_to_norm_torch_data(img, PATCH_SIZE, src=True, to_torch=True, lst_out=False):

    #Input image. Source: (64,64,3). Target (64,64)
    if not src and len(img.shape)>2: #target image with wrong shape
        img = img[:,:,0]

    # Resize image to patch size
    img = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_CUBIC)

    # Normalize
    img = np.float32(img)/255.0

    # Reshape
    if src: img = img.transpose((2,0,1))
    else:   img = img.reshape(1,img.shape[0], img.shape[1]) #add 1 dim

    if to_torch: img = torch.tensor(img).cuda()

    if lst_out and src: img = img.reshape(1,3,PATCH_SIZE,PATCH_SIZE) # img to list of one image

    return img


def norm_torch_data_to_numpy_images(data):
    imgs = []
    for item in data:
        imgs.append(norm_torch_data_to_numpy_image(item))
    return imgs

def numpy_images_to_norm_torch_data(imgs, PATCH_SIZE, src=True):
    data = []
    for img in imgs:
        img_norm = numpy_image_to_norm_torch_data(img, PATCH_SIZE, src=src, to_torch=False, lst_out=False)
        data.append(img_norm)
    data = torch.tensor(data).cuda()
    return data


def random_flip(img0, img1):
    flipcode = np.random.randint(2)
    if flipcode == 0:
        img0 = cv2.flip(img0,0)
        img1 = cv2.flip(img1,0)
    return img0, img1

def random_rotation(img0, img1):

    # Define the rotation angle (in degrees) and the scale
    angle = np.random.rand()*360
    scale = 1.0

    # Check that the images are of the same size
    if img0.shape[:2]!=img1.shape[:2]:
        print("Error. Images have different sizes.")
        print(img0.shape, img1.shape)

    # Rotation matrix
    W,H = img0.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)

    # Region of interest (ROI) crop parameters
    angle_rad_90 = math.radians(angle%90)
    L = int( W / ( math.sin(angle_rad_90) + math.cos(angle_rad_90) ))
    start = int((W-L)/2)
    end = int(start+L)

    # Apply rotation and crop
    img0 = cv2.warpAffine(img0, rotation_matrix, (W, H), flags=cv2.INTER_CUBIC)
    img1 = cv2.warpAffine(img1, rotation_matrix, (W, H), flags=cv2.INTER_CUBIC)
    img0 = img0[start:end, start:end]
    img1 = img1[start:end, start:end]

    return img0, img1

def random_scale(img0, img1, PATCH_SIZE, max_scale_ratio = 1.00, proportional = True):


    if proportional:

        L = min(img0.shape[0], img0.shape[1])
        if L<PATCH_SIZE: print("Random scale error. Image is too small", img0.shape)
        L2 = np.random.randint(PATCH_SIZE, int(L*max_scale_ratio))
        img0 = cv2.resize(img0, (L2, L2), interpolation=cv2.INTER_CUBIC)
        img1 = cv2.resize(img1, (L2, L2), interpolation=cv2.INTER_CUBIC)

    else:

        W = img0.shape[0]
        H = img0.shape[1]
        if W<PATCH_SIZE or H<PATCH_SIZE: print("Random scale error. Image is too small", img0.shape)
        W2 = np.random.randint(PATCH_SIZE, int(W*max_scale_ratio)+1)
        H2 = np.random.randint(PATCH_SIZE, int(H*max_scale_ratio)+1)
        img0 = cv2.resize(img0, (W2, H2), interpolation=cv2.INTER_CUBIC)
        img1 = cv2.resize(img1, (W2, H2), interpolation=cv2.INTER_CUBIC)

    return img0, img1

def random_crop(img0, img1, PATCH_SIZE):

    L = min(img0.shape[0], img0.shape[1])

    if L>PATCH_SIZE:

        # Calculate random crop parameters
        start_x = np.random.randint(0, L-PATCH_SIZE)
        start_y = np.random.randint(0, L-PATCH_SIZE)
        end_x = start_x + PATCH_SIZE
        end_y = start_y + PATCH_SIZE

        # Apply crop to image pair
        img0 = img0[start_x:end_x, start_y:end_y]
        img1 = img1[start_x:end_x, start_y:end_y]

    return img0, img1

def save_images(imgs, img_idx, aug_idx, PATH):

    for i in range(len(imgs)):
        img = Image.fromarray(imgs[i])
        img.save(PATH + str(img_idx) + '_' + str(aug_idx) + '_' + str(i) + '.png')

    print("Images saved in", PATH)

def get_image_data(PATH, n=-1, bw=False):
    imgs_0, imgs_1 =[], []
    file_names = os.listdir(PATH)
    if n<0: n=len(file_names)
    for i in range(n):
        fn = file_names[i]
        if fn.endswith("_gtf.png"):
            idx_str = fn.split('_')[0]
            if os.path.isfile(PATH + "/" + idx_str + ".png"):
                imgs_0.append(cv2.imread(PATH + "/" + idx_str + ".png"))
                if bw:
                    gray_values = np.mean(imgs_0[-1], axis=2, keepdims=True)
                    imgs_0[-1] = np.repeat(gray_values, 3, axis=2).astype(np.uint8)
                imgs_1.append(cv2.imread(PATH + "/" + idx_str + "_gtf.png"))

    return imgs_0, imgs_1

def import_const_img(fname):
    # trunk
    img = cv2.imread(fname)
    H, W, _ = img.shape
    output = np.full((H,W), -1, dtype=np.int32)
    for c in range(0,50):
        output[img[:,:,0]==c] = c
    return output

class ImageGrid():

    def __init__(self, src_imgs, tgt_imgs, folder_path, filename):
        self.H = src_imgs[0].shape[0]
        self.W = src_imgs[0].shape[1]
        self.C = src_imgs[0].shape[2]
        self.n = len(src_imgs)
        output_shape = (self.H*self.n, int(self.W * 2.2), self.C) #2.2 instead of 2.0 to provide some space to the right of target image
        self.output = np.zeros(output_shape, dtype=np.uint8)
        self.out_path = folder_path + '//' + filename + '.png'

        #make first columns
        for i in range(len(src_imgs)):
            src_imgs[i] = cv2.resize(src_imgs[i], (self.H, self.W), interpolation=cv2.INTER_CUBIC)
            tgt_imgs[i] = cv2.resize(tgt_imgs[i], (self.H, self.W), interpolation=cv2.INTER_CUBIC)
        src_img_col = np.vstack(src_imgs) #stack to column
        tgt_img_col = np.vstack(tgt_imgs) #stack to column
        if len(tgt_img_col.shape)==2 or tgt_img_col.shape[2]!=3:
            tgt_img_col = cv2.cvtColor(tgt_img_col,cv2.COLOR_GRAY2RGB) #grayscale to color
        self.output = np.hstack((src_img_col,tgt_img_col))

        #add space
        self.add_vertical_line(width=int(0.2*self.W), color=False)

        #save
        self.save()


    def save(self):
        print('Saving', self.out_path)
        cv2.imwrite(self.out_path, self.output)

    def add_column(self, imgs):
        for i in range(len(imgs)):
            imgs[i] = cv2.resize(imgs[i], (self.H, self.W), interpolation=cv2.INTER_CUBIC)
        img_col = np.vstack(imgs) #stack to column
        img_col = cv2.cvtColor(img_col,cv2.COLOR_GRAY2RGB) #grayscale to color
        self.output = np.hstack((self.output,img_col))
        #save
        self.save()

    def add_vertical_line(self, width=5, color=True):
        vline = np.zeros((self.n*self.H, width, self.C))+255
        if color: vline[:,:,1:] = 0
        self.output = np.hstack((self.output,vline))
        self.save()

def get_ranges(params, out_img_coords, ABs, ARF_TYPE):
    all_heights = []
    all_angles = []
    all_radiuses = []
    for j,px_coords in enumerate(out_img_coords):
        A,B = ABs[j]
        px_coords = px_coords.view(-1,3)
        heights, angles, radiuses = procedural_wood_function_for_initialization(params, px_coords, A=A, B=B, return_reshaped=True, arf_type=ARF_TYPE, return_cylindrical_coords=True)
        all_heights.extend(heights.numpy().tolist())
        all_angles.extend(angles.numpy().tolist())
        all_radiuses.extend(radiuses.numpy().tolist())

    # get height range and ring range
    height_range = [min(all_heights), max(all_heights)]
    ring_range = [min(all_radiuses), max(all_radiuses)]

    # get spoke range
    # sort angles and get rough middle
    all_angles = np.array(all_angles)
    sorted_angles = np.sort(all_angles)
    a_middele_angle = sorted_angles[int(0.5*sorted_angles.shape[0])]

    # offset angles by 180 degrees from center of rough middle
    offset = (a_middele_angle + torch.pi) % (2*torch.pi)
    all_angles = np.mod(all_angles + offset, 2*torch.pi)

    # get refined offset
    sorted_angles = np.sort(all_angles)
    a_middele_angle_2 = sorted_angles[int(0.5*sorted_angles.shape[0])]
    refined_offset = (a_middele_angle_2 - a_middele_angle - offset + 2*torch.pi) % (2*torch.pi)

    # apply refined offset
    offset = (offset + refined_offset + 2*torch.pi) % (2*torch.pi)
    all_angles = np.mod(all_angles + refined_offset + 2*torch.pi, 2*torch.pi)

    # calcualte min/max
    spoke_min = all_angles.min()
    spoke_max = all_angles.max()

    spoke_range = [spoke_min, spoke_max, offset] #radians
    
    return height_range, spoke_range, ring_range
