import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg
import math
import random


from COMMON.procedural_wood_function import *

torch.pi = torch.acos(torch.zeros(1)).item() * 2
torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(True)


def get_cube_side_images(IN_PATH, H=-1, W=-1, unet_output=True, return_annotations=False, display=False):

    rgb_imgs = []
    arl_imgs = []
    ann_imgs = []

    ltrs = ['A', 'B', 'C', 'D', 'E', 'F']
    
    for ltr in ltrs:

        #rgb image
        file_path = IN_PATH + ltr + '_col.png'
        img = cv2.imread(file_path)
        if H>0 and W>0: img = cv2.resize(img, (H, W), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rgb_imgs.append(img)

        #arf image
        if unet_output: file_path = IN_PATH + ltr + '_arl-unet.png'
        else: file_path = IN_PATH + ltr + '_arl.png'
        img = cv2.imread(file_path)
        if H>0 and W>0: img = cv2.resize(img, (H, W), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        arl_imgs.append(img)

        #annotation lines image
        if return_annotations:
            file_path = IN_PATH + ltr + '_ann.png'
            img = cv2.imread(file_path)
            if H>0 and W>0: img = cv2.resize(img, (H, W), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ann_imgs.append(img)

    return rgb_imgs, arl_imgs, ann_imgs
    

def generate_cuboid_coordinates(W, H, D):

    max_hwd = max(max(W, H),D)

    #AF
    px_pos_A = xyz_coords(W,H,D,max_hwd,2)
    px_pos_A[:, :, 1] = -px_pos_A[:, :, 1] 
    px_pos_A[:, :, 2] = -px_pos_A[:, :, 2] 
    px_pos_F = px_pos_A.clone()
    px_pos_F[:, :, 0] = -px_pos_F[:, :, 0] 
    px_pos_F[:, :, 2] = -px_pos_F[:, :, 2] 

    #BD
    px_pos_B = xyz_coords(W,D,H,max_hwd,1)
    px_pos_B[:, :, 1] = -px_pos_B[:, :, 1] 
    px_pos_D = px_pos_B.clone()
    px_pos_D[:, :, 1] = -px_pos_D[:, :, 1] 
    px_pos_D[:, :, 2] = -px_pos_D[:, :, 2] 

    #CE
    px_pos_C = xyz_coords(D,H,W,max_hwd,0)
    px_pos_C[:, :, 1] = -px_pos_C[:, :, 1] 
    px_pos_E = px_pos_C.clone()    
    px_pos_E[:, :, 0] = -px_pos_E[:, :, 0]
    px_pos_E[:, :, 2] = -px_pos_E[:, :, 2] 

    return [px_pos_A, px_pos_B, px_pos_C, px_pos_D, px_pos_E, px_pos_F]


def generate_cuboid_coordinates_cuts(hwd):

    #cut 1
    px_pos_C = xyz_coords(hwd,hwd,hwd,hwd,0)
    px_pos_C[:, :, 1] = -px_pos_C[:, :, 1] 
    px_pos_E1 = px_pos_C.clone()    
    px_pos_E1[:, :, 0] = -px_pos_E1[:, :, 0]
    px_pos_E1[:, :, 2] = -px_pos_E1[:, :, 2]   
    # 
    px_pos_E1[:, :, 0] = px_pos_E1[:, :, 1]
    px_pos_E1[:, :, 0] += 0.5 
    px_pos_E1[:, :, 0] *= 0.575
    px_pos_E1[:, :, 0] -= 0.375 

    #cut 2
    px_pos_C = xyz_coords(hwd,hwd,hwd,hwd,0)
    px_pos_C[:, :, 1] = -px_pos_C[:, :, 1] 
    px_pos_E = px_pos_C.clone()    
    px_pos_E[:, :, 0] = -px_pos_E[:, :, 0]
    px_pos_E[:, :, 2] = -px_pos_E[:, :, 2] 
    # 
    px_pos_E[:, :, 0] = px_pos_E[:, :, 2]
    px_pos_E[:, :, 0] += 0.5 
    px_pos_E[:, :, 0] *= 0.575
    px_pos_E[:, :, 0] -= 0.375 
   
    return px_pos_E1, px_pos_E


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



def get_plot_image(loss_logs, loss_lbls, reg_log, best_i, best_loss, N, H=256, VL0s=[9999999], VL1s=[9999999], VL2s=[9999999], VL1=999999, VL2=999999, VL3=999999, display=False, ymax=-1.0, id=-1):

    plt.clf()
    plt.figure(figsize=(6,4))

    for i, loss_log in enumerate(loss_logs):

        lbl = loss_lbls[i]
        lw = 1.0
        if i==0: plt.plot(loss_log, label=lbl,  linewidth=2.0, color='k')
        else: plt.plot(loss_log, label=lbl,  linewidth=lw)

    for VL0 in VL0s:
        plt.axvline(x=VL0, color='blue', linestyle='dashed', linewidth=0.5)
    for vl11 in VL1s:
        plt.axvline(x=vl11, color='lime', linestyle='dashed', linewidth=0.5)
    for vl22 in VL2s:
        plt.axvline(x=vl22, color='orchid', linestyle='dashed', linewidth=0.5)

    plt.axvline(x=VL1, color='lime', linestyle='dashed')
    plt.axvline(x=VL2, color='orchid', linestyle='dashed')

    if len(reg_log)>0: plt.plot(reg_log, label='Reg. term',  linewidth=1.0)
    plt.scatter(best_i, float(best_loss), 50, marker='o', color='k')

    plt.tight_layout()
    plt.legend()
    plt.xlim([0,N-1])
    if ymax>0: plt.ylim([0.0, ymax])
    else: plt.ylim(bottom=0.0)

    plt_name = 'tmp_plt'
    if id>0: plt_name += str(id)

    #plt.savefig(plt_name + '.png')    
    #plt_img = cv2.imread(plt_name + '.png')
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()    
    width, height = canvas.get_width_height()
    argb_data = np.frombuffer(canvas.tostring_argb(), dtype='uint8').reshape((height, width, 4))
    plt_img = argb_data[:, :, 1:]  # Drop the alpha channel
    #plt_img = np.array(canvas.tostring_rgb(), dtype='uint8')
    #plt_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    plt_img = plt_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # Convert to (height, width, 3)
    ##
    plt_W = int((plt_img.shape[1])*(H/plt_img.shape[0]))
    plt_img = cv2.resize(plt_img, (plt_W,H))

    if display:
        cv2.imshow("plt", plt_img)
        cv2.waitKey(1)

    plt.close()

    return plt_img


def assemble_images(imgs, txts, map_imgs, map_txts, map_cmaps, H):

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
        cv2.putText(img, txt, (5,25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), thickness=2) # Outline (white, thicker)
        cv2.putText(img, txt, (5,25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), thickness=1) # Main text (black, normal)
        all_imgs.append(img)

    for i, (img, txt, mcmap) in enumerate(zip(map_imgs, map_txts, map_cmaps)):
        h = img.shape[0]
        w = img.shape[1]
        if h!=H:
            w = int(w*H/h)
            img = cv2.resize(img, (w, H), interpolation=cv2.INTER_CUBIC)
        if img.dtype == np.float32: img = cv2.cvtColor(floatImg_to_intImg(img),cv2.COLOR_GRAY2RGB)        
        img = cv2.applyColorMap(img, get_mpl_colormap(mcmap))
        cv2.putText(img, txt, (5,25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), thickness=2) # Outline (white, thicker)
        cv2.putText(img, txt, (5,25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), thickness=1) # Main text (black, normal)
        all_imgs.append(img)

    return np.hstack(all_imgs)


def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]

    return color_range.reshape(256, 1, 3)

def floatImg_to_intImg(img, scale_255=False, clip_on=True):
    if scale_255: img = (255.0*img)
    if clip_on: img = img.clip(0.0,255)
    img = img.astype(np.uint8)
    return img

def get_unfolded_image(imgs, black_bg=False): #false before, does it cause any problems?

    all_imgs = []

    for i in range(6):
        if i<len(imgs):
            img = np.copy(imgs[i])
            if img.dtype!=np.uint8:
                img = (255.0*img).astype(np.uint8)
            all_imgs.append(img)
        else:
            all_imgs.append(np.ones(imgs[0].shape, dtype=np.uint8)*255)

    depth = all_imgs[1].shape[0]
    shape_base = list(all_imgs[0].shape)
    shape_base[0] = depth
    shape_f  = tuple(shape_base)
    shape_ce = list(shape_base)
    shape_ce[1] = depth
    shape_ce = tuple(shape_ce)

    if black_bg: 
        empty_img_ce = np.zeros(shape_ce, dtype=np.uint8)
        empty_img_f =  np.zeros(shape_f, dtype=np.uint8)
        
    else: #white bg         
        empty_img_ce = np.ones(shape_ce, dtype=np.uint8)*255
        empty_img_f = np.ones(shape_f, dtype=np.uint8)*255
    
    col1 = np.vstack(((empty_img_ce, all_imgs[2], empty_img_ce)))
    col2 = np.vstack(((all_imgs[1], all_imgs[0], all_imgs[3])))
    col3 = np.vstack(((empty_img_ce, all_imgs[4], empty_img_ce)))
    col4 = np.vstack(((empty_img_f, all_imgs[5], empty_img_f)))

    unfolded_img = np.hstack((col1,col2,col3,col4))

    return unfolded_img


def get_image_open_contours(img, dis_img, out_torch=False, THRESHOLD=200, MIN_LEN_COUNTOUR=32, display=False, set_max_contrast=True, save_img=False, index=0):

    H = img.shape[0]
    W = img.shape[1]

    # Make sure the input is a grayscale image
    if len(img.shape)==3 and img.shape[2]==3: # a 3 channel color image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    
    # Make sure the range is 0-255
    if gray_img.dtype==torch.float32 or gray_img.dtype==torch.float32:
        gray_img = (255*gray_img).type(torch.int16)

    # Get contours
    _, thresh = cv2.threshold(gray_img, THRESHOLD, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # Remove contour points on image border, and open contours
    trimmed_contours = []
    for contour in contours:

        new_contour = []
        first_segment = []
        first_segment_done = False

        edge_width = 5

        for px in contour:
            if px[0][0]<=edge_width or px[0][0]>=W-edge_width or px[0][1]<=edge_width or px[0][1]>=H-edge_width: # on or near edge
                if len(new_contour)>MIN_LEN_COUNTOUR: trimmed_contours.append(np.array(new_contour))
                new_contour = []
                first_segment_done=True
            if first_segment_done: new_contour.append(px)
            else: first_segment.append(px)

        new_contour.extend(first_segment)

        if len(new_contour)>MIN_LEN_COUNTOUR:
            trimmed_contours.append(np.array(new_contour))
    contours = trimmed_contours

    squeezed_contours = []
    for contour in contours:
        squeezed_contours.append(np.squeeze(np.array(contour), axis=1))
    contours = squeezed_contours
    
    # unsqueeze and datatype
    unsqueezed_contours = []
    for contour in contours:
        unsqueezed_contours.append(contour[:, np.newaxis, :].astype(np.int32))
    contours = unsqueezed_contours

    # Get contours on image for verification

    img_copy = gray_img.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
    img_copy = np.ones_like(img_copy) * 255
    for cont in contours:
        col = (255 * np.random.rand(3)).astype(np.uint8)
        col = tuple([int(x) for x in col])
        try:
            img_copy = cv2.drawContours(img_copy, cont, contourIdx=-1, color=col, thickness=2, lineType=cv2.LINE_AA)
        except:
            print("failed")
    if display:
        cv2.imshow('Contours', img_copy)
        cv2.waitKey(0)
    
    if save_img:
        white_image = np.ones_like(gray_img) * 255
        white_image = cv2.cvtColor(white_image, cv2.COLOR_GRAY2BGR)
        for cont in contours:
            col = (255 * np.random.rand(3)).astype(np.uint8)
            col = tuple([int(x) for x in col])
            try:
                white_image = cv2.drawContours(white_image, cont, contourIdx=-1, color=col, thickness=2, lineType=cv2.LINE_AA)
            except:
                print("failed")
        img_name = 'contour_img' + str(index) + '.png'
        cv2.imwrite(img_name, white_image)
        print('Saved', img_name)

    if out_torch:
        tcontours=[]
        for contour in contours:
            tcount = torch.tensor(contour)
            tcount = tcount.view(tcount.shape[0], 2) # shape [N,1,2] to [N,2]
            tcontours.append(tcount)
        contours = tcontours

    return contours, img_copy


def lift_contours_to_3D(contours_px, xyz):

    contours_3d = []

    for cont in contours_px:
        cont_3d = torch.zeros([len(cont),3])

        for i,px in enumerate(cont):
            pos = xyz[px[0],px[1]]
            cont_3d[i] = pos
        contours_3d.append(cont_3d)

    return contours_3d


def get_image_contours(img, out_torch=False, THRESHOLD=240, MIN_LEN_COUNTOUR=20, display=False):

    H = img.shape[0]
    W = img.shape[1]

    # Make sure the input is a grayscale image
    if len(img.shape)==3 and img.shape[2]==3: # a 3 channel color image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # Make sure the range is 0-255
    if gray_img.dtype==torch.float32 or gray_img.dtype==torch.float32:
        gray_img = (255*gray_img).type(torch.int16)

    # Get contours
    ret, thresh = cv2.threshold(gray_img, THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # Remove contour points on image border
    trimmed_contours = []
    for contour in contours:
        new_contour = []
        for px in contour:
            if px[0][0]==0 or px[0][0]==W: continue
            if px[0][1]==0 or px[0][1]==H: continue
            new_contour.append(px)
        trimmed_contours.append(np.array(new_contour))
    contours = trimmed_contours

    # Remove short segments
    long_contours = []
    for contour in contours:
        if len(contour)>=MIN_LEN_COUNTOUR:
            long_contours.append(contour)

    # Display contours on image for verification
    if display:
        img_copy = img.copy()
        cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.drawContours(image=img_copy, contours=long_contours, contourIdx=-1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow('None approximation', img_copy)
        cv2.waitKey(0)

    if out_torch:
        tcontours=[]
        for contour in long_contours:
            tcount = torch.tensor(contour)
            tcount = tcount.view(tcount.shape[0], 2) # shape [N,1,2] to [N,2]
            tcontours.append(tcount)
        long_contours = tcontours

    return long_contours

def get_closest_point_and_remove_from_array(point_array, point):

    distances = np.linalg.norm(point_array - point, axis=1)
    closest_index = np.argmin(distances)
    closest_point = point_array[closest_index]

    point_array = np.delete(point_array, closest_point, axis=0)

    return point_array, closest_point


def get_peak_centers_from_1d_gray_colormap(signal,params):

    threshold = signal.mean()
    print("threashold", threshold)
    above_threshold = signal >= threshold
    diff = np.diff(above_threshold.astype(int))  # Find the indices where the signal transitions from below to above the threshold and vice versa
    start_indices = np.where(diff == 1)[0] + 1 # Start of the peaks (where it goes from 0 to 1)
    end_indices = np.where(diff == -1)[0] + 1 # End of the peaks (where it goes from 1 to 0)
    if above_threshold[0]: # If the signal starts above the threshold, add the first index as a start
        start_indices = np.insert(start_indices, 0, 0)
    if above_threshold[-1]: # If the signal ends above the threshold, add the last index as an end
        end_indices = np.append(end_indices, len(signal))
    peak_centers = 0.5*(start_indices + end_indices) # Calculate the center of each peak (mean of start and end indices)

    #insert extra first and last values
    if peak_centers.shape[0]>=2:
        first_value = peak_centers[0] - (peak_centers[1] - peak_centers[0]) 
        if first_value>0: peak_centers = np.insert(peak_centers, 0, first_value)
        #insert two extra last values
        last_value = peak_centers[-1] + (peak_centers[-1] - peak_centers[-2])
        peak_centers = np.append(peak_centers, last_value)    
        median_peak_dist = (peak_centers[1:] - peak_centers[:-1]).mean()
        for i in range(20):
            #insert extra first values
            first_value = peak_centers[0] - median_peak_dist
            if first_value>0: peak_centers = np.insert(peak_centers, 0, first_value)
            #insert extra last values
            last_value = peak_centers[-1] + median_peak_dist
            peak_centers = np.append(peak_centers, last_value)

    #format
    peak_centers /= signal.shape[0]
    peak_centers = peak_centers * (params.ring_max - params.ring_min) + params.ring_min


    
    return peak_centers



def get_ranges(params, out_img_coords, dim):
    all_heights = []
    all_angles = []
    all_radiuses = []
    for j,px_coords in enumerate(out_img_coords):
        px_coords = px_coords.view(-1,3)
        heights, angles, radiuses = procedural_wood_function_for_initialization(params, px_coords, A=dim, B=dim, return_reshaped=True, return_cylindrical_coords=True)
        try:
            all_heights.extend(heights.numpy().tolist())
            all_angles.extend(angles.numpy().tolist())
            all_radiuses.extend(radiuses.numpy().tolist())
        except:
            all_heights.extend(heights.detach().numpy().tolist())
            all_angles.extend(angles.detach().numpy().tolist())
            all_radiuses.extend(radiuses.detach().numpy().tolist())

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



def get_image_data(PATH, n=-1, grayscale=False):
    src_imgs, tgt_imgs = [], []
    file_names = os.listdir(PATH)
    cnt = 0
    for fn in file_names:
        if fn.endswith("_arl.png"):
            cnt+=1
            if n>0 and cnt>n: break
            idx_str = fn.split('_')[0]
            fp = PATH + "/" + idx_str + ".png"
            if os.path.isfile(PATH + "/" + idx_str + ".png"):
                simg = cv2.imread(PATH + "/" + idx_str + ".png")
                if grayscale:
                    simg = np.mean(simg, axis=-1, keepdims=True)
                    simg = np.repeat(simg, 3, axis=-1)
                    simg = np.uint8(simg)
                src_imgs.append(simg)
                tgt_imgs.append(cv2.imread(PATH + "/" + idx_str + "_arl.png"))
    return src_imgs, tgt_imgs


class ImageGrid():

    def __init__(self, folder_path, filename, src_imgs, tgt_imgs, CONT=False):

        self.out_path = folder_path + '//' + filename + '.png'
        self.n = len(src_imgs)
        self.H = src_imgs[0].shape[0]
        self.W = src_imgs[0].shape[1]
        self.C = src_imgs[0].shape[2]

        if CONT:
            previous_grid_img = cv2.imread(self.out_path)
            self.output = previous_grid_img

        else: # new
            output_shape = (self.H*self.n, int(self.W * 2.2), self.C) #2.2 instead of 2.0 to provide some space to the right of target image
            self.output = np.zeros(output_shape, dtype=np.uint8)

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


    def save(self):
        print('Saving', self.out_path)
        cv2.imwrite(self.out_path, self.output)

    def add_column(self, imgs):
        for i in range(len(imgs)): imgs[i] = cv2.resize(imgs[i], (self.H, self.W), interpolation=cv2.INTER_CUBIC)
        img_col = np.vstack(imgs) #stack to column
        if len(img_col.shape) == 2: img_col = cv2.cvtColor(img_col, cv2.COLOR_GRAY2RGB)
        self.output = np.hstack((self.output,img_col))
        #save
        self.save()

    def replace_last_column(self, imgs):
        for i in range(len(imgs)):
            imgs[i] = cv2.resize(imgs[i], (self.H, self.W), interpolation=cv2.INTER_CUBIC)
        img_col = np.vstack(imgs) #stack to column
        img_col = cv2.cvtColor(img_col,cv2.COLOR_GRAY2RGB) #grayscale to color
        self.output[:, -img_col.shape[1]:] = img_col
        #save
        self.save()

    def add_vertical_line(self, width=5, color=True):
        vline = np.zeros((self.n*self.H, width, self.C))+255
        if color: vline[:,:,1:] = 0
        self.output = np.hstack((self.output,vline))
        self.save()

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

def random_scale(img0, img1, PATCH_SIZE, min_scale_ratio=0.9, max_scale_ratio = 1.1, proportional = True):


    if proportional:

        L = min(img0.shape[0], img0.shape[1])
        if L<PATCH_SIZE: print("Random scale error. Image is too small", img0.shape)
        min_size = max( PATCH_SIZE, int(L*min_scale_ratio) )
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
            v[v <= lim] = np.clip(v[v <= lim] + value, 0, 255)
        else:
            lim = np.absolute(value)
            v = v.astype(np.int16)  # Temporarily allow negative values
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        v = np.clip(v, 0, 255).astype(np.uint8)
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
            s[s <= lim] = np.clip(s[s <= lim] + value, 0, 255)
        else:
            lim = np.absolute(value)
            s = s.astype(np.int16) 
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        s = np.clip(s, 0, 255).astype(np.uint8)
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

def norm_torch_data_to_numpy_image(data):
    #img = np.uint8(data.cpu().detach().numpy().copy()*255.0)
    img = data.cpu().detach().numpy().copy() * 255.0
    if np.any(np.isnan(img)):
        img[np.isnan(img)] = 255.0
        print("nan")
    if np.any(np.isinf(img)):
        img[np.isinf(img)] = 0.0
        print("inf")
    img = img.clip(0.0,255.0)
    ## this could be sorce of image artifacts!!!### <----------------------------------------------------------------------------
    # Print unique values in the array
    #print("Unique values in img:", np.unique(img))
    img = np.uint8(img)
    img = img.transpose((1, 2, 0))
    img = np.clip(img, 0, 255)
    return img

def numpy_images_to_norm_torch_data(imgs, PATCH_SIZE, src=True):
    data = []
    for img in imgs:
        img_norm = numpy_image_to_norm_torch_data(img, PATCH_SIZE, src=src, to_torch=False, lst_out=False)
        data.append(img_norm)
    data = torch.tensor(np.array(data)).cuda()
    return data



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
