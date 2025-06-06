import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg

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



def get_plot_image(loss_logs, loss_lbls, reg_log, best_i, best_loss, N, H=256, VL0s=[9999999], VL1s=[9999999], VL2s=[9999999], VL1=999999, VL2=999999, VL3=999999, display=False, simple=False, ymax=-1.0, id=-1):

    plt.clf()
    plt.figure(figsize=(6,4))

    for i, loss_log in enumerate(loss_logs):

        if simple and i>0: break
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

    if not simple: plt.plot(reg_log, label='Reg. term',  linewidth=1.0)
    plt.scatter(best_i, float(best_loss), 50, marker='o', color='k')

    plt.tight_layout()
    plt.legend()
    plt.xlim([0,N-1])
    if ymax>0: plt.ylim([0.0, ymax])

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


def get_unfolded_image(imgs, black_bg=False): #false before, does it cause any problems?

    all_imgs = []

    for i in range(6):
        if i<len(imgs):
            img = np.copy(imgs[i])
            if img.dtype!=np.uint8:
                img = (255.0*img).astype(np.uint8)
            all_imgs.append(imgs[i])
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


"""
def get_test_sample_names(species_names):
    test_sample_names = []
    BASE_PATH = os.getcwd().split('git')[0] + "Dropbox\\my_unet_data_
    for name in species_names:
        temp = []
        file_path = BASE_PATH + name + "\\test_data_sample_names.txt"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                temp = [line.strip() for line in file]
        test_sample_names.extend(temp)
    return test_sample_names


def get_species_names_and_counts(foldernames):

    species_names = []
    species_counts = []
    species_numbers = []
    full_samples_names = []

    for fon in foldernames:
        match = re.match(r"([A-Z]+)(\d+)", fon)
        if match:
            spe_name = match.group(1)
            spe_num  = match.group(2)
            if spe_name not in species_names:
                species_names.append(spe_name)
                species_numbers.append([spe_num])
                species_counts.append(1)
            else:
                si = species_names.index(spe_name)
                species_numbers[si].append(spe_num)
                species_counts[si] += 1
            full_samples_names.append(spe_name+spe_num)


    return species_names, species_numbers, species_counts, full_samples_names



def get_hwd(imgs, print_hwd=False):

    H,W = imgs[0].shape[:2]
    D =   imgs[1].shape[0]

    if print_hwd: print(H,W,D)

    return H, W, D


def get_block_side_contours(imgs, MIN_LEN_COUNTOUR, display=False):

    contours_all = []

    for i in range(6):
        img = imgs[i]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        long_contours = []
        for contour in contours:
            if len(contour)>=MIN_LEN_COUNTOUR:
                long_contours.append(contour)
        contours_all.append(long_contours)

        if display:
            img_copy = img.copy()
            cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.imshow('None approximation', img_copy)
            cv2.waitKey(0)

    return contours_all

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





def find_indices_of_closest_nonsame_pair(value_list):

    min_diff = 99999999
    sel_i = -1
    sel_j = -1

    for i in range(len(value_list)):

        for j in range(len(value_list)):

            if i==j: continue #itself
            if value_list[i]==value_list[j]: continue #same value

            diff = abs(value_list[i]-value_list[j]) # difference

            if diff<min_diff:
                min_diff=diff
                sel_i = i
                sel_j = j

    return sel_i, sel_j, min_diff


def ordered_contours_by_distance(count_pxs, dst_imgs, tgt_imgs, params, threshold):

    ages = []

    H,W = dst_imgs[0].shape

    dists = []
    for con_pxs, dst_img in zip(count_pxs, dst_imgs):
        if dst_img.requires_grad: dst_img = dst_img.detach()
        temp = []
        for cpx in con_pxs:
            dsts = dst_img[cpx[:, 1], cpx[:, 0]]
            dsts = dsts.numpy()
            temp.append(dsts.mean())
        dists.append(temp)

    flat_dists = np.concatenate(dists)
    continue_joining_IDs = True
    cnt = 0

    while continue_joining_IDs:

        cnt+=1

        # search for closest pair of distances
        sel_i, sel_j, min_d = find_indices_of_closest_nonsame_pair(flat_dists)
        if sel_i<0 or sel_j<0: break

        # is the distance within a treshhold?
        if min_d<threshold and cnt<200:

            # take the average value of all numbers with the same new ID - overvrite the distnace with that average distance for that ID
            grouped_inds = []
            grouped_dists = []
            for i in range(len(flat_dists)):
                if flat_dists[i] == flat_dists[sel_i] or flat_dists[i] == flat_dists[sel_j]:
                    grouped_dists.append(flat_dists[i])
                    grouped_inds.append(i)

            group_mean_dist = sum(grouped_dists)/len(grouped_dists)

            # overwrite group items with average distance
            for i in grouped_inds: flat_dists[i] = group_mean_dist

        else:
            if cnt>=200: print("Stopping at cnt", cnt)
            continue_joining_IDs = False

    unique_values, order_indices = np.unique(flat_dists, return_inverse=True)

    year_offset = int( flat_dists.min())

    id_list = []
    cnt = 0
    dists2 = []
    for i in range(len(dists)):
        temp = []
        temp2 = []
        for j in range(len(dists[i])):
            d = flat_dists[cnt]
            index = np.argmax(unique_values == d)
            temp.append(index+year_offset)
            temp2.append(d)
            cnt+=1
        id_list.append(temp)
        dists2.append(temp2)



    #yrs
    yr_list = id_list
    #for ids in id_list:
    #    temp = []
    #    for id in ids:
    #        temp.append(year_offset+1)
    #    yr_list.append(temp)


    # countour images
    #n_colors = 20
    #color_list = (255*cm.tab20(np.linspace(0, 1, n_colors))).astype(np.uint8)

    #n_colors = max(max(id_list))+2
    #color_list = (255*cm.cool(np.linspace(0, 1, n_colors))).astype(np.uint8)

    n_colors = 8
    color_list = (255*cm.viridis(np.linspace(0, 1, n_colors))).astype(np.uint8)

    cimgs = []
    for j in range(6):
        #img_copy = 255*np.ones([H,W,3]).astype(np.uint8)
        tgt_img = tgt_imgs[j]
        #if dst_img.requires_grad: dst_img = dst_img.detach()
        #img_dist = (255.0*img_dist.clone().numpy()).astype(np.uint8)
        tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_GRAY2BGR)

        for contour,id,yr,d in zip(count_pxs[j],id_list[j],yr_list[j],dists2[j]):
            contour = contour.numpy()
            col_id = id
            if col_id>n_colors-1: col_id = id%n_colors
            col = color_list[col_id]
            #c = int(255*d)
            #col = np.array([c,c,c]).astype(np.uint8)
            col = tuple([int(x) for x in col])
            cv2.polylines(tgt_img, [contour], isClosed=False, color=col, thickness=2)

            # add text
            txt = str(yr)
            font = cv2.FONT_HERSHEY_SIMPLEX # font
            org = contour[0] # org
            fontScale = 0.4 # fontScale
            color = (0, 0, 255)
            thickness = 1
            image = cv2.putText(tgt_img, txt, org, font, fontScale, color, thickness, cv2.LINE_AA)

        cimgs.append(tgt_img)




    return yr_list, cimgs


def get_gradient_magnitude_image(img, display=False):

    gradient_x, gradient_y = torch.gradient(img)
    gradient_magnitude_img = torch.norm(torch.stack([gradient_x, gradient_y], dim=0), p=2, dim=0)

    if display:

        img = gradient_magnitude_img.detach().numpy()
        img = image_utils.floatImg_to_intImg(img)
        img = cv2.applyColorMap(img, cv2.COLORMAP_COOL)
        cv2.imshow("Gradient magnitude image", img)
        cv2.waitKey(1)

    return gradient_magnitude_img

def min_filter(image, kernel_size):
    # Add batch and channel dimensions for convolution
    image = image.unsqueeze(0).unsqueeze(0)

    # Define a kernel with ones of size kernel_size
    kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=image.dtype).to(image.device)

    # Use convolution with 'valid' padding to compute the minimum value
    min_values = F.conv2d(image, kernel, padding=0, stride=1)

    # Remove batch and channel dimensions to get the result as [256, 256]
    min_values = min_values.squeeze(0).squeeze(0)

    return min_values

def max_filter(image, kernel_size):
    # Add batch and channel dimensions for convolution
    image = image.unsqueeze(0).unsqueeze(0)

    # Use max pooling operation with kernel_size
    max_values = F.max_pool2d(image, kernel_size, padding=(kernel_size//2), stride=1)

    # Remove batch and channel dimensions to get the result as [256, 256]
    max_values = max_values.squeeze(0).squeeze(0)

    return max_values

def gaussian_blur(input_image, kernel_size=5, sigma=1.0):


    #image = image.unsqueeze(0).unsqueeze(0)
    #gaussian_blur_filter = v2.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 2.0) )
    #blurred_image = gaussian_blur_filter(image)
    #blurred_img = F.gaussian_blur(image, kernel_size=(5,5), sigma=(0.1,2.0))
    #gaussian_blur_layer = T.GaussianBlur(kernel_size=(5, 5), sigma=(1.0, 1.0))
    #blurred_image = gaussian_blur_layer(image)

    padding = kernel_size // 2
    x = torch.arange(-padding, padding + 1, dtype=torch.float32)
    y = x.view(-1, 1)
    kernel = torch.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).to(input_image)

    blurred_image = F.conv2d(input_image.unsqueeze(0).unsqueeze(0), kernel, padding=padding)
    blurred_image = blurred_image.squeeze()

    return blurred_image


def plot_image(img, ax):

    ax.imshow(img)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def get_3D_image_pixel_positions(imgs, noramlized=False):

    xyzs = []
    IMG_SIZE = imgs[0].shape[0]

    for i in range(6):
        img = imgs[i]
        xy = np.indices(img[:,:,0].shape)
        z = np.expand_dims(xy[0,:], axis=0)
        xyz = np.vstack((xy, z))
        xyz[2] = 0
        if i==0: #A
            xyz[2] = IMG_SIZE-1 #zmax
            xyz[1] = np.flip(xyz[1], axis=1) # flip y
        if i==1: #B
            xyz[2] = IMG_SIZE-1 #zmax
            xyz[:2] = np.flip(xyz[:2], axis=1) # flip x and y
            xyz[1], xyz[2] = xyz[2].copy(), xyz[1].copy() # swap y and z
            xyz[2] = np.flip(xyz[2], axis=1) # flip z
        if i==2:
            xyz[:2] = np.flip(xyz[:2], axis=1) # flip x and y
            xyz[0], xyz[2] = xyz[2].copy(), xyz[0].copy()
            xyz[1], xyz[2] = xyz[2].copy(), -xyz[1].copy() # rotate 90 degrees around x-axis
        if i==3:
            xyz[0] = np.flip(xyz[0], axis=1) # flip x
            xyz[1], xyz[2] = xyz[2].copy(), xyz[1].copy() # swap y and z
            xyz[2] = np.flip(xyz[2], axis=1) # flip z
        if i==4:
            xyz[2] = IMG_SIZE-1 #zmax
            xyz[0], xyz[2] = xyz[2].copy(), xyz[0].copy() # swap x and z
            xyz[1], xyz[2] = xyz[2].copy(), -xyz[1].copy() # rotate 90 degrees around x-axis
            xyz[0] = np.flip(xyz[0], axis=1) # flip x
        if i==5:
            xyz[1] = np.flip(xyz[1], axis=1) # flip y

        if noramlized:
            xyz = xyz/IMG_SIZE-0.5
        xyzs.append(np.transpose(xyz))

    return xyzs

def plot_3D_images(imgs,xyzs,ax):

    IMG_SIZE = imgs[0].shape[0]

    # Create axis
    axes = [IMG_SIZE, IMG_SIZE, IMG_SIZE]

    # Create Data
    data = np.ones(axes, dtype=bool)
    data[1:-1, 1:-1, 1:-1] = False # Set non-edge voxels in the data array to False

    # Control Transparency
    alpha = 0.5

    # Initiate colors
    colors = np.zeros(axes + [4], dtype=np.float32)

    for i in range(6):

        #img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2RGBA)
        #img[3] = 0.2 #alpha channel
        b_ch, g_ch, r_ch = cv2.split(imgs[i])
        a_ch = np.ones(b_ch.shape, dtype=b_ch.dtype)*255
        img = cv2.merge((b_ch, g_ch, r_ch, a_ch))

        colors_array = img.reshape(-1,4)
        colors_array = colors_array / 255.0
        positions_array = xyzs[i].reshape(-1,3)

        # Assign colors to the corresponding positions
        colors[positions_array[..., 0], positions_array[..., 1], positions_array[..., 2]] = colors_array


    # Voxels is used to customizations of
    # the sizes, positions and colors.
    ax.voxels(data, facecolors=colors, edgecolors=None)


def get_3D_contours(contours_all, xyzs):

    contours_3d = []

    for i in range(6):
        xyz = xyzs[i]

        for cont in contours_all[i]:
            cpts = cont.squeeze()
            plt_pts = np.zeros((cpts.shape[0],3))

            for k in range(len(cpts)):
                pt = cpts[k]
                plt_pts[k] = xyz[pt[0],pt[1],:]
            contours_3d.append(plt_pts)

    return contours_3d

def plot_3D_contours(contours_3d, ax):

    for plt_pts in contours_3d:

        plt_pts = np.array(plt_pts).transpose()
        x = plt_pts[0].flatten()
        y = plt_pts[1].flatten()
        z = plt_pts[2].flatten()
        ax.plot(x, y, z, c='blue', linewidth=2)
        ax.scatter(x, y, z, c='blue', s=6.0, depthshade=False)





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
"""