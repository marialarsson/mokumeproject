import torch
import numpy as np
import math
import sys
import cv2
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt

sys.path.append("COMMON")
import data_utils
import image_utils

torch.autograd.set_detect_anomaly(True)
torch.pi = torch.acos(torch.zeros(1)).item() * 2
torch.set_default_dtype(torch.float32)


class DataInstance:
    def __init__(self, H, W, D, TARGET=False, OUTPUT=False, GT=False):

        self.TARGET = TARGET
        self.OUTPUT = OUTPUT
        self.GT = GT
        self.H = H
        self.W = W
        self.D = D

        # Initiate empty images
        HWs = [(H,W),(D,W),(H,D),(D,W),(H,D),(H,W)]
        self.rgb_imgs_torch = []
        self.rgb_imgs_np = []
        self.arf_imgs_torch = []
        self.arf_imgs_np = []
        self.gtf_imgs_np = []
        self.norm_gtf_np = []
        self.gtf_map_imgs_np = []
        self.gtf_field_imgs_np = []
        self.gtf_imgs_torch = []
        self.loss_imgs_np = []
        self.contour_imgs_np = []
        self.ann_imgs_np = []
        
        for i in range(6):

            black_img_1ch = torch.zeros(HWs[i], dtype=torch.float32)
            black_img_3ch = torch.zeros((*HWs[i], 3), dtype=torch.float32)

            self.rgb_imgs_torch.append(black_img_3ch)
            self.rgb_imgs_np.append(image_utils.floatImg_to_intImg(black_img_3ch.numpy(), scale_255=True))
            self.arf_imgs_torch.append(black_img_1ch)
            self.arf_imgs_np.append(image_utils.floatImg_to_intImg(black_img_1ch.numpy(), scale_255=True))
            self.gtf_imgs_torch.append(black_img_1ch)
            self.gtf_imgs_np.append(black_img_1ch.numpy())
            self.norm_gtf_np.append(black_img_1ch.numpy())
            self.gtf_map_imgs_np.append(black_img_3ch.numpy()+255)
            self.gtf_field_imgs_np.append(black_img_3ch.numpy()+255)
            self.contour_imgs_np.append(black_img_3ch.numpy()+255)
            self.loss_imgs_np.append(black_img_1ch.numpy())
            self.ann_imgs_np.append(black_img_1ch.numpy()+255)
            
        
        self.unfolded_rgb_img =  data_utils.get_unfolded_image(self.rgb_imgs_np)
        self.unfolded_arf_img =  data_utils.get_unfolded_image(self.arf_imgs_np)
        self.unfolded_gtf_img =  data_utils.get_unfolded_image(self.gtf_imgs_np)
        self.unfolded_loss_img = data_utils.get_unfolded_image(self.loss_imgs_np, black_bg=True)

    def update_rgb_imgs_from_numpy(self, imgs_np):

        # Numpy
        self.rgb_imgs_np = imgs_np
        self.unfolded_rgb_img = data_utils.get_unfolded_image(self.rgb_imgs_np)
        
        # torch
        for i,img_np in enumerate(imgs_np):
            img_torch = torch.tensor(img_np, dtype=torch.float32)
            img_torch = img_torch / 255.0
            self.rgb_imgs_torch[i] = img_torch

    def create_white_balanced_rgb_imgs(self):
        
        self.wb_rgb_imgs_np = []
        self.wb_rgb_imgs_torch = []
        self.channel_means_np = []

        for img in self.rgb_imgs_np:
            # get and store mean color
            channel_means = img.mean(axis=(0, 1))
            self.channel_means_np.append(channel_means)
            # white balance np img
            wb_img = img / (2.0 * channel_means)
            # torch
            img_torch = torch.tensor(wb_img, dtype=torch.float32)
            self.wb_rgb_imgs_torch.append(img_torch)
            # numpy
            wb_img = np.clip(255.0*wb_img,0,255).astype(np.uint8)
            self.wb_rgb_imgs_np.append(wb_img)
            
        self.unfolded_wb_rgb_img = data_utils.get_unfolded_image(self.wb_rgb_imgs_np)


    def update_rgb_imgs_from_torch(self, imgs_torch):

        #torch images
        self.rgb_imgs_torch = imgs_torch

        #numpy
        for i,img_torch in enumerate(imgs_torch):
            img_torch_copy = img_torch.clone()
            if img_torch_copy.requires_grad: img_torch_copy=img_torch_copy.detach()
            img_np = 255.0*img_torch_copy.numpy()
            img_np = img_np.astype(np.uint8)
            self.rgb_imgs_np[i] = img_np
        self.unfolded_rgb_img = data_utils.get_unfolded_image(self.rgb_imgs_np)

    def update_arf_imgs_from_numpy(self, imgs_np):
        for i,img_np in enumerate(imgs_np):

            #make sure its greyscale
            if len(img_np.shape)>2: img_np = img_np[:,:,0] 

            #make sure input image is 0-255
            if img_np.dtype != np.uint8: img_np = (img_np * 255.0).astype(np.uint8)

            #numpy range 0-255
            self.arf_imgs_np[i]=img_np
            
            #torch range 0.0-1.0           
            img_torch = torch.from_numpy(img_np).clone()
            img_torch = img_torch.type(torch.float32)
            img_torch = img_torch/255.0
            self.arf_imgs_torch[i]= img_torch
        
        self.unfolded_arf_img = data_utils.get_unfolded_image(self.arf_imgs_np)
    
    def update_arf_imgs_from_torch(self, imgs_torch):
        for i,img_torch in enumerate(imgs_torch):

            #torch
            self.arf_imgs_torch[i] = img_torch

            #numpy
            if img_torch.requires_grad: img_torch=img_torch.detach()
            self.arf_imgs_np[i] = image_utils.floatImg_to_intImg(img_torch.numpy(), scale_255=True)
        
        self.unfolded_arf_img = data_utils.get_unfolded_image(self.arf_imgs_np)

    def update_gtf_imgs_from_torch(self, imgs_torch):
        
        # add later: contours to the gtf

        for i,img_torch in enumerate(imgs_torch):

            #torch
            self.gtf_imgs_torch[i] = img_torch

            #numpy
            if img_torch.requires_grad: img_torch=img_torch.detach()
            self.gtf_imgs_np[i] = img_torch.numpy()
    
        self.unfolded_gtf_img = data_utils.get_unfolded_image(self.gtf_imgs_np)
    
    def update_iso_gtf_imgs_from_torch(self, imgs_torch):
        
        # add later: contours to the gtf

        self.iso_gtf_imgs_torch = []
        self.iso_gtf_imgs_np = []

        for i,img_torch in enumerate(imgs_torch):

            #torch
            self.iso_gtf_imgs_torch.append(img_torch)

            #numpy
            if img_torch.requires_grad: img_torch=img_torch.detach()
            self.iso_gtf_imgs_np.append(img_torch.numpy())
    
    def update_average_rgb_color(self):
        average_colors = [] 
        for img_np in self.rgb_imgs_np:
            average_colors.append(img_np.mean(axis=(0, 1)))
        self.average_rgb_color = np.array(average_colors).mean(axis=0)
        #print("Average rgb color", self.average_rgb_color)

    def update_average_wb_rgb_color(self):
        average_colors = [] 
        for img_np in self.wb_rgb_imgs_np:
            average_colors.append(img_np.mean(axis=(0, 1)))
        self.average_wb_rgb_color = np.array(average_colors).mean(axis=0)
        #print("Average rgb color", self.average_rgb_color)

    def unwhitebalance_rgb_imgs(self, channel_means_np):

        self.unwb_rgb_imgs_np = []

        for img,channel_means in zip(self.rgb_imgs_np,channel_means_np):
            # white balance np img
            unwb_img = (img/255.0) * 2.0 * channel_means
            unwb_img = np.clip(unwb_img,0,255).astype(np.uint8)
            self.unwb_rgb_imgs_np.append(unwb_img)
            
        self.unfolded_unwb_rgb_img = data_utils.get_unfolded_image(self.unwb_rgb_imgs_np)

        

    def update_average_arf_color(self):
        average_colors = [] 
        for img_np in self.arf_imgs_np:
            average_colors.append(img_np.mean())
        self.average_arf_color = np.array(average_colors).mean()
        #print("Average arf color", self.average_arf_color)

    def update_normalized_gtf(self):

        #find lowest/highest value (for normalization)
        min_value = np.min(np.stack(self.gtf_imgs_np))
        max_value = np.max(np.stack(self.gtf_imgs_np))

        for i,gtf_img_np in enumerate(self.gtf_imgs_np):

            # normalize gtf
            norm_gtf = gtf_img_np.copy()

            # normalize
            norm_gtf -= min_value
            norm_gtf /= max_value-min_value

            self.norm_gtf_np[i] = norm_gtf


    def update_gtf_field_imgs(self):

        for i,norm_gtf in enumerate(self.norm_gtf_np):

            # map image
            field_img_np = norm_gtf.copy()

            #change from float to integer image
            field_img_np = 255.0*field_img_np    
            field_img_np = field_img_np.astype(np.uint8)
            
            self.gtf_field_imgs_np[i] = field_img_np

        self.unfolded_gtf_field_img = data_utils.get_unfolded_image(self.gtf_field_imgs_np)

    def update_gtf_map_imgs(self, with_contours=True):
        #map

        for i,gtf_img_np in enumerate(self.gtf_imgs_np):

            # map image
            map_img_np = 255.0*gtf_img_np.copy()
            map_img_np = map_img_np.astype(np.uint8)
            #map_img_np = cv2.applyColorMap(map_img_np, image_utils.get_mpl_colormap('hsv'))
            col_map = cv2.COLORMAP_HSV
            map_img_np = cv2.applyColorMap(map_img_np, col_map)
            map_img_np = cv2.addWeighted(map_img_np, 0.5, np.ones_like(map_img_np) * 255, 0.5, 0)
            

            if with_contours:
                # contours
                nc = 0.2
                map_cont_img = gtf_img_np.copy()
                map_cont_img = map_cont_img%nc
                map_cont_img = (255.0*map_cont_img/nc).astype(np.uint8) 
                _, thresh = cv2.threshold(map_cont_img, 127, 255, cv2.THRESH_BINARY)
                #cv2.imshow("thresh", thresh)
                #cv2.waitKey(1)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                ## remove contours on edges of image
                min_length_contour = 1
                edge_width = 1
                width, height = map_cont_img.shape[:2]
                trimmed_contours = []
                for contour in contours:
                    new_contour = []
                    first_segment = []
                    first_segment_done = False
                    for px in contour:
                        if px[0][0] <= edge_width or px[0][0] >= width - edge_width or px[0][1] <= edge_width or px[0][1] >= height - edge_width:  # on or near edge
                            if len(new_contour) > min_length_contour:
                                trimmed_contours.append(np.array(new_contour))
                            new_contour = []
                            first_segment_done = True
                        if first_segment_done:
                            new_contour.append(px)
                        else:
                            first_segment.append(px)
                    new_contour.extend(first_segment)
                    if len(new_contour) > min_length_contour:
                        trimmed_contours.append(np.array(new_contour))
                ### end remove contours on edges of image
                contours = trimmed_contours
                
                # add contours to map image 
                for contour in contours: 
                    cv2.polylines(map_img_np, [contour], isClosed=False, color=(255, 255, 255), thickness=1)
            
            self.gtf_map_imgs_np[i] = map_img_np

        self.unfolded_gtf_map_img = data_utils.get_unfolded_image(self.gtf_map_imgs_np)

    def update_gtf_map_with_iso_curves_imgs(self):

        #map with iso-curves at ring-rads

        self.gtf_map_iso_imgs_np = []

        for gtf_img_np,iso_gtf_img_np in zip(self.gtf_imgs_np, self.iso_gtf_imgs_np):

            # map image
            map_img_np = 255.0*gtf_img_np.copy()
            map_img_np = map_img_np.astype(np.uint8)
            col_map = cv2.COLORMAP_HSV
            map_img_np = cv2.applyColorMap(map_img_np, col_map)
            map_img_np = cv2.addWeighted(map_img_np, 0.5, np.ones_like(map_img_np) * 255, 0.5, 0)


            iso_map_img_np = 255.0*iso_gtf_img_np.copy()
            iso_map_img_np = iso_map_img_np.astype(np.uint8)

            # contours
            nc = 2.0
            map_cont_img = iso_gtf_img_np.copy()
            map_cont_img = map_cont_img%nc
            map_cont_img = (255.0*map_cont_img/nc).astype(np.uint8) 
            #cv2.imshow("map", map_cont_img)
            #cv2.waitKey(0)
            _, thresh = cv2.threshold(map_cont_img, 127, 255, cv2.THRESH_BINARY)
            #cv2.imshow("thresh", thresh)
            #cv2.waitKey(1)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            ## remove contours on edges of image
            min_length_contour = 1
            edge_width = 1
            width, height = map_cont_img.shape[:2]
            trimmed_contours = []
            for contour in contours:
                new_contour = []
                first_segment = []
                first_segment_done = False
                for px in contour:
                    if px[0][0] <= edge_width or px[0][0] >= width - edge_width or px[0][1] <= edge_width or px[0][1] >= height - edge_width:  # on or near edge
                        if len(new_contour) > min_length_contour:
                            trimmed_contours.append(np.array(new_contour))
                        new_contour = []
                        first_segment_done = True
                    if first_segment_done:
                        new_contour.append(px)
                    else:
                        first_segment.append(px)
                new_contour.extend(first_segment)
                if len(new_contour) > min_length_contour:
                    trimmed_contours.append(np.array(new_contour))
            ### end remove contours on edges of image
            contours = trimmed_contours
            
            # add contours to map image 
            for contour in contours: 
                cv2.polylines(map_img_np, [contour], isClosed=False, color=(0, 0, 0), thickness=1)
            
            self.gtf_map_iso_imgs_np.append(map_img_np)
            cv2.imshow("img", map_img_np)
            cv2.waitKey(1)

        self.unfolded_gtf_map_iso_img = data_utils.get_unfolded_image(self.gtf_map_iso_imgs_np)


    def update_gtf_imgs_from_np(self, imgs_np):
        
        # add later: contours to the gtf

        for i,img_np in enumerate(imgs_np):

            self.gtf_imgs_np[i] = img_np

        self.unfolded_gtf_img = data_utils.get_unfolded_image(self.gtf_imgs_np)
    
    def update_loss_imgs_from_np(self, imgs_np, index=0):

        for i,img_np in enumerate(imgs_np):

            if len(img_np.shape)>2: img_np = img_np[:,:,0]
            if img_np.dtype != np.uint8: img_np = (img_np * 255.0).astype(np.uint8)
            self.loss_imgs_np[i]=img_np
        
        if index==0:   self.unfolded_loss_img = data_utils.get_unfolded_image(self.loss_imgs_np, black_bg=True)
        elif index==1: self.unfolded_loss_img1 = data_utils.get_unfolded_image(self.loss_imgs_np, black_bg=True)
        else: self.unfolded_loss_img2 = data_utils.get_unfolded_image(self.loss_imgs_np, black_bg=True)
    
    def update_ann_imgs_from_numpy(self, imgs_np):
        
        for i,img_np in enumerate(imgs_np):

            #make sure its greyscale
            if len(img_np.shape)>2: img_np = img_np[:,:,0] 

            #make sure input image is 0-255
            if img_np.dtype != np.uint8: img_np = (img_np * 255.0).astype(np.uint8)

            #numpy range 0-255
            self.ann_imgs_np[i]=img_np
            
            #torch range 0.0-1.0           
            #img_torch = torch.from_numpy(img_np).clone()
            #img_torch = img_torch.type(torch.float32)
            #img_torch = img_torch/255.0
            #self.ann_imgs_torch[i]= img_torch
        
        self.unfolded_ann_img = data_utils.get_unfolded_image(self.ann_imgs_np)

    def get_contours_from_annotations(self, xyzs, display=True):
        
        self.contour_pixels = []
        self.contour_positions = []

        for j,(img,display_img,xyz) in enumerate(zip(self.ann_imgs_np,self.rgb_imgs_np,xyzs)):

            # Find all unique pixel values excluding the background (255)
            unique_ids = np.unique(img)
            contour_ids = unique_ids[unique_ids < 200]

            # For each unique value, save the pixels of that value
            con_pxs = []
            for id in contour_ids:
                pxs = np.argwhere(img == id)
                pxs = pxs[:, [1, 0]]  # Swap columns to convert [y, x] -> [x, y]
                con_pxs.append(pxs)

            #visualize
            img_copy = display_img.copy()
            #img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            #img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
            img_copy = np.ones_like(img_copy) * 255
            for cont in con_pxs:
                col = (255 * np.random.rand(3)).astype(np.uint8)
                col = tuple([int(x) for x in col])                    
                try:
                    for px in cont:
                        img_copy = cv2.circle(img_copy, center=tuple(px), radius=1, color=col, thickness=-1)
                except:
                    print("failed")
            self.contour_imgs_np[j] = img_copy
            if display:
                cv2.imshow('Contours', img_copy)
                cv2.waitKey(1)
            
            # to torch
            torch_con_pxs = []
            for pxs in con_pxs:
                pxs = torch.tensor(pxs)
                torch_con_pxs.append(pxs)
            con_pxs = torch_con_pxs
            self.contour_pixels.append(con_pxs)

            con_pos = data_utils.lift_contours_to_3D(con_pxs,xyz)
            self.contour_positions.append(con_pos)

        self.unfolded_edge_img = data_utils.get_unfolded_image(self.contour_imgs_np)
    
                
    def get_gradient_magnitude_imgs(self):
        self.gradient_magnitude_imgs = []
        for img_torch in self.arf_imgs_torch:
            gm_img = data_utils.get_gradient_magnitude_image(img_torch)
            self.gradient_magnitude_imgs.append(gm_img)

    def get_contours(self, xyzs, save_cont_img=False):

        self.contour_pixels = []
        self.contour_positions = []
        
        for j,(img,display_img,xyz) in enumerate(zip(self.arf_imgs_np,self.rgb_imgs_np,xyzs)):

            con_pxs, con_img = data_utils.get_image_open_contours(img, display_img, out_torch=True, display=False, save_img=save_cont_img, index=j)  # Get ground truth image countours in 2D
            self.contour_pixels.append(con_pxs)
            self.contour_imgs_np[j]=con_img

            con_pos = data_utils.lift_contours_to_3D(con_pxs,xyz)
            self.contour_positions.append(con_pos)


        self.unfolded_edge_img = data_utils.get_unfolded_image(self.contour_imgs_np)

    def order_contours(self, dist_imgs, params, threshold):
        self.contour_ages, self.nedge_imgs = data_utils.ordered_contours_by_distance(self.contour_pixels, dist_imgs, self.imgs_np, params, threshold)
        self.unfolded_nedge_img = data_utils.get_unfolded_image(self.nedge_imgs)
