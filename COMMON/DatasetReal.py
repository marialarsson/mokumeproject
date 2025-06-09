import torch
import numpy as np
import sys
import random
import data_utils

class RealWoodData(torch.utils.data.Dataset):

    def __init__(self, src_imgs, tgt_imgs, PATCH_SIZE, TARGET_DATA_SIZE, noise=False, train=True):
        self.src_imgs = []
        self.tgt_imgs = []
        self.train = train
        self.PATCH_SIZE = PATCH_SIZE
        self.dataset_size = TARGET_DATA_SIZE
        self.noise = noise

        for img_idx in range(len(src_imgs)):
            src_img = src_imgs[img_idx]
            tgt_img = tgt_imgs[img_idx][:,:,0]

            src_img = np.array(src_img)
            tgt_img = np.array(tgt_img)

            #if not self.augmented:
            #    src_img, tgt_img = data_utils.random_scale(src_img, tgt_img, PATCH_SIZE, max_scale_ratio = 1.0, proportional = False)
            #    src_img, tgt_img = data_utils.random_crop( src_img, tgt_img, PATCH_SIZE)

            # for debugging
            #save_path =  'C:\\Users\\makal\\Dropbox\\UnetAnnualRingDetectionData_debugg\\'
            #data_utils.save_images([src_img, tgt_img], img_idx, i, save_path)

            self.src_imgs.append(src_img)
            self.tgt_imgs.append(tgt_img)

        print('Actual data set size.', 'Source images:', len(self.src_imgs), '- Target images:', len(self.tgt_imgs) )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        #src_img = self.src_imgs[idx % len(self.src_imgs)]
        #tgt_img = self.tgt_imgs[idx % len(self.tgt_imgs)]

        # randomly pick an image from your dataset
        random_idx = random.randint(0, len(self.src_imgs)-1)
        src_img = self.src_imgs[random_idx % len(self.src_imgs)]
        tgt_img = self.tgt_imgs[random_idx % len(self.tgt_imgs)]

        # perform image augmentation : flip, rotationm scale
        src_img, tgt_img = data_utils.random_flip(    src_img, tgt_img)
        src_img, tgt_img = data_utils.random_rotation(src_img, tgt_img)
        src_img, tgt_img = data_utils.random_scale(   src_img, tgt_img, self.PATCH_SIZE,
                                                        min_scale_ratio = 0.8, max_scale_ratio = 1.2, proportional = True)
        # for debugging
        #save_path =  'C:\\Users\\makal\\Dropbox\\UnetAnnualRingDetectionData_debugg\\'
        #data_utils.save_images([src_img, tgt_img], idx, 0, save_path)

        src_img, tgt_img = data_utils.random_crop(    src_img, tgt_img, self.PATCH_SIZE)

        if self.noise:
            #rand_prob = random.uniform(0.00,0.15);
            #src_img = data_utils.sp_noise(src_img,rand_prob)
            src_img = data_utils.colorjitter(src_img, cj_type_n=random.randint(0,2))
            src_img = data_utils.filters(src_img, f_type_n=random.randint(0,2), fsize=(2*random.randint(0,1)+1))
            #cv2.imshow("img", src_img) #for debugg
            #cv2.waitKey(0)         #for debugg

        src_data = data_utils.numpy_image_to_norm_torch_data(src_img, self.PATCH_SIZE, src=True)
        tgt_data = data_utils.numpy_image_to_norm_torch_data(tgt_img, self.PATCH_SIZE, src=False)

        return src_data, tgt_data

    def get_items(self, n, numpy_format=True):

        src_imgs = []
        tgt_imgs = []

        indexes = np.arange(len(self.src_imgs))
        np.random.shuffle(indexes)

        for i in range(n):
            idx = indexes[i%indexes.shape[0]]
            src_data, tgt_data = self.__getitem__(idx)
            if numpy_format:
                src_data = data_utils.norm_torch_data_to_numpy_image(src_data)
                tgt_data = data_utils.norm_torch_data_to_numpy_image(tgt_data)
            src_imgs.append(src_data)
            tgt_imgs.append(tgt_data)

        return src_imgs, tgt_imgs
