import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import vgg16, VGG16_Weights
import numpy as np
import sys
import cv2

sys.path.append("COMMON")
from procedural_wood_function import *

torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)
torch.pi = torch.acos(torch.zeros(1)).item() * 2



def iso_contour_loss(gt_contours_px, gt_contours_nrm, params, H, W, init_stage=True, show_knot=False):

    # loss based on assumption that points on the same trace should have a similar distance in the field
    cnt = 0
    loss = 0
    loss_img = np.zeros([H,W])

    for i in range(len(gt_contours_px)):

        cpts_px = gt_contours_px[i]
        cpts_nm = gt_contours_nrm[i]

        # Get edge--field differences
        if init_stage:
            contour_gt_values = procedural_wood_function_for_initialization(params, cpts_nm)
        else: # Refinement stage
            contour_gt_values = procedural_wood_function_for_refinement(params, cpts_nm, show_knot=show_knot)
        
        # Calculate deviations
        gt_deviations = torch.abs( contour_gt_values - torch.median(contour_gt_values) )

        # Add deviation to loss
        loss += torch.sum(gt_deviations)

        # Loss image
        mask = np.zeros([H,W])
        gtf_devs_for_img =  gt_deviations.detach().clone()
        gtf_devs_for_img = np.clip(gtf_devs_for_img, 0.0, 1.0)
        mask[cpts_px[:, 1], cpts_px[:, 0]] = gtf_devs_for_img
        loss_img += mask

        cnt += cpts_px.shape[0]

    if cnt>0: loss = loss/cnt

    # Loss image (make more visible)
    kernel = np.ones((3, 3), np.uint8)
    loss_img = cv2.dilate(loss_img, kernel, iterations=2)

    return loss, loss_img



class VGGStyleLoss(nn.Module):
    def __init__(self, layers=None):
        super(VGGStyleLoss, self).__init__()
        # Load the pre-trained VGG-19 model
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

        # Specify which layers to use for style loss (default is layers for style transfer)
        self.layers = layers if layers else [2, 5, 12, 19]  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 #21, 28
        self.vgg = nn.Sequential()
        
        for i, layer in enumerate(vgg):
            self.vgg.add_module(str(i), layer)
            if i == max(self.layers):
                break

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Define the normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def gram_matrix(self, feature):
        """Computes the normalized Gram matrix for a given feature map."""
        b, c, h, w = feature.size()  # Batch size, Channels, Height, Width
        feature = feature.view(b, c, -1)  # Flatten the spatial dimensions into one
        gram = torch.bmm(feature, feature.transpose(1, 2))  # Batch matrix multiplication
        return gram / (c * h * w)  # Normalize by the total number of elements in each feature map

    def get_features(self, x):
        """Extracts the features from the selected layers."""
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features

    def forward(self, input, target):
        input = input.to(torch.float32)
        target = target.to(torch.float32)

        # Add batch dimension if missing
        if len(input.shape) == 3:  # If shape is (256, 256, 3)
            input = input.unsqueeze(0)  # Add batch dimension at the start
            target = target.unsqueeze(0)

        # If the input is in the shape (batch_size, 256, 256, 3), permute it to (batch_size, 3, 256, 256)
        if input.shape[-1] == 3:
            input = input.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)

        # Normalize the input images to match VGG-19 pre-training
        input = (input - self.mean.to(input.device)) / self.std.to(input.device)
        target = (target - self.mean.to(target.device)) / self.std.to(target.device)

        # Compute feature maps for both images
        input_features = self.get_features(input)
        target_features = self.get_features(target)

        # Compute the Gram matrices for these feature maps
        input_grams = [self.gram_matrix(f) for f in input_features]
        target_grams = [self.gram_matrix(f) for f in target_features]

        # Compute the style loss as the MSE between the Gram matrices
        style_loss = 0
        for input_gram, target_gram in zip(input_grams, target_grams):
            style_loss += F.mse_loss(input_gram, target_gram)

        return style_loss/len(input_grams)

    def get_features(self, x):
        """Extract the features from the selected layers of the VGG model"""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if int(name) in self.layers:
                features.append(x)
        return features

    def gram_matrix(self, feature_map):
        """Compute the Gram matrix for a given feature map"""
        a, b, c, d = feature_map.size()
        features = feature_map.view(a * b, c * d)
        gram = torch.mm(features, features.t())
        return gram / (a * b * d * c)



#------------------------- Relaxed OT loss from Ehsan------------------
class RelaxedOTLoss(torch.nn.Module):
    """https://arxiv.org/abs/1904.12785"""
    def __init__(self, n_samples=1024):
        super().__init__()
        self.n_samples = n_samples
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        #self.vgg = torch_models.vgg16(weights=torch_models.VGG16_Weights.IMAGENET1K_V1).features
        #vgg = torch_models.vgg16(weights=torch_models.VGG16_Weights.IMAGENET1K_V1).features.to(device)
        vgg_layers = []
        for l in vgg:
            if isinstance(l, torch.nn.MaxPool2d):
                l = torch.nn.AvgPool2d(l.kernel_size, l.stride, l.padding, l.ceil_mode)
            vgg_layers.append(l)
        self.vgg = torch.nn.Sequential(*vgg_layers)


    def get_vgg_features(self, imgs):
        style_layers = [1, 6, 11, 18, 25]
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        x = (imgs - mean) / std
        b, c, h, w = x.shape
        features = [x.reshape(b, c, h * w)]
        #features = []
        for i, layer in enumerate(self.vgg[:max(style_layers) + 1]):
            x = layer(x)
            if i in style_layers:
                b, c, h, w = x.shape
                features.append(x.reshape(b, c, h * w))
        return features

    @staticmethod
    def pairwise_distances_cos(x, y):
        x_norm = torch.norm(x, dim=2, keepdim=True)  # (b, n, 1)
        y_t = y.transpose(1, 2)  # (b, c, m) (m may be different from n)
        y_norm = torch.norm(y_t, dim=1, keepdim=True)  # (b, 1, m)
        dist = 1. - torch.matmul(x, y_t) / (x_norm * y_norm + 1e-10)  # (b, n, m)
        return dist

    @staticmethod
    def style_loss(x, y):
        pairwise_distance = RelaxedOTLoss.pairwise_distances_cos(x, y)
        m1, m1_inds = pairwise_distance.min(1)
        m2, m2_inds = pairwise_distance.min(2)
        remd = torch.max(m1.mean(dim=1), m2.mean(dim=1))
        return remd

    @staticmethod
    def moment_loss(x, y):
        mu_x, mu_y = torch.mean(x, 1, keepdim=True), torch.mean(y, 1, keepdim=True)
        mu_diff = torch.abs(mu_x - mu_y).mean(dim=(1, 2))

        x_c, y_c = x - mu_x, y - mu_y
        x_cov = torch.matmul(x_c.transpose(1, 2), x_c) / (x.shape[1] - 1)
        y_cov = torch.matmul(y_c.transpose(1, 2), y_c) / (y.shape[1] - 1)

        cov_diff = torch.abs(x_cov - y_cov).mean(dim=(1, 2))
        return mu_diff + cov_diff

    def forward(self, generated_image, target_image):

        # my addition
        
        generated_image = generated_image.to(torch.float32)
        target_image = target_image.to(torch.float32)

        # Add batch dimension if missing
        if len(generated_image.shape) == 3:  # If shape is (256, 256, 3)
            generated_image = generated_image.unsqueeze(0)  # Add batch dimension at the start
            target_image = target_image.unsqueeze(0)

        # If the input is in the shape (batch_size, 256, 256, 3), permute it to (batch_size, 3, 256, 256)
        if generated_image.shape[-1] == 3:
            generated_image = generated_image.permute(0, 3, 1, 2)
            target_image = target_image.permute(0, 3, 1, 2)
        
        # my addition until here
        
        loss = 0.0
        generated_features = self.get_vgg_features(generated_image)
        with torch.no_grad():
            target_features = self.get_vgg_features(target_image)
        # Iterate over the VGG layers
        for x, y in zip(generated_features, target_features):
            (b_x, c, n_x), (b_y, _, n_y) = x.shape, y.shape
            assert ((b_x == b_y) or (b_y == 1))
            n_samples = min(n_x, n_y, self.n_samples)
            indices_x = torch.argsort(torch.rand(b_x, 1, n_x, device=x.device), dim=-1)[..., :n_samples]
            x = x.gather(-1, indices_x.expand(b_x, c, n_samples))
            indices_y = torch.argsort(torch.rand(b_y, 1, n_y, device=y.device), dim=-1)[..., :n_samples]
            y = y.gather(-1, indices_y.expand(b_y, c, n_samples))
            x, y = x.transpose(1, 2), y.transpose(1, 2)  # (b, n_samples, c)
            loss += self.style_loss(x, y) + self.moment_loss(x, y)

        return loss.mean()


def image_loss(output, target):

    diff_img = torch.abs(output-target)
    loss = diff_img.mean()
    loss_img = diff_img.clone().detach().numpy()

    # If the loss image is color, convert it to grayscale
    if loss_img.ndim == 3:  # Color image
        loss_img = np.mean(loss_img, axis=2)  # Convert to grayscale

    return loss, loss_img

