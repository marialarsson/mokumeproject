import torch
import numpy as np
import cv2

torch.autograd.set_detect_anomaly(True)
torch.pi = torch.acos(torch.zeros(1)).item() * 2
torch.set_default_dtype(torch.float64)

def get_unit_vector(theta, phi):

    # Generate random angles in spherical coordinates
    #theta  # Azimuthal angle (0 to 2*pi)
    #phi # Polar angle (0 to pi)

    # Convert spherical coordinates to Cartesian coordinates (x, y, z)
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    # Create a 3D unit vector tensor
    unit_vector = torch.tensor([x, y, z])

    return unit_vector

def get_unit_vector_uniform(costheta,phi):

    #costheta: -1 to 1
    #phi: 0 to 2pi

    theta = torch.arccos( costheta )
    x = torch.sin( theta) * torch.cos( phi )
    y = torch.sin( theta) * torch.sin( phi )
    z = torch.cos( theta )

    return torch.tensor([x, y, z])

def closest_points_on_ray(origin, direction, points):
    vector_to_points = points - origin                                      # Calculate the vector from the origin to each point
    projection_lengths = torch.sum(vector_to_points * direction, dim=1)     # Calculate the projection of the vector onto the ray direction
    closest_points = origin + projection_lengths.view(-1, 1) * direction    # Calculate the closest points on the ray
    return closest_points



class ProceduralParameters:

    def __init__(self):
        self.pith_origin = torch.tensor([0.000001, -0.00001, 0.00001])
        self.pith_direction = torch.tensor([-0.0000001, 1.0, 0.000001])
        self.update_ref_vec()
    
    def update_init_pith_parameters(self, X):
        self.pith_origin = X[:3] # origin
        temp_pith_dir = X[3:6]
        self.pith_direction = temp_pith_dir / ( temp_pith_dir.norm() + 1e-8 )
        self.update_ref_vec()
    
    def center_origin(self):
        self.pith_origin = closest_points_on_ray(self.pith_origin, self.pith_direction, torch.tensor([[0.0,0.0,0.0]]))[0]

    def update_ref_vec(self):
        yvec = torch.tensor([1.000001, 0.0000001, 0.000001])
        ref_vec = torch.linalg.cross(yvec,self.pith_direction)
        self.ref_vec = ref_vec/torch.norm(ref_vec)

    ### DEFORMATIONS and RINGS ###

    def init_refined_procedual_parameters(self, height_num, height_range, spoke_num, spoke_range, ring_num, ring_range, large_sample=False):

        self.height_num = height_num
        height_step = height_range[1]-height_range[0]
        self.height_min = height_range[0] - height_step - 0.1 #added padding
        self.height_max = height_range[1] + height_step + 0.1 #added padding
        self.height_step = (self.height_max-self.height_min)/(height_num-1)
        self.height_levels = self.height_step * torch.arange(self.height_num, dtype=torch.float64) + self.height_min
        
        self.spoke_num = spoke_num
        spoke_step = (spoke_range[1]-spoke_range[0])/(spoke_num-1)
        self.spoke_min = max(spoke_range[0] - spoke_step - torch.pi/32, 0) #added padding
        self.spoke_max = min(spoke_range[1] + spoke_step + torch.pi/32, 2*torch.pi) #added padding
        self.spoke_offset = spoke_range[2]
        self.spoke_step = (self.spoke_max-self.spoke_min)/(spoke_num-1)
        self.spoke_angs = self.spoke_step * torch.arange(self.spoke_num, dtype=torch.float64) + self.spoke_min
        self.spoke_angs = torch.remainder(self.spoke_angs, 2*torch.pi)

        self.ring_num = ring_num
        #self.ring_step = (ring_range[1]-ring_range[0])/(ring_num-1)
        #print("ring num", self.ring_num)
        #print("ring range", ring_range, ring_range[1]-ring_range[0])
        self.ring_min = max(ring_range[0],0) #added padding
        self.ring_max = ring_range[1] #+ 0.25 #added padding
        #print("ring min", self.ring_min)
        #print("ring max", self.ring_max)
        self.ring_step = (self.ring_max-self.ring_min)/(ring_num-1)
        #print("ring step", self.ring_step)
        
    def update_spoke_rads(self, X):
        self.deformations = X
        rates_of_change = (1.0 + X)*self.ring_step
        ring_rads =  torch.cumsum(rates_of_change, dim=-1)
        zeros_tensor = torch.zeros(rates_of_change.size(0), rates_of_change.size(1), 1)
        ring_rads = torch.cat([zeros_tensor, ring_rads], dim=-1)
        self.ring_rads = ring_rads + self.ring_min
    
    def update_spoke_rads_explicitly(self, X):
        self.deformations = X
        shape = X.size()
        base_rads = self.ring_min + torch.arange(shape[2]).repeat(shape[0], shape[1], 1)*self.ring_step
        #base_rads = self.ring_min + torch.arange(X.size()[0])*self.ring_step
        self.ring_rads = base_rads + 0.5*X*self.ring_step        

    def update_ring_distances(self, X):
        self.ring_dists = X

    def update_median_ring_dist(self):
        self.median_ring_dist = (self.ring_dists[1:] - self.ring_dists[:-1]).median()

    ### INTENSITY FIELD (ARF) GREY-SCALE COLORS ###

    def update_average_arf_color(self, col):
        self.average_arf_col = col
    
    def update_base_arf_color_bar(self, length):
        self.base_arf_color_bar = self.average_arf_col.repeat(length)

    def update_arf_color_bar(self, X):
        self.arf_color_bar = self.base_arf_color_bar + X

    ### COLORS - COLOR BAR ###
    def update_base_color_bar(self, X):
        self.base_color_bar = X
    
    def update_color_bar(self, color_bar, side_cols, col_bar_weight=1.0):
        self.color_bar = self.base_color_bar + col_bar_weight * color_bar
        self.side_cols = 0.1*side_cols

    ### COLORS - PROCEDURAL ###

    def get_init_early_late_wood_colors(self, imgs):
        K = 2
        early_wood_cols = np.zeros((len(imgs),3)).astype(int)
        late_wood_cols = np.zeros((len(imgs),3)).astype(int)
        average_cols = np.zeros((len(imgs),3)).astype(int)
        for i,img in enumerate(imgs):
            Z = img.reshape((-1,3))
            Z = np.float32(Z) # convert to np.float32
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) # define criteria, number of clusters(K) and apply kmeans()
            ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center) # Now convert back into uint8, and make original image
            res = center[label.flatten()]     
            # Sort centers by their brightness (luminance) to determine lighter and darker colors
            luminance = np.sum(center, axis=1)  # Calculate brightness as sum of R, G, B values
            sorted_idx = np.argsort(luminance)
            # Store the lighter and darker colors in the corresponding arrays
            early_wood_cols[i] = center[sorted_idx[1]]  # Lighter color (late wood)
            late_wood_cols[i] = center[sorted_idx[0]]   # Darker color (early wood)
            average_cols[i] = Z.mean(axis=0)
            #print(Z.shape, average_cols[i])

        # Compute the average of the lighter and darker colors across all images
        lighter_col = np.mean(early_wood_cols, axis=0).astype(int)
        darker_col  = np.mean(late_wood_cols, axis=0).astype(int)
        average_col  = np.mean(average_cols, axis=0).astype(int)

        #self.base_early_wood_col = torch.tensor(lighter_col, dtype=torch.float32) / 255.0
        #self.base_late_wood_col = torch.tensor(darker_col, dtype=torch.float32) / 255.0
        self.base_early_wood_col = torch.tensor(average_col, dtype=torch.float32) / 255.0
        self.base_late_wood_col = torch.tensor(average_col, dtype=torch.float32) / 255.0

    def update_annual_ring_colors(self, ew_lw_cols, ew_lw_side_cols, ew_lw_col_bars, lw_end_start):

    
        self.early_wood_color = self.base_early_wood_col + 0.5*ew_lw_cols[:, 0]
        self.late_wood_color =  self.base_late_wood_col + 0.5*ew_lw_cols[:, 1]

        self.color_sides_earlywood = 0.2*ew_lw_side_cols[:, :, 0]
        self.color_sides_latewood = 0.2*ew_lw_side_cols[:, :, 1]

        self.color_bar_earlywood = 0.5*ew_lw_col_bars[:, :, 0]
        self.color_bar_latewood = 0.5*ew_lw_col_bars[:, :, 1]
        
        self.late_wood_end = torch.clamp(0.2 + 0.1*lw_end_start[0], min=0.1,max=0.25)
        self.late_wood_start = torch.clamp(0.2 + 0.1*lw_end_start[1], min=0.1,max=0.25)

        self.late_wood_end_smooth_linear_ratio = 0.0
        self.late_wood_start_smooth_linear_ratio = 0.0
    
    def update_base_detailed_annual_ring_colors(self):
        self.base_early_wood_color = self.early_wood_color.clone()
        self.base_late_wood_color = self.late_wood_color.clone()

        self.base_late_wood_end = self.late_wood_end.clone()
        self.base_late_wood_start = self.late_wood_start.clone()

        self.base_color_sides_earlywood = self.color_sides_earlywood.clone()
        self.base_color_sides_latewood = self.color_sides_latewood.clone()

        self.base_color_bar_earlywood = self.color_bar_earlywood.clone()
        self.base_color_bar_latewood = self.color_bar_latewood.clone()

    def update_detailed_annual_ring_colors(self, extra_ew_lw_cols, extra_lw_end_start, lw_end_start_linear):

        self.early_wood_color = self.base_early_wood_color + 0.1*extra_ew_lw_cols[:, 0] #0.02
        self.late_wood_color =  self.base_late_wood_color  + 0.1*extra_ew_lw_cols[:, 1] #0.05

        self.late_wood_end =   torch.clamp(self.base_late_wood_start + 0.2*extra_lw_end_start[0], min=0.01,max=0.25)
        self.late_wood_start = torch.clamp(self.base_late_wood_start + 0.2*extra_lw_end_start[1], min=0.01,max=0.25)

        self.late_wood_end_smooth_linear_ratio =   0.5 + 0.2 * lw_end_start_linear[0]
        self.late_wood_start_smooth_linear_ratio = 0.5 + 0.2 * lw_end_start_linear[1]


    ### FIBERS ###

    def init_fiber_parameters(self):
        self.fiber_shadow_strength = 0.03
        self.fiber_colblend_strength = 0.2
        self.fiber_size = 0.001
        self.fiber_lw_strength = 0.2

    def update_continous_fiber_parameters(self,X):
        self.fiber_shadow_strength = torch.clamp(0.03 + 0.01*X[0], min=0.0, max=0.05)
        self.fiber_colblend_strength = torch.clamp(0.2 + 0.10*X[1], min=0.0, max=0.30)
        self.fiber_lw_strength = torch.clamp(0.2 + X[1], min=0.0, max=1.00)
        #print("continous fiber parameters", self.fiber_shadow_strength, self.fiber_colblend_strength)

    def update_discontinous_fiber_parameters(self,X):
        self.fiber_size = torch.clamp(0.0001 + 0.005*torch.pow(X,2), min=0.0, max=0.02)
        #print("Updating discontinous fiber parameters to", self.fiber_size)

    ### KNOT ###

    def init_knot_parameters(self, org, dir):
        self.knot_origin = org
        self.knot_direction = dir
        self.base_knot_origin = org
        self.base_knot_direction = dir
        self.knot_color = torch.zeros(3) + 0.5
        self.knot_density = torch.tensor(7.0)
        self.knot_smoothness = torch.tensor(2.0)

    def update_knot_deform_parameters(self, Y):
        self.knot_deformations = Y
        self.knot_density = 10.0 + 20.0*Y[0]
        self.base_knot_density = self.knot_density.clone()
        self.knot_smoothness = 2.0 + 4.0*Y[1]
        self.knot_origin = self.base_knot_origin + 0.01*Y[2:5]
        self.knot_direction = self.base_knot_direction + 0.01*Y[5:9]
        #self.knot_direction = self.knot_direction/torch.norm(knot_dir)
        
        #print(self.knot_density.item(), self.knot_smoothness.item())
    
    def update_knot_colors(self, color_bar, ani_fac):
        #self.knot_color = 0.5 + X[:3]
        #self.knot_shadow_color = 0.1*X[3:6]
        #self.knot_color_expansion = 0.1*X[6]
        #self.knot_color_smoothness = 0.2*X[7]
        #self.knot_density = self.base_knot_density + 5*X[0]
        self.knot_color_bar = 0.5 + color_bar
        self.knot_color_anisotrophy_factor = 0.5 + 5.0*ani_fac        
    
    def update_base_knot_colors(self):
        self.base_knot_color = self.knot_color
        self.base_knot_color_expansion = self.knot_color_expansion
        self.base_knot_color_smoothness = self.knot_color_smoothness

    def update_detailed_knot_colors(self, X):
        self.knot_color = self.base_knot_color + X[:3]
        self.knot_color_expansion = self.base_knot_color_expansion + 0.1*X[3]
        self.knot_color_smoothness = self.base_knot_color_smoothness + 0.1*X[4]
        
    def update_simple_colors(self,Z):
        self.knot_color_anisotrophy_factor = 0.5 + 10.0*Z[0]
        self.background_color = 0.6+0.5*Z[1:4]
        self.knot_color = 0.4+0.5*Z[4:]
         
    ### PORES ###

    def init_pore_parameters(self):
        self.pore_cell_dim_ad = 0.02
        self.pore_rad = 0.3*self.pore_cell_dim_ad
        self.base_pore_rad = 0.3*self.pore_cell_dim_ad
        self.pore_cell_dim_h = 0.4
        self.pore_color = torch.zeros(3) + 0.2
        self.pore_direction_strength = torch.tensor(0.5)
        self.pore_occurance_ratio = torch.tensor(0.8)
        self.pore_occ_ring_correlation = torch.tensor(0.3)
        self.pore_rad_scale_ring_correlation = torch.tensor(0.3)
        self.pore_latewood_occ_dist = torch.tensor(0.0)
        self.base_pore_color = self.pore_color.clone()
        self.base_pore_direction_strength = self.pore_direction_strength.clone()
        self.base_pore_occurance_ratio = self.pore_occurance_ratio.clone()
        self.base_pore_occ_ring_correlation = self.pore_occ_ring_correlation.clone()
        self.base_pore_rad_scale_ring_correlation = self.pore_rad_scale_ring_correlation.clone()
        self.base_pore_latewood_occ_dist = self.pore_latewood_occ_dist.clone()
        
    def update_discontinous_pore_parameters(self, X):
        #cell size
        self.pore_cell_dim_ad = torch.clamp(0.01 + 0.03*torch.pow((X[0]),2), 0.005, 0.10)
        self.pore_rad = 0.3*self.pore_cell_dim_ad
        self.base_pore_rad = self.pore_rad.clone()
        self.pore_cell_dim_h = torch.clamp(0.2 + X[1], 0.2, 2.0)
        #occurances
        self.pore_occurance_ratio = X[2]
        self.base_pore_occurance_ratio = self.pore_occurance_ratio.clone()
        self.pore_occ_ring_correlation = X[3]
        self.base_pore_occ_ring_correlation = self.pore_occ_ring_correlation.clone()
        self.pore_latewood_occ = 1.5*X[4]
        self.base_pore_latewood_occ = self.pore_latewood_occ.clone()
        #print("self.pore_occurance_ratio", self.pore_occurance_ratio)
        #print("self.pore_occ_ring_correlation", self.pore_occ_ring_correlation)
        #print("self.pore_latewood_occ", self.pore_latewood_occ)
        
    def update_continous_pore_parameters(self, X):
        self.pore_rad = torch.clamp(self.base_pore_rad * (1.0 + 0.1*X[0]), 0.001, 0.10)
        #print("pore_rad", self.pore_rad.requires_grad)
        self.pore_color = self.base_pore_color + 0.5*X[1:4]
        #print("pore_color", self.pore_color.requires_grad)
        self.pore_direction_strength = torch.clamp(self.base_pore_direction_strength + 0.2*X[4], 0.0, 0.75)
        #print("pore_direction_strength", self.pore_direction_strength.requires_grad)
        self.pore_occurance_ratio = self.base_pore_occurance_ratio + 0.5*X[5]
        self.pore_occ_ring_correlation = self.base_pore_occ_ring_correlation + 0.5*X[6]
        self.pore_rad_scale_ring_correlation = self.base_pore_rad_scale_ring_correlation + 0.5*X[7]
        self.pore_latewood_occ = self.base_pore_latewood_occ + 0.5*X[8]
    
    ### RAYS ###

    def init_ray_parameters(self, ring_porous=True):
        # ray width
        self.ray_cell_dim_a = torch.tensor(0.05)
        self.ray_width = 0.4*self.ray_cell_dim_a
        self.base_ray_width = self.ray_width.clone()
        # ray length
        self.ray_cell_dim_d = torch.tensor(0.20)
        self.ray_length = 0.2*self.ray_cell_dim_d
        self.base_ray_length = self.ray_length.clone()
        # ray height
        self.ray_cell_dim_h = torch.tensor(0.10)
        self.ray_height = 0.2*self.ray_cell_dim_h
        self.base_ray_height = self.ray_height.clone()
        # ray color
        self.ray_multiply_color = torch.zeros(3) + 0.2
        self.base_ray_multiply_color = self.ray_multiply_color.clone()
        self.ray_overlay_color = torch.zeros(3) + 0.6
        self.base_ray_overlay_color = self.ray_overlay_color.clone()
        self.ray_mult_over_balance = torch.tensor(0.2) # more multiply
        self.ray_base_mult_over_balance = self.ray_mult_over_balance.clone()
        # ray occurance
        self.ray_occurance_ratio = torch.tensor(0.7)
        self.base_ray_occurance_ratio = self.ray_occurance_ratio.clone()
        
    def update_discontinous_ray_parameters(self, X):
        # ray width
        self.ray_cell_dim_a = torch.clamp(0.005 + 0.008*torch.pow((X[0]),2), 0.001, 0.10)
        self.ray_width = 0.4*self.ray_cell_dim_a
        self.base_ray_width = self.ray_width.clone()
        #ray length
        self.ray_cell_dim_d = torch.clamp(self.ray_cell_dim_a * 10 * (1.0 + 5*X[1]), 0.01, 1.00)
        self.ray_length = 0.2*self.ray_cell_dim_d
        self.base_ray_length = self.ray_length.clone()
        #ray height
        self.ray_cell_dim_h = torch.clamp(self.ray_cell_dim_a * 3 * (1.0 + 5*X[2]), 0.01, 1.00)
        self.ray_height = 0.4*self.ray_cell_dim_h
        self.base_ray_height = self.ray_height.clone()
        # occ
        self.ray_occurance_ratio = 0.7 * X[3]
        self.base_ray_occurance_ratio = self.ray_occurance_ratio
        
    def update_continous_ray_parameters(self, X):
        self.ray_width = torch.clamp(self.base_ray_width * (1.0 + 1.0*X[0]), 0.001, 0.50)
        self.ray_length = torch.clamp(self.base_ray_length * (1.0 + 1.0*X[1]), 0.001, 0.50)
        self.ray_height = torch.clamp(self.base_ray_height * (1.0 + 1.0*X[2]), 0.001, 0.50)
        self.ray_occurance_ratio = self.base_ray_occurance_ratio + X[3]
        self.ray_multiply_color = torch.clamp(self.base_ray_multiply_color + 0.25*X[4:7], 0.0, 1.00)
        self.ray_overlay_color = torch.clamp(self.base_ray_overlay_color + 0.25*X[7:10], -1.00, 1.00)
        self.ray_mult_over_balance = torch.clamp(self.ray_base_mult_over_balance + 0.2*X[10], 0.0, 1.00)

    ### Other ###
    def get_color_palette(self, imgs):
        K = 2
        # get K colors for each image in the training data set
        cols = np.zeros((K*len(imgs),3)).astype(int)
        for i,img in enumerate(imgs):
            Z = img.reshape((-1,3))
            Z = np.float32(Z) # convert to np.float32
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) # define criteria, number of clusters(K) and apply kmeans()
            ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center) # Now convert back into uint8, and make original image
            res = center[label.flatten()]
            cols[K*i:K*i+K] = np.unique(res,axis=0)
        #sort colors
        row_sums = np.sum(cols, axis=1)
        sort_index = np.argsort(row_sums) # index sort in descending order
        cols = cols[sort_index, :]
        self.cols = cols
        #print(cols)


    ### Detaching before pickling ###
    def detach_tensors(self):
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                self.__dict__[attr_name] = attr_value.detach().clone()