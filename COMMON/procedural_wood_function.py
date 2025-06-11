import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2
import sys

torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

def closest_angle(v1, v2):
    # Ensure the vectors are normalized
    v1 = v1 / v1.norm(dim=-1, keepdim=True)
    v2 = v2 / v2.norm(dim=-1, keepdim=True)
    
    # Calculate the dot product between the two vectors
    dot_product = torch.sum(v1 * v2, dim=-1)
    
    # Clip dot product to ensure values are within valid range for acos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle = torch.acos(dot_product)
    
    return angle

def generate_3d_noise(input_tensor):
    # Apply different trigonometric operations for each axis to create structured noise
    noise_x = torch.sin(input_tensor[:, 0] * 10.234 + input_tensor[:, 1] * 7.345 + input_tensor[:, 2] * 1222.44)  # Noise for x-axis
    noise_y = torch.cos(input_tensor[:, 0] * 122.312 + input_tensor[:, 1] * 8.453 + input_tensor[:, 2] * 55.355)  # Noise for y-axis
    noise_z = torch.sin(input_tensor[:, 0] * 5.234  + input_tensor[:, 1] * 15.1445 + input_tensor[:, 2] * 1.33440)  # Noise for z-axis
    
    # Combine the noise values into a 3D tensor (Nx3)
    noise = torch.stack((noise_x, noise_y, noise_z), dim=1)
    
    return noise


def power_min_smooth(A, B, k):
    A = A.to(torch.float32)
    if torch.isnan(A).any() or torch.isinf(A).any():
        print("NaNs or Infs found in tensor A.")
    if torch.any(A == 0):
        print("Negative or zero values in A for fractional or negative power.")
    A = torch.pow(A, k)
    B = torch.pow(B, k)
    return torch.pow((A*B)/(A+B), 1.0/k)

def project_3d_vectors_to_2d_plane(vectors_3d, plane_normal):
    dot_product = torch.sum(vectors_3d * plane_normal, dim=1)
    projection = vectors_3d - dot_product.view(-1, 1) * plane_normal
    return projection

def normal_dist(x, mean, sigma):
    prob_density = (torch.pi * sigma) * torch.exp(-0.5 * ((x - mean) / sigma) ** 2)
    return prob_density

def profile_distortion_function(dists, profile_parameters):
    num_peaks = profile_parameters.size(1) // 3
    add = torch.zeros_like(dists)

    for i in range(num_peaks):
        mean_idx = i * 3
        sigma_idx = mean_idx + 1
        magnitude_idx = mean_idx + 2

        mean = 5.0 * (i / num_peaks) + profile_parameters[:, mean_idx]
        sigma = (5.0 / num_peaks) * torch.abs(profile_parameters[:, sigma_idx])
        magnitude = profile_parameters[:, magnitude_idx]

        add += magnitude * normal_dist(dists, mean, sigma)

    return dists + add

def signed_angles_in_plane(vectors, reference_vector, plane_norm_vec):

    # Normalize the vectors
    normalized_reference = reference_vector / torch.norm(reference_vector)
    normalized_vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)

    # Project the vectors onto the 2D plane defined by the reference vector and plane normal
    projected_vectors = normalized_vectors - (normalized_vectors * plane_norm_vec).sum(dim=1, keepdim=True) * plane_norm_vec
    normalized_projected = projected_vectors / torch.norm(projected_vectors, dim=1, keepdim=True)

    # Calculate the dot products for each vector
    dot_products = (normalized_reference * normalized_projected).sum(dim=1)
    dot_products = torch.clamp(dot_products, min=-1.0, max=1.0)

    # Calculate the cross product of the vectors
    cross_product = normalized_reference[0] * normalized_projected[:, 1] - normalized_reference[1] * normalized_projected[:, 0]

    # Calculate angles
    angles = torch.acos(dot_products) #unsigned angles, just for debugging

    # Determine the sign of the angle based on the cross product
    angles[cross_product < 0.0] *= -1.0

    # change range from -pi-->pi to 0-->2pi
    angles += torch.pi

    return angles

def closest_points_on_ray(origin, direction, points):
    direction = direction / direction.norm(dim=-1, keepdim=True)
    vector_to_points = points - origin                                      # Calculate the vector from the origin to each point
    projection_lengths = torch.sum(vector_to_points * direction, dim=1)     # Calculate the projection of the vector onto the ray direction
    closest_points = origin + projection_lengths.view(-1, 1) * direction    # Calculate the closest points on the ray
    return closest_points

def cos_based_function(X, P):

    N = P.shape[1] // 2
    Y = torch.zeros(N, X.shape[0])

    for i in range(N):
        a = P[:, 2 * i]  # Extracting the entire column, not just a single element
        b = 2 * torch.pi * P[:, 2 * i + 1]
        term = a * torch.cos(i * X + b)
        Y[i] += term

    return Y.mean(dim=0)

def smoothstep_colors(A, B, C):
    # Ensure the blending factor is clamped between 0.0 and 1.0
    C_clamped = torch.clamp(C, 0.0, 1.0)
    
    # Apply the smoothstep function: C_clamped^2 * (3 - 2 * C_clamped)
    C_smooth = C_clamped * C_clamped * (3 - 2 * C_clamped)
    
    # Interpolate between A and B using the smooth blending factor
    D =  A.unsqueeze(0) + C_smooth.unsqueeze(1) * (B.unsqueeze(0) - A.unsqueeze(0))

    return D

def smoothstep(A, B, C):
    # Ensure C is clamped between 0.0 and 1.0
    C = torch.clamp(C, 0.0, 1.0)
    
    # Apply smoothstep formula: C * C * (3 - 2 * C)
    smooth_C = C * C * (3 - 2 * C)
    
    # Interpolate between A and B
    return A * (1.0 - smooth_C) + B * smooth_C

def linearstep(A, B, C):
    # Ensure C is clamped between 0.0 and 1.0
    C = torch.clamp(C, 0.0, 1.0)
    
    # Linear interpolation between A and B
    return A * (1.0 - C) + B * C

def mix(A, B, mix_factors):
    mix_factors = torch.clamp(mix_factors, 0.0, 1.0)
    A_expanded = A.unsqueeze(0)  # Shape [1, 3]
    B_expanded = B.unsqueeze(0)  # Shape [1, 3]
    mix_factors_expanded = mix_factors.unsqueeze(1)  # Shape [n, 1]
    return A_expanded * (1.0 - mix_factors_expanded) + B_expanded * mix_factors_expanded
##################

def procedural_wood_function_for_initialization(params, px_coords, A=0, B=0, return_reshaped=False, arl_type=0, return_cylindrical_coords=False):

    # Calculate distance between pixel coordinates and closest point on the pith axis
    pith_closest_pts = closest_points_on_ray(params.pith_origin, params.pith_direction, px_coords)
    pvecs = px_coords-pith_closest_pts
    dists = torch.norm(pvecs, p=2, dim=1)

    if return_cylindrical_coords: 
        # Calculate the height 
        cp_org_vec = pith_closest_pts - params.pith_origin
        cp_org_dist = torch.norm(cp_org_vec, p=2, dim=1)
        # Update values in D based on the sign of the angles
        dot_product = (cp_org_vec * params.pith_direction).sum(dim=1) # if do product is negative, the pith dir and cp_org_vec are pointing in different directions
        cp_org_dist[dot_product < 0.0] = -cp_org_dist[dot_product < 0]
        # Calculate angle of point around the central pith axis (for picking spoke)
        omegas = signed_angles_in_plane(pvecs, params.ref_vec, params.pith_direction)
        return cp_org_dist, omegas, dists
    
    if not return_reshaped: return dists

    # Reshape and re-orient image
    img_gtf = dists.reshape(A,B).t()

    return img_gtf


def procedural_wood_function_for_refinement(params, px_coords, A=0, B=0, show_knot=False, return_reshaped=False, return_cylindrical_coords=False, return_stem_knot_gtfs=False):

    # Calculate distance between pixel coordinates and closest point on the pith axis
    pith_closest_pts = closest_points_on_ray(params.pith_origin, params.pith_direction, px_coords)
    pvecs = px_coords-pith_closest_pts
    dists = torch.norm(pvecs, p=2, dim=1)

    # Calculate the height of point on the central pith axis (for picking heght section)
    cp_org_vec = pith_closest_pts - params.pith_origin
    cp_org_dist = torch.norm(cp_org_vec, p=2, dim=1)
    # Update values in D based on the sign of the angles
    dot_product = (cp_org_vec * params.pith_direction).sum(dim=1) # if do product is negative, the pith dir and cp_org_vec are pointing in different directions
    cp_org_dist[dot_product < 0.0] = -cp_org_dist[dot_product < 0]

    # Find inbetween which height sections the point lies, and interpolate annual ring values between them
    upper_bound_idxs = torch.searchsorted(params.height_levels, cp_org_dist)
    lower_bound_idxs  = upper_bound_idxs-1
    upper_bound_idxs[upper_bound_idxs>=len(params.height_levels)] = len(params.height_levels)-1 #edge case
    lower_bound_idxs[upper_bound_idxs>=len(params.height_levels)] = len(params.height_levels)-1 #edge case
    lower_bound_idxs[lower_bound_idxs<0] = 0 #edge case
    lower_height = params.height_levels[lower_bound_idxs]
    hprog = (cp_org_dist - lower_height)/params.height_step
    hprog = (3-2*hprog) * hprog * hprog # smoothstep/interpolation between scales of left/right sp
    #print("hprog range", round(hprog.min().item(),2), round(hprog.max().item(),2))
    #temp_dis = upper_bound_idxs

    # Calculate angle of point around the central pith axis (for picking spoke)
    omegas = signed_angles_in_plane(pvecs, params.ref_vec, params.pith_direction)

    # Find inbetween which spokes the point lies, and interpolate annual ring values between the two spokes
    offsetted_omegas = torch.remainder(omegas + params.spoke_offset, 2*torch.pi)
    right_bound_idxs = torch.searchsorted(params.spoke_angs, offsetted_omegas) 
    right_bound_idxs = torch.clamp(right_bound_idxs, min=0, max=params.spoke_num-1)
    right_right_bound_idxs = torch.clamp(right_bound_idxs + 1, max=params.spoke_num-1)
    left_bound_idxs = torch.clamp(right_bound_idxs - 1, min=0)
    left_left_bound_idxs = torch.clamp(right_bound_idxs - 2, min=0)
    left_angs = params.spoke_angs[left_bound_idxs]
    aprog = (offsetted_omegas - left_angs)/params.spoke_step
    #aprog = (3-2*aprog) * aprog * aprog # smoothstep/interpolation between scales of left/right sp
    
    #spoke RINGRADS. interpolate height
    lower_ring_rads = params.ring_rads[lower_bound_idxs]
    upper_ring_rads = params.ring_rads[upper_bound_idxs]
    ring_rads = lower_ring_rads * (1.0 - hprog.view(-1,1,1)) + upper_ring_rads * hprog.view(-1,1,1) # apply smoothstep between height levels
    v0 = ring_rads[torch.arange(len(left_left_bound_idxs)),    left_left_bound_idxs]
    v1 = ring_rads[torch.arange(len(left_bound_idxs)),         left_bound_idxs]
    v2 = ring_rads[torch.arange(len(right_bound_idxs)),        right_bound_idxs]
    v3 = ring_rads[torch.arange(len(right_right_bound_idxs)),  right_right_bound_idxs]
    #ring_rads = left_ring_rads * (1.0 - aprog.view(-1,1)) + right_ring_rads * aprog.view(-1,1)
    ring_rads = 0.5*( 2*v1 + (-v0+v2)*aprog.view(-1,1) + (2*v0-5*v1+4*v2-v3)*torch.pow(aprog, 2).view(-1,1) + (-v0+3*v1-3*v2+v3)*torch.pow(aprog, 3).view(-1,1))
    
    # Find in between which two annual rings each point lies
    outward_bound_idxs = torch.sum(dists.unsqueeze(1) > ring_rads, dim=1)
    outward_bound_idxs = torch.clamp(outward_bound_idxs, min=1, max=ring_rads.size()[1]-1)
    inward_bound_idxs = outward_bound_idxs-1
    inward_bounds = ring_rads[torch.arange(len(inward_bound_idxs)), inward_bound_idxs]
    outward_bounds = ring_rads[torch.arange(len(outward_bound_idxs)), outward_bound_idxs]

    ring_step = outward_bounds-inward_bounds

    # position between below/above point
    alfas = (dists-inward_bounds) / ring_step
    #print("rprog range", round(alfas.min().item(),2), round(alfas.max().item(),2))

    gtf = params.ring_min + params.ring_step * (inward_bound_idxs + alfas)
    #print("mean gtf", gtf.mean().item())
    #print("gtf range", gtf.max().item() - gtf.min().item())
    
    #gtf = inward_bounds + alfas * ring_step
    #gtf *= spoke_facs

    #gtf = upper_bound_idxs/5 #debugg
    #gtf = right_bound_idxs/5 #debugg
    #gtf = omegas
    #gtf = outward_bound_idxs/5 #debugg

    
    if show_knot:
        knot_skeleton_closest_pts = closest_points_on_ray(params.knot_origin, params.knot_direction, px_coords)
        kvecs = px_coords-knot_skeleton_closest_pts
        kdists = torch.norm(kvecs, p=2, dim=1)
        knot_gtf = kdists*params.knot_density
        stem_gtf = gtf.clone()
        gtf = torch.clamp(gtf, min=0.001)
        knot_gtf = torch.clamp(knot_gtf, min=0.001)
        gtf = power_min_smooth(gtf, knot_gtf, params.knot_smoothness)
    else:
        stem_gtf = gtf
        knot_gtf = torch.zeros_like(gtf)
        

    if not return_reshaped and return_cylindrical_coords: 

        if return_stem_knot_gtfs:   return gtf, dists, omegas, cp_org_dist, stem_gtf, knot_gtf
        else:                       return gtf, dists, omegas, cp_org_dist

    if not return_reshaped: return gtf

    img_gtf = gtf.reshape(A,B).t()

    return img_gtf

    
def procedural_wood_function_refined_and_with_1dmap(params, px_coords, side_index = 0, surface_normal_axis=0, A=0, B=0, return_reshaped=False, show_knot=False, color_map = False): # procedural, based on sampling of light and dark color in image

    # get growth time field, and coordinates for fiber direction calcualtion

    gtf = procedural_wood_function_for_refinement(params, px_coords, return_reshaped=False, show_knot=show_knot)
    
    if color_map:   M = params.color_bar
    else:           M = params.arl_color_bar

    # sample color bar
    gtf = (gtf - params.ring_min) / (params.ring_max - params.ring_min)
    gtf = torch.clamp(gtf,0.0,1.0)
    gtf *= len(M) - 2

    
    inds_floor = torch.floor(gtf).long()
    inds_floor = torch.clamp(inds_floor, 0, M.size()[0]-2)
    inds_ceil = inds_floor + 1
    frac = gtf - inds_floor.float()

    if color_map:   cols = (1 - frac.unsqueeze(-1)) * M[inds_floor] + frac.unsqueeze(-1) * M[inds_ceil]
    else:           cols = (1 - frac) * M[inds_floor] + frac * M[inds_ceil]

    if color_map:
        cols += params.side_cols[side_index]
        
    if not return_reshaped: return cols, gtf

    img_gtf = gtf.reshape(A,B).t()
    if color_map:   img = cols.reshape(A, B, -1).permute(1, 0, 2)
    else:           img = cols.reshape(A,B).t() 
    
    
    return img, img_gtf



def procedural_wood_function_knot_only(params, px_coords, side_index=0, side_axis=0, A=0, B=0, return_reshaped=False): # procedural, based on sampling of light and dark color in image

    # get growth time field, and coordinates for fiber direction calcualtion
    gtf, dists, omegas, heights, stem_gtf, knot_gtf = procedural_wood_function_for_refinement(params, px_coords, show_knot=True, return_reshaped=False, return_cylindrical_coords=True, return_stem_knot_gtfs=True)
            
    #surface normal / knot skeleton alignment (for anisotrophic color)
    norm = torch.zeros(3)
    norm[side_axis] = 1.0
    knot_surface_angle = closest_angle(norm, params.knot_direction)
    if knot_surface_angle>0.5*torch.pi: knot_surface_angle = torch.pi - knot_surface_angle
    knot_surface_angle /= 0.5*torch.pi #range 0.0-1.0

    #knot color
    #anis
    factor = 1.0 - params.knot_color_anisotrophy_factor*knot_surface_angle
    knot_col = factor*params.knot_color + (1.0-factor)*params.background_color
    #color
    sharpness = 5.0
    knot_mask = torch.sigmoid(sharpness * (stem_gtf - knot_gtf))
    cols = mix(params.background_color, knot_col, knot_mask)
    cols = torch.clamp(cols,min=0.0,max=1.0)


    if not return_reshaped: return cols

    img = cols.reshape(A,B,3) 
    img = torch.transpose(img, 0, 1)
    
    return img



def procedural_wood_function_refined_and_colors_and_details(params, px_coords, side_index=0, side_axis=0, A=0, B=0, show_fiber=False, show_pore=False, show_ray=False, show_knot=False, color_map=False, return_reshaped=False, return_gtf=False, return_iso_gtf=False): # procedural, based on sampling of light and dark color in image

    # get growth time field, and coordinates for fiber direction calcualtion
    gtf, dists, omegas, heights, stem_gtf, knot_gtf = procedural_wood_function_for_refinement(params, px_coords, show_knot=show_knot, return_reshaped=False, return_cylindrical_coords=True, return_stem_knot_gtfs=True)
            
    # calcualte how much the side axis alinged with the pith
    norm = torch.zeros(3)
    norm[side_axis] = 1.0
    pith_surface_angle = closest_angle(norm, params.pith_direction)
    if pith_surface_angle>0.5*torch.pi: pith_surface_angle = torch.pi - pith_surface_angle
    pith_surface_angle /= 0.5*torch.pi #range 0.0-1.0

    if show_knot:
        #knot mask
        sharpness = 10.0
        knot_mask = torch.sigmoid(sharpness * (stem_gtf - knot_gtf))
        #surface normal / knot skeleton alignment (for anisotrophic color)
        knot_surface_angle = closest_angle(norm, params.knot_direction)
        if knot_surface_angle>0.5*torch.pi: knot_surface_angle = torch.pi - knot_surface_angle
        knot_surface_angle /= 0.5*torch.pi #range 0.0-1.0
        # progression within knot color map
        knot_gtf_scaled = knot_gtf / params.ring_max
        knot_gtf_scaled *= params.knot_color_bar.size()[0] - 1
        inds_floor = torch.floor(knot_gtf_scaled).long()
        inds_floor = torch.clamp(inds_floor, min=0, max=params.knot_color_bar.size()[0]-2)
        inds_ceil = inds_floor + 1
        frac = knot_gtf_scaled - inds_floor.float()
        frac = frac.unsqueeze(1).repeat(1, 3)
        frac = torch.clamp(frac, min=0.0, max=1.0)
        # sample knot color map
        knot_cols = (1 - frac) * params.knot_color_bar[inds_floor] + frac * params.knot_color_bar[inds_ceil]
        # apply anisotropic factor
        factor = 1.0 - params.knot_color_anisotrophy_factor*knot_surface_angle
        if color_map:
            early_knot_cols = knot_cols
            late_knot_cols = knot_cols
        else:
            early_knot_cols = mix(params.early_wood_color.expand(knot_cols.shape[0], -1), knot_cols, factor)
            late_knot_cols = mix(params.late_wood_color.expand(knot_cols.shape[0], -1), knot_cols, torch.clamp(factor-0.2,min=0.0,max=1.0))
        #0.25*params.late_wood_color.expand(knot_cols.shape[0], -1) + 0.75*knot_cols
           
        #knot_cols = -params.knot_color * knot_mask.float().unsqueeze(1)
        #around knot color
        #sharpness = 5.0 + 2*params.knot_color_smoothness   # Larger values = sharper transition
        #knot_mask_2 = torch.sigmoid(sharpness * (stem_gtf - knot_gtf + params.knot_color_expansion))
        #knot_cols = params.knot_shadow_color * knot_mask_2.float().unsqueeze(1)
    
    # Progression within year ring
    outward_bound_idxs = torch.sum(gtf.unsqueeze(1) > params.ring_dists, dim=1)
    outward_bound_idxs = torch.clamp(outward_bound_idxs, min=1, max=params.ring_dists.size()[0]-1)
    inward_bound_idxs = outward_bound_idxs-1
    inward_bounds = params.ring_dists[inward_bound_idxs]
    outward_bounds = params.ring_dists[outward_bound_idxs]
    alfas = (gtf-inward_bounds) / (outward_bounds-inward_bounds)
    alfas = torch.clamp(alfas, min=0.0, max=1.0)

    iso_gtf = inward_bound_idxs + alfas

    if not color_map:
        #shape of annual ring transition
        ring_widths = outward_bounds - inward_bounds
        ring_widths_ratio = params.median_ring_dist/ring_widths
        lw_end = params.late_wood_end*ring_widths_ratio
        lw_st =  params.late_wood_start*ring_widths_ratio
        alfas_1_smooth = smoothstep(1.0, 0.0, alfas/lw_end)
        alfas_2_smooth = smoothstep(0.0, 1.0, (alfas-1.0+lw_st)/lw_st)
        alfas_1_linear = linearstep(1.0, 0.0, alfas/lw_end)
        alfas_2_linear = linearstep(0.0, 1.0, (alfas-1.0+lw_st)/lw_st)
        alfas_1 = params.late_wood_end_smooth_linear_ratio*alfas_1_smooth + (1.0-params.late_wood_end_smooth_linear_ratio)*alfas_1_linear
        alfas_2 = params.late_wood_start_smooth_linear_ratio*alfas_2_smooth + (1.0-params.late_wood_start_smooth_linear_ratio)*alfas_2_linear
        alfas_12 = torch.max(alfas_1, alfas_2)
        alfas_12 = torch.clamp(alfas_12, min=0.0, max=1.0)

    if show_fiber and params.fiber_size>0.0:
        # Add noise to distances
        #noisy_distances = dists + 0.001*(1/params.fiber_size)*0.01 * torch.sin(dists/2000000 * 12.9898) * 43758.5453 % 1 
        #noisy_omegas = omegas + 0.001*(1/params.fiber_size)*0.01 * torch.sin(omegas/2000000 * 12.9898) * 43758.5453 % 1 
        #x = torch.cos(noisy_omegas) * noisy_distances
        #y = torch.sin(noisy_omegas) * noisy_distances
        x = torch.cos(omegas) * dists
        y = torch.sin(omegas) * dists
        grid = torch.stack((x, y), dim=1)
        grid = (1/params.fiber_size)*grid 
        seeds = torch.floor(grid)

        ## add fiber noise based on seed
        #distances_from_rings = torch.abs(params.ring_dists.unsqueeze(1)-gtf)
        #distances_from_nearest_ring, _ = torch.min(distances_from_rings, dim=0)
        #fiber_fac = distances_from_nearest_ring / params.median_ring_dist
        #fiber_fac = torch.clamp(fiber_fac,0.0,1.0)
        #fiber_fac = 1.2-torch.pow(1.0-fiber_fac, 8)
        if not color_map: fiber_fac = 1.0-params.fiber_lw_strength*alfas_12 
        else: fiber_fac = 1.0
        #print(fiber_fac.mean(), fiber_fac.min(), fiber_fac.max())
        #fiber shadow
        fiber_noise = torch.cos(torch.sin(seeds[:,0] * 12.9898 + seeds[:,1] * 78.233) * 43758.5453)
        #fiber_cols *= params.fiber_shadow_strength # less fiber shadow on facegrain ################### debug
        fiber_cols = fiber_noise * fiber_fac # less fiber on latewood
        fiber_shadow_direction_fac = 1.0-pith_surface_angle
        #print(fiber_shadow_direction_fac.min(),fiber_shadow_direction_fac.max(), fiber_shadow_direction_fac.mean())
        fiber_cols *= params.fiber_shadow_strength*fiber_shadow_direction_fac # less fiber shadow on facegrain ################### debug
        fiber_cols = fiber_cols.unsqueeze(1)
        fiber_cols = fiber_cols.expand(-1, 3) #grey to RGB

        fiber_noise2 = torch.cos(torch.sin(seeds[:,1] * 11.9898 + seeds[:,0] * 68.213) * 15758.5453)
        #fiber_noise = 0.001*torch.remainder(torch.sin(seeds[:, 0]/200 * 12.9898) * 43758.5453,1.0) + torch.remainder(torch.sin(seeds[:, 1]/901 * 112.9898) * 4758.5453,1.0)
        #fiber_noise = torch.cos(99*fiber_noise)
        fiber_color_intensity_variation = 1.0 + 0.1*fiber_noise2*params.fiber_colblend_strength    
    else:
        fiber_cols = torch.ones(3)

    if color_map:
        gtf_scaled = (gtf - params.ring_min) / (params.ring_max - params.ring_min)
        gtf_scaled *= params.color_bar.size()[0] - 1
        inds_floor = torch.floor(gtf_scaled).long()
        inds_floor = torch.clamp(inds_floor, min=0, max=params.color_bar.size()[0]-2)
        inds_ceil = inds_floor + 1
        frac = gtf_scaled - inds_floor.float()
        frac = frac.unsqueeze(1).repeat(1, 3)
        frac = torch.clamp(frac, min=0.0, max=1.0)
        cols = (1 - frac) * params.color_bar[inds_floor] + frac * params.color_bar[inds_ceil]   
        side_col = params.side_cols[side_index]
        pro_cols = cols + side_col
        
    else: #procedural annual ring colors
        # progression within color map
        gtf_scaled = (gtf - params.ring_min) / (params.ring_max - params.ring_min)
        gtf_scaled *= params.color_bar_earlywood.size()[0] - 1
        inds_floor = torch.floor(gtf_scaled).long()
        inds_floor = torch.clamp(inds_floor, min=0, max=params.color_bar_earlywood.size()[0]-2)
        inds_ceil = inds_floor + 1
        frac = gtf_scaled - inds_floor.float()
        frac = frac.unsqueeze(1).repeat(1, 3)
        frac = torch.clamp(frac, min=0.0, max=1.0)
        # earlywood
        cols_ew = (1 - frac) * params.color_bar_earlywood[inds_floor] + frac * params.color_bar_earlywood[inds_ceil]   
        col_sides_ew = params.color_sides_earlywood[side_index]
        earlywood_col = params.early_wood_color + 0.2*cols_ew.squeeze() + 0.2*col_sides_ew
        if show_knot: earlywood_col = mix(earlywood_col, early_knot_cols, knot_mask)
        # latewood
        cols_lw = (1 - frac) * params.color_bar_latewood[inds_floor] + frac * params.color_bar_latewood[inds_ceil]   
        col_sides_lw = params.color_sides_latewood[side_index]
        latewood_col = params.late_wood_color + 0.2*cols_lw.squeeze() + 0.2*col_sides_lw
        if show_knot: latewood_col = mix(latewood_col, late_knot_cols, knot_mask)
        #apply
        pro_cols = mix(earlywood_col, latewood_col, alfas_12)

    if show_pore:
        #prepare grid of cells
        grid_d = gtf / params.pore_cell_dim_ad
        grid_a = omegas * torch.floor(grid_d)
        grid_h = heights / params.pore_cell_dim_h
        grid = torch.stack((grid_d, grid_a, grid_h), dim=1)

        #get cell ids
        grid_ids = grid.floor().long()

        #height noise
        height_noise = torch.cos(torch.sin(grid_ids[:,0] * 12937.9898 + grid_ids[:,1] * 79008.233) * 43758.5453 % 1)
        height_noise = torch.sin(torch.cos(height_noise * 19020.9898))
        grid[:,2] += height_noise

        # get cell ids again, and coords
        grid_ids = grid.floor().long()
        grid_coords = torch.remainder(grid,1.0)-0.5

        #pore_cell_ratio_rad
        pore_cell_ratio_radius = params.pore_rad/params.pore_cell_dim_ad
        if pore_cell_ratio_radius>0.5 or pore_cell_ratio_radius<0.0:
            print("radius ratio error", pore_cell_ratio_radius)

        #position-in-cell noise
        pos_noise = generate_3d_noise(grid_ids) + generate_3d_noise(99*grid_ids)
        pos_noise = generate_3d_noise(pos_noise)
        grid_coords += (0.5-pore_cell_ratio_radius)*pos_noise 
        
        #calcualte if it is a pore or not
        # Progression within year ring / #stratificed alfa
        stratified_gtf = (grid_ids[:, 0])*params.pore_cell_dim_ad 
        outward_bound_idxs = torch.sum(stratified_gtf.unsqueeze(1) > params.ring_dists, dim=1)
        outward_bound_idxs = torch.remainder(outward_bound_idxs,params.ring_dists.size()[0])
        inward_bound_idxs = outward_bound_idxs-1
        inward_bounds = params.ring_dists[inward_bound_idxs]
        outward_bounds = params.ring_dists[outward_bound_idxs]
        ring_width = outward_bounds-inward_bounds
        stratified_alfas = (stratified_gtf-inward_bounds) / ring_width
        
        #within radius?
        coords_lengths = torch.sqrt(grid_coords[:, 0]**2 + grid_coords[:, 1]**2 + grid_coords[:, 2]**2)
        #occurance
        occ_noise = torch.abs(generate_3d_noise(pos_noise).mean(dim=1))
        occ_noise = torch.tanh(occ_noise * 2.0) * 1.0  # normal approximation
        occ_noise = 0.5*(occ_noise+1.0) #from -1.0-1.0 to 0.0-1.0
        occ_prob_general = params.pore_occurance_ratio - params.pore_occ_ring_correlation*(stratified_alfas-0.5)
        #print("pore occ lw", params.pore_occurance_ratio.requires_grad)
        sharpness = 20.0  # Larger values = sharper transition
        
        #occ_prob_latewood = torch.sigmoid(sharpness * (params.pore_latewood_occ_dist - 0.05 - (0.5-torch.abs(stratified_alfas-0.5))))   #params.base_pore_occ_latewood * (1.0 - torch.pow(stratified_alfas,2))
        occ_prob_latewood = params.pore_latewood_occ * torch.sigmoid(sharpness * (0.5*params.pore_cell_dim_ad/ring_width - (0.5-torch.abs(stratified_alfas-0.5)))) 
        #print("pore occ lw", params.pore_latewood_occ.requires_grad)
        occ_prob = occ_prob_general + occ_prob_latewood
        #rad variation
        scale_pore_cell_ratio_radii = pore_cell_ratio_radius*(1.0 - 0.01*params.pore_rad_scale_ring_correlation*stratified_alfas) #pore_mask = ((coords_lengths < pore_cell_ratio_radius) and ((occ_noise < occ_prob) or occ_prob).float()
        sharpness = 20.0  # Larger values = sharper transition
        pore_mask = torch.sigmoid(sharpness * (scale_pore_cell_ratio_radii - coords_lengths)) * torch.sigmoid(sharpness * (occ_prob-occ_noise))

        #apply pore col
        pore_cols = (1.0 - params.pore_direction_strength*pith_surface_angle) * params.pore_color
        #print("pore dir_str rg", params.pore_direction_strength.requires_grad)
        #print("pore col rg", params.pore_color.requires_grad)

        pore_cols = pore_cols * pore_mask.float().unsqueeze(1)
    else:
        pore_cols = torch.ones(3)
    #print("pore cols rg", pore_cols.requires_grad)

    if show_ray:
        
        #prepare grid of cells
        grid_d = gtf / params.ray_cell_dim_d
        grid_a = omegas * torch.floor(grid_d) * params.ray_cell_dim_d / params.ray_cell_dim_a
        grid_h = heights / params.ray_cell_dim_h
        grid = torch.stack((grid_d, grid_a, grid_h), dim=1)

        #get cell ids
        grid_ids = grid.floor().long()

        #height noise
        height_noise = torch.cos(torch.sin(grid_ids[:,0] * 12937.9898 + grid_ids[:,1] * 79008.233) * 43758.5453 % 1)
        height_noise = torch.sin(torch.cos(height_noise * 19020.9898))
        grid[:,2] += height_noise

        # get cell ids again, and coords
        grid_ids = grid.floor().long()
        grid_coords = torch.remainder(grid,1.0)-0.5

        #ray cell ratio width and length/height
        ray_cell_ratio_length = params.ray_length/params.ray_cell_dim_d
        ray_cell_ratio_width = params.ray_width/params.ray_cell_dim_a
        ray_cell_ratio_height = params.ray_height/params.ray_cell_dim_h
        
        #position-in-cell noise
        pos_noise = generate_3d_noise(grid_ids) + generate_3d_noise(99*grid_ids)
        pos_noise = generate_3d_noise(pos_noise)
        grid_coords[:, 0] += (0.5-ray_cell_ratio_length)*pos_noise[:, 0]
        grid_coords[:, 1] += (0.5-ray_cell_ratio_width)*pos_noise[:, 1] 
        grid_coords[:, 2] += (0.5-ray_cell_ratio_height)*pos_noise[:, 2] 

        #within ellipse?
        coords_lengths = torch.sqrt((grid_coords[:, 0] / ray_cell_ratio_length)**2 + 
                                    (grid_coords[:, 1] / ray_cell_ratio_width)**2 + 
                                    (grid_coords[:, 2] / ray_cell_ratio_height)**2)
        #occurance
        occ_noise = torch.abs(generate_3d_noise(pos_noise).mean(dim=1))
        occ_noise = torch.tanh(occ_noise * 2.0) * 1.0  # normal approximation
        occ_noise = 0.5*(occ_noise+1.0) #from -1.0-1.0 to 0.0-1.0
        occ_prob = params.ray_occurance_ratio
        sharpness = 20.0  # Larger values = sharper transition
        ray_mask = torch.sigmoid(sharpness * (1.0 - coords_lengths)) * torch.sigmoid(sharpness * (occ_prob - occ_noise))
        #apply ray col
        ray_mult_cols = params.ray_multiply_color * ray_mask.float().unsqueeze(1)
    else:
        ray_mult_cols = torch.ones(3)
        ray_mask = torch.zeros_like(gtf)

    #combine
    if show_knot: 
        if show_fiber: 
            knot_mask = 1.0-knot_mask
            fiber_cols = fiber_cols*knot_mask.unsqueeze(1)
            fiber_color_intensity_variation = 1.0 + (fiber_color_intensity_variation - 1.0) * knot_mask
        if show_pore: 
            pore_cols = pore_cols*knot_mask.unsqueeze(1)
        if show_ray: 
            ray_mult_cols = ray_mult_cols*knot_mask.unsqueeze(1)
     
    if show_fiber: 
        pro_cols = fiber_color_intensity_variation.unsqueeze(1) * pro_cols
        fiber_cols = torch.clamp(1.0-fiber_cols, min=0.0, max=1.0)
    
    if show_pore: pore_cols = torch.clamp(1.0-pore_cols, min=0.5, max=1.0)
    if show_ray:  ray_mult_cols = torch.clamp(1.0-ray_mult_cols, min=0.5, max=1.0)
    
    cols = pro_cols * fiber_cols * pore_cols * ray_mult_cols

    if show_ray:
        #ray_over_cols = (1.0-params.ray_mult_over_balance)*cols + params.ray_mult_over_balance*params.ray_overlay_color
        cols = mix(cols, params.ray_overlay_color, params.ray_mult_over_balance*ray_mask)

    #if show_pore: pore_cols = torch.clamp(pore_cols, min=0.0, max=0.5)
    #if show_ray:  ray_cols = torch.clamp(ray_cols, min=0.0, max=0.5)
    #cols = pro_cols * fiber_cols - 0.1*pore_cols - 0.1*ray_cols

    cols = torch.clamp(cols,0.0,1.0)

    #cols = alfas_12.unsqueeze(1).repeat(1, 3)

    if not return_reshaped: return cols

    img = cols.reshape(A,B,3) 
    img = torch.transpose(img, 0, 1)
    img = img.to(torch.float32)

    if return_gtf and return_iso_gtf:
        img_gtf = gtf.reshape(A,B).t() 
        img_iso_gtf = iso_gtf.reshape(A,B).t()
        return img, img_gtf, img_iso_gtf
    elif return_gtf: 
        img_gtf = gtf.reshape(A,B).t()
        return img, img_gtf
    elif return_iso_gtf:
        img_iso_gtf = iso_gtf.reshape(A,B).t()
        return img, img_iso_gtf

    return img
