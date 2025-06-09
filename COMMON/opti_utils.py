import torch
from datetime import datetime
import torch.nn.functional as F
from matplotlib import pyplot as plt

torch.autograd.set_detect_anomaly(True)

def get_elapsed_time(start_time, print_=False):
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_days = elapsed_time.days
    elapsed_hrs = elapsed_time.seconds // 3600
    elapsed_mins = (elapsed_time.seconds % 3600) // 60
    elapsed_secs = elapsed_time.seconds % 60

    if print_:
        print("\nElasped time:", str(elapsed_days), 'days', str(elapsed_hrs), "h", str(elapsed_mins), "min", str(elapsed_secs), "s\n")

    return elapsed_days, elapsed_hrs, elapsed_mins, elapsed_secs



def epoch_export_plot(OUT_PATH, train_loss_log, train_image_loss_log, PLT_SUBTITLE, elapsed_days, elapsed_hrs, elapsed_mins, NUM_EPOCHS, PLT_YLIM):
    plt.clf()
    plt.figure(figsize=(9, 6))
    plt.plot(train_image_loss_log, label='Train_L1-image_loss', linewidth=1.5)
    plt.plot(train_loss_log, label='Train loss', linewidth=2, color='green')
    plt.title(PLT_SUBTITLE + ". Duration: " + str(elapsed_days) + "days " + str(elapsed_hrs) + "h " + str(elapsed_mins) + "min", fontsize=10)
    plt.xlim([0,NUM_EPOCHS-1])
    plt.ylim(PLT_YLIM)
    plt.xlabel("Number of epochs")
    plt.ylabel("Error" )
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.9)
    plt.savefig(OUT_PATH+"//plot.png",bbox_inches='tight',dpi=300)
    #plt.show()
    plt.close()


def closest_point_on_line(org, dir, pt):
    t = torch.dot(pt-org, dir); #parameter (t) of closest point on ray from center
    cp = org + dir * t;    #closest point (cp) on ray from center
    return cp

def regularization_of_deformations(X):
    HDIFF = (torch.abs(X[:, :-1, :] - X[:, 1:, :])).mean()
    VDIFF = (torch.abs(X[:-1, :, :] - X[1:, :, :])).mean()
    regularization_term = HDIFF + VDIFF + torch.pow(X,2).mean()
    return regularization_term


def regularization_of_colorbar_smoothness(X):
    color_diff = X[1:] - X[:-1]
    regularization_term = torch.mean(torch.norm(color_diff, dim=1))
    return regularization_term     

def closest_points_on_line(org, dir, pts): #under developement
    a = pts-org
    b = torch.stack((dir,) * a.shape[0], axis=-1).transpose(1,0)
    ts = torch.tensordot(a, b, dims=([1],[1])) #parameter (t) of closest point on ray from center
    t = torch.dot(pts[0]-org, dir)
    cps = org + dir * ts;    #closest point (cp) on ray from center
    return cps

def calculate_loss(params, contours):

    # cost - points on the same trace should have similar distance to center
    cost = 0
    for cpts in contours:
        ccpts = torch.clone(cpts)
        for i,cpt in enumerate(cpts):
            ccpts[i] = closest_point_on_line(params.P, params.dir, cpts[0])
        #print("ccpt",ccpt)
        #ccpts = closest_points_on_line(params.P, params.dir, cpts)
        #print("ccpts0",ccpts[0])

        dists = torch.linalg.norm(torch.subtract(cpts,ccpts), dim=1)
        #dists2 = torch.linalg.norm(torch.subtract(cpts,params.P), dim=1)
        dist_diffs = torch.subtract(dists,torch.median(dists))
        cost += torch.sum(torch.pow(dist_diffs, 2))

    return cost


def get_step_value(val_range, num_steps, index):
    val_0 = val_range[0]
    val_step = (val_range[1]-val_range[0])/(num_steps-1)
    val = val_0 + index*val_step
    return val


def upsample_tensor(X, target_size):
    # Get the size of the last dimension
    #original_size = X.size(-1)
    
    # The new size will be double the current last dimension
    #target_size = original_size * 2
    
    # Perform the linear interpolation
    upsampled_tensor = F.interpolate(X, size=target_size, mode='linear', align_corners=True)

    
    return upsampled_tensor




