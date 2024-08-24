import numpy as np
def load_pointcloud(split_file, n_points, sigma_pc):
    """
    Loads a pointcloud from a given file and returns a dictionary containing the pointcloud, 
    a set of randomly sampled points from the pointcloud with added noise, and the random number generator used.

    Parameters:
    split_file (str): The path to the pointcloud file.
    n_points (int): The number of points to sample from the pointcloud.
    sigma_pc (float): The standard deviation of the noise to add to the sampled points.

    Returns:
    dict: A dictionary containing the input points, the pointcloud, and the random number generator.
    """
    scr = 183965288784846061718375689149290307792 #secrets.randbits(128)
    rng = np.random.default_rng( scr )     
    try:
        pointcloud = np.load(f'{split_file}pointcloud.npz')['points'].astype(np.float32)
    except:
        pointcloud = np.load(f'{split_file}/pointcloud.npz')['points'].astype(np.float32)
    idx = rng.choice(pointcloud.shape[0], n_points, replace = False)
    # simulate noisy points
    input_points= pointcloud[idx, :] + sigma_pc* rng.normal( 0,1 , (n_points,3))
    
    ##return a dict inputs pointclouds and rng
    input_dict = {'input_points' : input_points, 'pointcloud' : pointcloud, 'rng' : rng}
    return input_dict