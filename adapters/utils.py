import torch
from pykeops.torch import Genred
import random
import numpy as np
def init_args(args, X_nystrom):
    class Args :
        #sigma = torch.tensor([args.sigma] * X_nystrom.shape[1], dtype=torch.float32).cuda()
        sigma = torch.tensor([args.sigma], dtype=torch.float32).cuda()
        penalty = torch.tensor(args.penalty, dtype=torch.float32).cuda()#.to('cuda:0')#.requires_grad_()
        optim_sigma = args.optim_sigma
        #optim penalty and optim_nystrom
        optim_nystrom = args.optim_nystrom
        optim_penalty = args.optim_penalty
        n_nystrom = args.n_nystrom
        n_rff =  args.n_rff
        solver = args.solver
        #X_nystrom.requires_grad = args.optim_nystrom
    return Args
def median_heuristic(X: torch.Tensor, sigma_type: str, num_rnd_points=None):
    # https://arxiv.org/pdf/1707.07269.pdf
  
    if sigma_type == "diag":
        #sigmas = [median_heuristic(X[:, i : i + 1], "single", None) for i in range(X.shape[1])]
        #return torch.tensor(sigmas)
        D = X.shape[1]
        formula = " Sqrt(Square(x-y))"
        aliases = [
            "x = Vi(" + str(D) + ")",  # First arg:  i-variable of size D
            "y = Vj(" + str(D) + ")",  # Second arg: j-variable of size D
        # Third arg:  j-variable of size Dv
        ] 

        predict = Genred(formula, aliases, axis = 1)
        return (predict  (X,X) / X.shape[0]).median(0).values
    else:
        # Calculate pairwise distances
        dist = torch.pdist(X, p=2)
        med_dist = torch.median(dist)
        return med_dist
def fix_seeds():
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

def set_all_seeds(seed):
       #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set seed for numpy
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    random.seed(seed)
    # Set seed for torch
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())
    cuda_rng_state = torch.cuda.get_rng_state(0)
    torch.cuda.set_rng_state(cuda_rng_state)
    rng_state = torch.get_rng_state()
    torch.set_rng_state(rng_state)
    # Set seed for torch
    
def get_gt_srb(shape,n_points = 1000000,root = '../DiGS/data/deep_geometric_prior_data/'):
    gt_pc = trimesh.load(root + '/ground_truth/'+ shape + '.xyz')
    gt_points = gt_pc.vertices
    idx_new = np.random.randint(gt_points.shape[0], size=n_points)
    gt_points = gt_points[idx_new].astype(np.float32)
    return gt_points
def bbox_unscale(mesh, bbox):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    mesh = trimesh.Trimesh(vertices, faces)
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max() / (1 -0.01)

        # Transform input mesh
    mesh.apply_scale(scale)
    mesh.apply_translation(loc)

    scaled_mesh_o3d = o3d.geometry.TriangleMesh( o3d.utility.Vector3dVector( mesh.vertices),
                                      o3d.utility.Vector3iVector( mesh.faces) )
    return scaled_mesh_o3d