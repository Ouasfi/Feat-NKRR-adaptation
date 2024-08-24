   
#from __future__ import annotations
from tabulate import tabulate
import time
import pandas as pd
import torch
import trimesh
from convonet import load_convonet
from poco import load_poco
import dataset
from evaluation import MeshEvaluator
from reconstruction import Field,Reconstructor
from adapters import datamodule, krr, gp
from adapters import falkonnkrr as fkrr
from adapters.utils import  init_args, median_heuristic, get_gt_srb, bbox_unscale
import gpytorch
import warnings
warnings.filterwarnings("ignore")
import tqdm
class KernelField:
    
    def __init__(self, field, kernel_solver, feat_ctx, feat_fn, 
                 output_transform = lambda x :x):
        self.field = field
        self.feat_ctx = feat_ctx
        self.feat_fn = feat_fn
        self.kernel_solver = kernel_solver
        self.model = self.field.model
        self.output_transform = output_transform
        self.latents = field.latents
        
    def __call__(self, points):
        with self.feat_ctx(self.field.model) as f_i:
            outputs = self.field( points)
        features = f_i[0][0] 
        #features = datamodule.normalize_features(features)
        points_features = features.transpose(1,2).reshape(-1,features.size(1)).contiguous()
        outputs = self.kernel_solver( points_features)
        return self.output_transform (outputs)
class ReconstructionPipeline:
    def __init__(self, args):
        """
        Initializes a ReconstructionPipeline instance.

        Args:
            args: An object containing the necessary arguments for the pipeline.
                It should have the following attributes:
                    - classe: The class of the shape.
                    - shape: The name of the shape.
                    - n_points: The number of points in the point cloud.
                    - sigma_pc: The standard deviation of the point cloud.
                    - dataset: The dataset to use (e.g., 'srb').
                    - backbone: The backbone model to use (e.g., 'convonet').
        """
        self.args = args
        self.root = args.root
        self.shape_name = args.shape
        self.filesplit = f'{self.root}/{self.args.classe}/{self.shape_name}' 
        if args.__dict__.get('dataset')!='faust':
            self.filesplit = f'{self.filesplit}/'
        self.n_points, self.sigma_pc = args.n_points, args.sigma_pc
        self.resolution = 128
        self.meshevaluator = MeshEvaluator()
        self.scale_fn = lambda x:x
        if args.__dict__.get('dataset') == 'srb':
            self.pointcloud = get_gt_srb(self.args.shape, n_points=1000000, root=self.root)
            bounds = trimesh.load(f'{self.root}/{self.args.classe}/{self.shape_name}.ply').bounds
            self.scale_fn = lambda x: bbox_unscale(x, bounds)
    def load_data(self):
        """
        Loads point cloud data and preprocesses it for the reconstruction pipeline.

        This function takes no parameters and returns the preprocessed input points as a PyTorch tensor.
        The input points are loaded from a point cloud file, and then converted to a PyTorch tensor and moved to the CUDA device.
        The function also stores the random number generator, input points, and point cloud data as instance variables.

        Returns:
            torch.Tensor: The preprocessed input points as a PyTorch tensor.
        """

        #self.split_file = f'{self.root}/{self.args.classe}/{self.shape_name}/'
        input_dict = dataset.load_pointcloud( self.filesplit, self.n_points, self.sigma_pc)
        self.rng, self.input_points = input_dict['rng'], input_dict['input_points']
        inputs = torch.from_numpy(self.input_points).cuda().unsqueeze(0).float()
        self.pointcloud = input_dict['pointcloud']
        self.inputs = {'pos':inputs.transpose(1,2),"x" : torch.ones_like(inputs.transpose(1,2))  }

        return self.inputs

    def load_model(self):
        """
        Loads a backbone model based on the provided configuration.

        The model is loaded from a configuration file and moved to the CUDA device.
        The function supports two types of backbone models: 'convonet' and 'poco'.

        Returns:
            The loaded model as a PyTorch module.
        """
        if self.args.backbone == "convonet":
            conf = f"/home/amine/convolutional_occupancy_networks/configs/pointcloud/pretrained/shapenet_grid32.yaml"
            self.model = load_convonet(conf)
        elif self.args.backbone == "poco":
            conf = '/home/amine/NAS1/CVPR23/experiments/pretrained_poco_3000/config.yaml'
            self.model = load_poco(conf)
        elif self.args.backbone == "poco_abc":
            conf = '/home/amine/NAS1/CVPR23/experiments/ABC_10k_FKAConv_InterpAttentionKHeadsNet_None/config.yaml'
            self.model = load_poco(conf)
        self.model.to('cuda')
        return self.model

    def create_field(self, model, inputs):
        """
        Creates a field object based on the provided model and input points.

        Args:
            model: The backbone model used for reconstruction.
            inputs: The input points as a PyTorch tensor.

        Returns:
            Field: The created field object.
        """
        field = Field(model, inputs, encode_method="get_latent", output_transform=lambda x: x)
        return field

    def reconstruct(self, field, resolution):
        """
        Reconstructs a mesh using the given field.

        Args:
            field (Field): The field to use for reconstruction.

        Returns:
            Mesh: The reconstructed mesh.
        """
        rec = Reconstructor(field)
        mesh = rec(threshold=0.5, resolution=resolution, bounds=(-0.5, 0.5), mc_device='cpu')
        return mesh

    def evaluate_reconstruction(self, mesh):
        """
        Evaluates the quality of a reconstructed mesh by comparing it to a point cloud.

        Args:
            mesh: The reconstructed mesh to be evaluated.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the evaluation metrics, including chamfer-L1, chamfer-L2, and normals.
        """
        eval_dict = self.meshevaluator.eval_mesh(self.scale_fn(mesh), self.pointcloud, None, onet_samples=None)
        df = pd.DataFrame(eval_dict, index=[self.args.backbone])[['chamfer-L1', 'chamfer-L2', 'normals']]
        return df

    def run(self,resolution):
        """
        Runs the reconstruction pipeline using the backbone model.

        Args:
            None

        Returns:
            tuple: A tuple containing the reconstructed mesh and a pandas DataFrame with evaluation metrics.
        """
        inputs = self.load_data()
        self.load_model()
        self.field = self.create_field(self.model, inputs)
        mesh = self.reconstruct(self.field,resolution)
        df = self.evaluate_reconstruction(mesh)
        print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
        return mesh, df
    
    def prepare_krr_dataset(self,n_local_queries = 3):
        """
        Prepares the KRR dataset by normalizing features, creating a DataModule, 
        and computing the median heuristic for the sigma value.

        Args:
            n_local_queries (int, optional): The number of local queries. Defaults to 3.

        Returns:
            tuple: A tuple containing the training data (X_train, Y_train) and the Nystrom points (X_nystrom).
        """
        normalize_features = self.args.normalize
        feat_ctx = self.model.feat_ctx
        self.model.feat_ctx = lambda m: feat_ctx(m, normalize=normalize_features)

        krr_dataset = datamodule.DataModule(self.field, self.model.feat_ctx, self.model.feat_fn)
        data, X_train, Y_train = krr_dataset.get(n_local_queries=n_local_queries)
        idx = self.rng.choice(self.n_points, self.args.n_nystrom, replace=False)
        X_nystrom = X_train[idx]

        self.opt_args = init_args(self.args, X_nystrom)
        self.opt_args.sigma = median_heuristic(X_nystrom.cuda().contiguous(), sigma_type='single') / 2
        print("\033[1mMedian heuristic\033[0m: {:.5f}".format(self.opt_args.sigma))

        #print("Median heuristic: {:.5f}".format(self.opt_args.sigma))
        return X_train, Y_train,X_nystrom
    @staticmethod
    def get_field_volume(kernel_solver,  split_feature_volume,resolution = 128 ):
        """
        Computes the field volume from a kernel solver and split feature volume.

        Args:
            kernel_solver: The kernel solver to use for computing the field volume.
            split_feature_volume: The split feature volume to process.
            resolution (int, optional): The resolution of the field volume. Defaults to 128.

        Returns:
            torch.Tensor: The computed field volume.
        """
        logits_list = []
        with torch.no_grad():
            for split_features in split_feature_volume:
                #print(split_features.shape)

                preds = kernel_solver( split_features.cuda())
                logits_list.append(preds.squeeze(0).detach())

        logits = torch.cat(logits_list, dim=0)
        field_volume = logits.squeeze().view( (resolution,) * 3)
        return field_volume
    def fit_krr_solver(self, X_train, Y_train,X_nystrom):
        """
        Fits the KRR solver with the given training data and returns the trained solver.

        Parameters:
            X_train (torch.Tensor): The training input data.
            Y_train (torch.Tensor): The training output data.
            X_nystrom (torch.Tensor): The Nystrom points for the KRR solver.

        Returns:
            krr_solver (KernelSolver): The trained KRR solver.
        """
        
        krr_solver = krr.KernelSolver(self.opt_args, X_nystrom)

        start = time.perf_counter()
        krr_solver.fit(X_train.cuda(), Y_train.cuda())
        end = time.perf_counter()
        elapsed = end - start
        print(f"\033[1m\033[92mTime taken in Fit:\033[0m {elapsed:.6f} seconds")

        return krr_solver

    def reconstruct_krr(self, krr_solver, resolution, bounds=(-0.5, 0.5)):
        """
        Reconstructs a mesh using the fitted KRR solver .

        Parameters:
            krr_solver: The KRR solver to be used for reconstruction.
            resolution (int): The resolution of the reconstruction.
            bounds (tuple, optional): The bounds of the reconstruction. Defaults to (-0.5, 0.5).

        Returns:
            tuple: A tuple containing the reconstructed mesh kernel and the feature volume.
        """
        kernelfield = KernelField(self.field, krr_solver, self.model.feat_ctx, self.model.feat_fn, output_transform=lambda x: x)
        rec = Reconstructor(kernelfield)

        start = time.perf_counter()
        grid_points = rec.get_mc_points(resolution, bounds=bounds, batch_points=50000)
        feature_volume = Reconstructor.compute_feature_volume(self.field, grid_points, resolution=resolution, mc_device='cuda')
        split_feature_volume = torch.split(feature_volume, 50000, dim=0)
        field_volume = self.get_field_volume(kernelfield.kernel_solver, split_feature_volume, resolution=resolution)
        field_volume = torch.reshape(field_volume, (resolution,) * 3)
        mesh_kernel = rec.run_mc(field_volume, threshold=0.5, resolution=resolution, bounds=bounds, mc_device='cuda')
        end = time.perf_counter()
        elapsed = end - start
        print(f"\033[1m\033[92mTime taken in Kernel Reconstruction:\033[0m {elapsed:.6f} seconds")

        return mesh_kernel,feature_volume
    def run_nkrr_adaptation(self,  feature_volume, X_nystrom, X_train, Y_train, epochs,resolution = 128):
        """
        Runs the NKRR adaptation process using the provided feature volume, Nystrom points, training data, and epochs.

        Parameters:
            feature_volume (torch.Tensor): The feature volume to be used for NKRR validation.
            X_nystrom (torch.Tensor): The Nystrom points for the NKRR solver.
            X_train (torch.Tensor): The training input data.
            Y_train (torch.Tensor): The training output data.
            epochs (int): The number of epochs for the NKRR adaptation process.

        Returns:
            tuple: A tuple containing the chamfer distance (cd1_gt) and the nkrr_adapter function.
        """
        mc_device = "cuda"
        DEVICE = 'cuda'
        
        print("\033[1m\033[94mEvaluation of Falkon NKRR Adaptation\033[0m") 
        fkrr_adapter = fkrr.FalkonNKRRAdapter(opt_args=self.opt_args, 
                                            feature_volume=feature_volume,
                                            centers_init=X_nystrom.contiguous().cuda())
        alphas, centers, sigmas  = fkrr_adapter.adapt(X_train, Y_train, epochs)

        cd1_gt, cd1_val, params = fkrr_adapter.eval_epochs( alphas, centers, sigmas, resolution, Reconstructor, self.meshevaluator, self.input_points, self.pointcloud, verbose=not self.args.silent)
        eval_fkrr = {'chamfer-L1': cd1_gt}
        df_fkrr = pd.DataFrame(eval_fkrr, index=[f'{self.args.backbone}_fkrr'])[['chamfer-L1']]
        fkrr_adapter.model.kernel.sigma.data = params['sigma'].data.cuda()
        solver_predict = lambda split_features: fkrr_adapter.model.kernel.mmv(
                                                    split_features, 
                                                    params['center'].clone().cuda(), 
                                                    params['alpha'].clone().cuda(),
                                                    opt=fkrr_adapter.model.flk_opt).squeeze()
        return cd1_gt, solver_predict
    def evaluate_nkrr_adaptater(self,  nkrr_adapter, split_feature_volume, resolution):
        """
        Evaluates the NKRR adapter by reconstructing a mesh  and computing the evaluation metrics.

        Parameters:
            nkrr_adapter: The NKRR adapter to be evaluated.
            split_feature_volume (torch.Tensor): The backbone feature volume used for reconstruction.
            resolution (int): The resolution of the reconstruction.

        Returns:
            tuple: A tuple containing the reconstructed mesh kernel and the evaluation metrics.
        """

        field_volume = self.get_field_volume(nkrr_adapter,  split_feature_volume,resolution = resolution )
        #field_volume =  kernelfield.kernel_solver( feature_volume.cuda()).squeeze()
        field_volume = torch.reshape(field_volume, (resolution,) * 3) 
        mesh_kernel = Reconstructor.run_mc (field_volume, threshold = 0.5, resolution  =resolution,  bounds= (-0.5, 0.5),mc_device = 'cuda' )
        df = self.evaluate_reconstruction(mesh_kernel)
        return mesh_kernel, df
    
    def run_sgpr_adatation(self, feature_volume, X_nystrom, X_train, Y_train,epochs = 102,resolution = 128):
        split_feature_volume = torch.split(feature_volume,20000, dim=0) 
        likelihood = gpytorch.likelihoods.GaussianLikelihood()#gpytorch.likelihoods.SimpleGaussianLikelihood(noise = 1e-3*torch.ones_like(X_train[:,1]),
        #noise_constraint=gpytorch.constraints.GreaterThan(0.0)).cuda()
        X_train = X_train.contiguous().to('cuda')
        Y_train = Y_train.contiguous().to('cuda').squeeze() 
        model = gp.GPRegressionModel(X_train, Y_train, X_train[:self.args.n_nystrom, :],
                                    likelihood, sigma = self.opt_args.sigma.data).cuda()
  
        model.likelihood.noise_covar.noise = 0.005
        gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True)
        # To make sure that the noise is not optimized, we need to do:
            #model.likelihood.noise_covar.raw_noise.requires_grad_(False);   
        model.base_covar_module.raw_lengthscale.data =self.opt_args.sigma.data
        model.base_covar_module.raw_lengthscale.requires_grad = False
        for name, param in model.named_parameters():
            if name not in ['covar_module.inducing_points']:
                #continue
                print(f'Parameter name: {name:42} value = {param.mean().item():1.2f}')
   
        def train(model, train_x, train_y, n_iter=10, lr=0.1):
            """Train the model.

                Arguments
                model   --  The model to train.
                train_x --  The training inputs.
                train_y --  The training labels.
                n_iter  --  The number of iterations.
            """
            model.train()

            training_iterations = n_iter
            # Find optimal model hyperparameters
            model.train()
            likelihood.train()
            # Use the adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            iterator = tqdm.tqdm(range(training_iterations), desc="Train")
            min_cd_val = float('inf')
            min_cd_gt = None
            for i in iterator:
                # Zero backprop gradients
                def closure():
                    optimizer.zero_grad()
                # Get output from model
                    output = model(train_x)
                # Calc loss and backprop derivatives
                    loss = -mll(output, train_y.squeeze())
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
                #scheduler.step()
                torch.cuda.empty_cache()
    
                if (i+1)%30==0:
                    for name, param in model.named_parameters():
                        if name not in ['covar_module.inducing_points']:
                            #continue
                            print(f'Parameter name: {name:42} value = {param.mean().item():1.2f}')
                    field_volume = gp.get_field_volume(model,likelihood, split_feature_volume )
                    mesh_kernel = Reconstructor.run_mc (field_volume, threshold = 0.5, resolution  =resolution,  bounds= (-0.5, 0.5),mc_device = 'cuda' )
                    #cd_gt = self.meshevaluator.eval_mesh(self.scale_fn(mesh_kernel), self.pointcloud, None, onet_samples=None)
                    cd_gt = self.evaluate_reconstruction(mesh_kernel)
                    cd_val = self.meshevaluator.eval_mesh(self.scale_fn(mesh_kernel), self.input_points, None, onet_samples=None)
                #print(f" CD (GT): {cd_gt:.4f} | CD (Val): {cd_val:.4f}| ")
                    if cd_val['chamfer-L1'] < min_cd_val:
                        min_cd_val = cd_val['chamfer-L1'] 
                        min_cd_gt = cd_gt
                    torch.cuda.empty_cache()
                    iterator.set_postfix(loss=cd_gt['chamfer-L1'] )
            return min_cd_gt ,mesh_kernel
        return  train(model, X_train, Y_train, n_iter=epochs, lr=0.1)   

