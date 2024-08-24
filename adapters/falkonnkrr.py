
import torch
import numpy as np
import falkon.hopt
from falkon import FalkonOptions
import  falkon.hopt.objectives.stoch_objectives.stoch_new_compreg as stnc
from importlib import reload
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor,as_completed
import time
## class named Falkon adapter has init, fit and evaluate methods uses code frpù @krr_falkon example 
class FalkonNKRRAdapter():
    def __init__(self, *args, **kwargs):
        """
        Initializes a FalkonNKRRAdapter instance.

        Args:
            *args: Variable length argument list.
            **kwargs: keyword arguments.

        kwargs:
            feature_volume: The feature volume.
            opt_args: Optimization arguments.
            centers_init: Initial centers.

        Raises:
            ValueError: If opt_args, penalty, or centers_init are not provided.

        Returns:
            None
        """
        #super().__init__(*args, **kwargs)
        self.DEVICE ='cuda'
        self.flk_opt = FalkonOptions(use_cpu=False)

        self.feature_volume = kwargs.get("feature_volume")
        opt_args = kwargs.get("opt_args")
        if opt_args is None:
            raise ValueError("opt_args is required")

        sigma = opt_args.sigma
        optim_sigma = opt_args.optim_sigma
        self.falkonkernel = falkon.kernels.GaussianKernel(
            sigma=torch.tensor([sigma] * 32).requires_grad_(optim_sigma), opt=self.flk_opt
        )

        penalty = opt_args.penalty
        if penalty is None:
            raise ValueError("penalty is required")

        centers_init = kwargs.get("centers_init")
        if centers_init is None:
            raise ValueError("centers_init is required")

        self.model = stnc.StochasticNystromCompReg(
            flk_opt=self.flk_opt,
            kernel=self.falkonkernel,
            penalty_init=penalty.detach().requires_grad_(opt_args.optim_penalty),
            centers_init=centers_init,
            opt_penalty=opt_args.optim_penalty,
            opt_centers=opt_args.optim_nystrom,
        )
        self.model.to(self.DEVICE)
        ## printif self.model params requires grad

        print("Optimize Nystrom samples: ", self.model.centers_.requires_grad)
        print("Optimize Sigma: ", self.model.kernel.sigma.requires_grad)
        print("Optimize penalty: ", self.model.penalty.requires_grad)

    def adapt(self,X_train, Y_train, epochs):
        """
        Adapt the model to the given training data.

        Args:
            X_train (torch.Tensor): The input training data.
            Y_train (torch.Tensor): The target training data.
            epochs (int): The number of epochs to train the model.

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]: A tuple containing three lists. The first list contains the alpha values, the second list contains the centers, and the third list contains the sigmas.

        """
        alphas, centers, sigmas, penalties = [], [], [], []
        X_train = X_train.contiguous().to(self.DEVICE)
        Y_train = Y_train.contiguous().to(self.DEVICE)
        opt_hp, schedule = create_optimizer(opt_type = 'adam', model = self.model,  learning_rate = 1e-2)
        def closure():
            opt_hp.zero_grad()  # Reset gradients
            loss =  self.model(X_train, Y_train)
             # Backward pass to compute gradients
            loss.backward() 
            return loss
        for epoch in tqdm(range(epochs))  :
        # Evaluate and print the loss after each epoch
            loss = opt_hp.step(closure)
            #Detach clone and store centers alphas and sigmas in the corresponding lists
            alphas.append(stnc.NystromCompRegFn.last_alpha.detach().clone().cpu())
            centers.append(self.model.centers.detach().clone().cpu())
            sigmas.append(self.model.kernel.sigma.detach().clone().cpu())

        return alphas, centers, sigmas   
    def evaluate_kernel(self, center, alpha, sigma, resolution, rec, meshevaluator, pointcloud,scale_fn = lambda x:x):
        """
        Evaluates the kernel of the model given the center, alpha, sigma, resolution, 
        reconstruction method, mesh evaluator, point cloud, and a scaling function.

        Parameters:
            center (torch.Tensor): The center of the kernel.
            alpha (torch.Tensor): The alpha value of the kernel.
            sigma (torch.Tensor): The sigma value of the kernel.
            resolution (int): The resolution of the kernel.
            rec (Reconstructor): The reconstruction method.
            meshevaluator (MeshEvaluator): The mesh evaluator.
            pointcloud (torch.Tensor): The point cloud.
            scale_fn (function, optional): The scaling function. Defaults to lambda x:x.

        Returns:
            tuple: A tuple containing the chamfer distance, center, sigma, and alpha.
        """
        self.model.kernel.sigma.data =sigma.data.cuda()
        field_volume = self.model.kernel.mmv(self.feature_volume, center.clone().cuda(), 
                                        alpha.clone().cuda(),
                                        opt=self.model.flk_opt).squeeze().view( (resolution,) * 3)
        mesh_kernel = rec.run_mc (field_volume, threshold = 0.5, resolution  =resolution,  bounds= (-0.5, 0.5),mc_device = 'cuda' )
        cd = meshevaluator.eval_mesh( scale_fn (mesh_kernel), pointcloud, None, onet_samples=None)['chamfer-L1']
        #cd_val = meshevaluator.eval_mesh( mesh_kernel, input_points, None, onet_samples=None)['chamfer-L1']
        return  cd, center, sigma,alpha
    def eval_epochs(self, alphas, centers, sigmas,resolution, rec, meshevaluator,input_points, pointcloud,scale_fn = lambda x:x, verbose = True):
        """
        Evaluates the performance of the model over multiple epochs.

        Parameters:
            alphas (list): A list of alpha values.
            centers (list): A list of center values.
            sigmas (list): A list of sigma values.
            resolution (int): The resolution of the kernel.
            rec (Reconstructor): The reconstruction method.
            meshevaluator (MeshEvaluator): The mesh evaluator.
            input_points (torch.Tensor): The input points.
            pointcloud (torch.Tensor): The point cloud.
            scale_fn (function, optional): The scaling function. Defaults to lambda x:x.
            verbose (bool, optional): Whether to print the results. Defaults to True.

        Returns:
            tuple: A tuple containing the minimum chamfer distance on the ground truth, 
                   the minimum chamfer distance on the validation set, and a dictionary 
                   containing the best alpha, center, and sigma values.
        """
        # with torch.inference_mode():
        #     start_time = time.time()
        #     with ThreadPoolExecutor(max_workers=15) as executor:
        #         futures = []
        #         for alpha, center, sigma in zip(alphas, centers, sigmas):
        #             futures.append(executor.submit(self.evaluate_kernel,  center, alpha, sigma, resolution, rec, meshevaluator,input_points))
        #         min_cd_val = float('inf')
        #         min_cd_gt = None
        #         best_alpha, best_center, best_sigma = None, None, None
        #         for f in as_completed(futures):
        #             cd_val, center, sigma, alpha = f.result()
        #             if cd_val < min_cd_val:
        #                 min_cd_val = cd_val
        #                 best_alpha, best_center, best_sigma = alpha, center, sigma
        #         min_cd_gt , _,_,_ = self.evaluate_kernel(  best_center, best_alpha, best_sigma, resolution, rec, meshevaluator, pointcloud)
        #             #if verbose:
        #                 #print(f" sigma: {sigma.mean().item():.4f}, CD (GT): {cd_gt:.4f}, CD (Val): {cd_val:.4f}" ) #, print(f" sigma: {sigma.mean().item():.4f}, CD (Val): {cd_val:.4f}" )
        # if verbose:
        #     print(f" CD (GT): {min_cd_gt:.4f},CD (Val): {min_cd_val:.4f}" )
        # print("\033[91mThreadPoolExecutor:\033[0m :", time.time() - start_time)
        with torch.inference_mode():
            start_time = time.time()
            # Evaluate the performance of each hyperparameter combination
            futures = []
            for alpha, center, sigma in zip(alphas, centers, sigmas):
                futures.append(self.evaluate_kernel(  center, alpha, sigma, resolution, rec, meshevaluator,input_points,scale_fn) )
            # Keep track of the best combination
            min_cd_val = float('inf')
            min_cd_gt = None
            best_alpha, best_center, best_sigma = None, None, None
            for f in futures:
                cd_val, center, sigma, alpha = f
                print(f" sigma: {sigma.mean().item():.4f}, CD (Val): {cd_val:.4f}" )
                if cd_val < min_cd_val:
                    min_cd_val = cd_val
                    best_alpha, best_center, best_sigma = alpha, center, sigma
            # Evaluate the best combination on the ground truth
            min_cd_gt , _,_,_ = self.evaluate_kernel(  best_center, best_alpha, best_sigma, resolution, rec, meshevaluator, pointcloud,scale_fn)
        if verbose:
            print(f" CD (GT): {min_cd_gt:.4f},CD (Val): {min_cd_val:.4f}" )
        print("\033[1m\033[91mTime taken in  Evaluation:\033[0m :", time.time() - start_time)
        return min_cd_gt, min_cd_val, {"alpha": best_alpha, "center": best_center, "sigma": best_sigma}

def create_optimizer(opt_type: str, model: torch.nn.Module, learning_rate: float):
        """
        Creates an optimizer based on the specified type and model parameters.

        Parameters:
        opt_type (str): The type of optimizer to create. Can be "adam", "sgd", "lbfgs", or "rmsprop".
        model (torch.nn.Module): The model for which the optimizer is being created.
        learning_rate (float): The learning rate for the optimizer.

        Returns:
        opt_hp (torch.optim.Optimizer): The created optimizer.
        schedule (torch.optim.lr_scheduler._LRScheduler): The learning rate schedule for the optimizer.
        """
        center_lr_div = 1
        schedule = None
        named_params = dict(model.named_parameters())
        print("Creating optimizer with the following parameters:")
        for k, v in named_params.items():
            print(f"\t{k} : {v.shape}")
        if opt_type == "adam":
            if "penalty" not in named_params:
                opt_modules = [{"params": named_params.values(), "lr": learning_rate}]
            else:
                opt_modules = []
                if "sigma" in named_params:
                    opt_modules.append({"params": named_params["sigma"], "lr": learning_rate})
                if "penalty" in named_params:
                    opt_modules.append({"params": named_params["penalty"], "lr": learning_rate})
                if "centers" in named_params:
                    opt_modules.append({"params": named_params["centers"], "lr": learning_rate / center_lr_div})
            opt_hp = torch.optim.AdamW(opt_modules)#betas=(0.6, 0.99))
            # schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_hp, factor=0.5, patience=1)
            # schedule = torch.optim.lr_scheduler.MultiStepLR(opt_hp, [2, 10, 40], gamma=0.5)
            schedule = torch.optim.lr_scheduler.StepLR(opt_hp, 20, gamma=0.1)
        elif opt_type == "sgd":
            opt_hp = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif opt_type == "lbfgs":
            #if model.losses_are_grads:
                #raise ValueError("L-BFGS not valid for model %s" % (model))
            opt_hp = opt_hp = torch.optim.LBFGS(
                model.parameters(), 
                lr = 0.1,  # Learning rate
                max_iter=5,            # Maximum number of iterations per optimization step
                max_eval=None,          # Maximum number of function evaluations (if None, set to max_iter * 1.25)
                tolerance_grad=1e-7,    # Termination tolerance on first-order optimality (gradient norm)
                tolerance_change=1e-9,  # Termination tolerance on function value/parameter changes
                history_size=100,       # Number of previous updates to store (for approximating the inverse Hessian)
                line_search_fn='strong_wolfe'     # Line search algorithm ('strong_wolfe' is the only option, None for default)
            )
        elif opt_type == "rmsprop":
            opt_hp = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Optimizer type %s not recognized" % (opt_type))

        return opt_hp, schedule
     