
import math
import numpy as np
import torch
import pykeops
import numpy as np
from pykeops.torch import LazyTensor
import os
from pykeops.torch import KernelSolve,Genred

os.environ['CUDA_PATH']= "/usr/local/cuda-11.7"


import numpy as np
from sklearn.utils import check_random_state

from math import sqrt
from scipy.linalg import qr_multiply
from scipy.stats import chi
import warnings
import torch
from falkon.center_selection import UniformSelector

from collections.abc import Mapping
from falkon import Falkon, kernels,FalkonOptions

def KernelSolver(args, X_nystrom=None):
    """
    Returns a solver object based on the provided solver type.

    Parameters:
    args (object): An object containing the solver type and other solver-specific parameters.
    X_nystrom (object, optional): An optional object used for some solver types. Defaults to None.

    Returns:
    object: A solver object of the specified type.
    """
    if args.solver == 'KRR':
        return KRRSolver(**{
            'kernel': GaussianKernel(sigma=args.sigma),
            'penalty': args.penalty
        })
    
    elif args.solver == 'KRRG':
        return KRRGenredSolver(**{
            "sigma": args.sigma,
            "dim": 32,
            'penalty': args.penalty
        })
    
    elif args.solver == 'KRRFalkon':
        return KRRFalkonSolver(**{
            'kernel': kernels.GaussianKernel(sigma=args.sigma),
            'penalty': args.penalty
        })
    
    elif args.solver == 'NKRRFalkon':
        return NKRRFalkonSolver(**{
            'kernel': kernels.GaussianKernel(sigma=args.sigma),
            'penalty': args.penalty,
            'n_nystrom': args.n_nystrom
        })
    
    elif args.solver == 'NKRR':
        return NKRRSolver(**{
            'kernel': kernels.GaussianKernel(sigma=args.sigma),
            'penalty': args.penalty,
            'centers_init': X_nystrom.cuda() if X_nystrom is not None else None
        })
    
    elif args.solver == 'NKRRC':
        return NKRRSolverChol(**{
            'kernel': GaussianKerneLTorch(sigma=args.sigma),
            'penalty': args.penalty,
            'centers_init': X_nystrom.cuda() if X_nystrom is not None else None
        })
    
    elif args.solver == 'RFF':
        return RFFSolver(**{
            'sigma': args.sigma,
            'penalty': args.penalty,
            'rff_dim': args.n_rff
        })
    
    else:
        raise ValueError(f"Unknown solver type: {args.solver}")


def _get_random_matrix(distribution):
    return lambda rng, size:  rng.randn(*size)
def gaussian_kernel_keops(x, y, sigma=0.1):
    x_i = LazyTensor(x[:, None, :])  # (M, 1, 1)
    y_j = LazyTensor(y[None, :, :])  # (1, N, 1)
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
    return (-D_ij / (2 * sigma**2)).exp()    

# Inner function to be compiled:
def _gaussianconv_pytorch(x, y, sigma = 0.1):
    """(B,N,D), (B,N,D), (B,N,1) -> (B,N,1)"""
    # Note that cdist is not currently supported by torch.compile with dynamic=True.

    D_xx = (x * x).sum(-1).unsqueeze(1)  # (N,1)
    D_yy = (y * y).sum(-1).unsqueeze(0)  # (1,M)
    #D_xy = torch.matmul(x, y.T)  # (N,D) @ (B,D,M) = (N,M)
    D_xy = torch.addmm( D_xx+ D_yy ,x, y.T , alpha = -2  )
    #D_xy = D_xx - 2 * D_xy + D_yy  # (B,N,M)
    K_xy = (-D_xy/(2 * sigma**2)).exp()  # (B,N,M)

    return K_xy  # (B,N,1)

 
# Compile the function:
gaussianconv_pytorch_compiled = torch.compile(_gaussianconv_pytorch, dynamic=True)
class GaussianKerneLTorch:
    def __init__(self, sigma = 0.1):
        self.sigma = sigma
    def __call__(self,x,y):
        return gaussianconv_pytorch_compiled (x, y, sigma=self.sigma )

#def GaussianKerneLTorch(sigma = 0.1): return lambda x,y : gaussianconv_pytorch_compiled (x, y, sigma=sigma)

# Wrap it to ignore optional keyword arguments:
def gaussianconv_pytorch_dynamic(x, y, b, **kwargs):
    return gaussianconv_pytorch_compiled(x, y, b)
def GaussianKernel(sigma = 0.1): return lambda x,y : gaussian_kernel_keops(x, y, sigma=sigma)


def create_random_fourier_features(X, W, offset):
    """
    Creates Gaussian random features using PyTorch, with updated input shapes.
    Parameters:
    - X: The input data, PyTorch tensor (d, N).
    - D: The number of random features to generate.
    - sigma: The standard deviation of the Gaussian distribution.    
    Returns:
    - Z: The generated random Fourier features, PyTorch tensor.
    """
    # sample weights from a Gaussian distribution
    #W = torch.randn(X.shape[1], D) * sigma
    D = W.shape[1]
    # Z = torch.sqrt(torch.tensor(1/D)) * torch.cat([torch.sin(torch.mm(X, W)), torch.cos(torch.mm(X, W))], dim=1)
    Z = torch.mm(X, W.T)
    Z.add_(offset).cos_().mul_(torch.sqrt(torch.tensor(2/D)) )
    
    return Z
def generate_orthogonal_matrix(d, D, device, sigma,use_offset = True,seed = None):

    random_state = check_random_state(seed)
    n_features, n_components_ = d, D
    n_stacks = int(np.ceil(n_components_/n_features))
    n_components = n_stacks * n_features
    if n_components != n_components_:
        msg = "n_components is changed from {0} to {1}.".format(
                D, n_components
        )
        msg += " You should set D to an n-tuple of n_features."
        warnings.warn(msg)
        n_components_ = n_components

    if  not use_offset:
        n_stacks = int(np.ceil(n_stacks / 2))
        n_components = n_stacks*n_features
        if n_components*2 != n_components_:
            msg = "n_components is changed from {0} to {1}.".format(
                n_components_, n_components*2
            )
            msg += " When random_fourier=True and use_offset=False, "
            msg += " n_components should be larger than 2*n_features."
            warnings.warn(msg)
            n_components_ = n_components * 2

    if sigma == 'auto':
        gamma = 1.0 / n_features
    else:
        gamma = sigma
    size = (n_features, n_features)

    distribution = _get_random_matrix('gaussian')
    
    random_weights_ = []
    for _ in range(n_stacks):
        W = distribution(random_state, size)
        S = np.diag(chi.rvs(df=n_features, size=n_features,
                            random_state=random_state))
        SQ, _ = qr_multiply(W, S)
        random_weights_ += [SQ]

    random_weights_ = np.vstack(random_weights_)#.T
    random_offset_ = np.zeros(n_components)
    random_weights_ *= sqrt(2*gamma)
    if use_offset:
        random_offset_ = random_state.uniform(
                0, 2*np.pi, size=n_components
            )
    return torch.from_numpy(random_weights_).to(device) .float(),torch.from_numpy(random_offset_).to(device).float()



def cholesky(M, upper=False, check_errors=True):
    if upper:
        U, info = torch.linalg.cholesky_ex(M.transpose(-2, -1).conj())
        if check_errors:
            if info > 0:
                raise RuntimeError("Cholesky failed on row %d" % (info))
        return U.transpose(-2, -1).conj()
    else:
        L, info = torch.linalg.cholesky_ex(M, check_errors=False)
        if check_errors:
            if info > 0:
                raise RuntimeError("Cholesky failed on row %d" % (info))
        return L


def jittering_cholesky(mat, upper=False):
    eye = torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
    epsilons = [1e-9, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    last_exception = None
    for eps in epsilons:
        try:
            return cholesky(mat + eye * eps, upper=upper, check_errors=True)
        except RuntimeError as e:  # noqa: PERF203
            last_exception = e
    raise last_exception

class NKRRSolverChol(torch.nn.Module):
    
    """
    Cholesky decomposition of kmm
    L @ L.T = kmm 
    torch.triangular_solve L, K_mn
    A = L^{-1} K_mn  
    AAT = A @ A.T = L^{-1} K_mn @ K_nm L.T ^{-1}
    B = A @ A.T + I = L^{-1} K_mn @ K_nm L.T ^{-1} + I = LB @ LB.T
    AY = A @ Y / sqrt_var  = L^{-1} K_mn Y
    torch.triangular_solve LB AY
    C = LB^{-1} AY =  LB^{-1} L^{-1} K_mn Y       
    torch.triangular_solve LB.T C                                                                                           
    T1 = B_{-1} AY= LB.T^{-1} C 
       =  LB.T^{-1} LB^{-1} L^{-1} K_mn Y
       = B^{-1} L^{-1} K_mn Y 
       = (L^{-1} K_mn @ K_nm L.T ^{-1} + I)^{-1}  L^{-1} K_mn Y
       = (K_mn @ K_nm L.T ^{-1} + L )^{-1}  K_mn Y 
                                    
    alpha  = L.T^{-1} (K_mn @ K_nm L.T ^{-1} + L )^{-1}  K_mn Y
           = (K_mn @ K_nm  + L L.T )^{-1} K_mn Y  = (K_mn @ K_nm  + kmm )^{-1} K_mn Y 

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        kernel,
        centers_init: torch.Tensor,
        penalty: torch.Tensor,
        centers_transform  = None,
        pen_transform = None,
    ):
        super().__init__()
        self.kernel = kernel
        self.centers = centers_init
        self.penalty = penalty
        self.centers_transform = centers_transform
        self.pen_transform = pen_transform
        #self.x_train, self.y_train = None, None
        self.losses = None
        self.params = None
    @torch.no_grad()
    def fit(self, X,Y):
        variance = self.penalty * X.shape[0]
        sqrt_var = math.sqrt(variance)
        kmn = self.kernel(self.centers, X)
        kmm = self.kernel(self.centers, self.centers)

        L = jittering_cholesky(kmm)
        A = torch.linalg.solve_triangular(L, kmn, upper=False)
        AAT = A @ A.T  # m*n @ n*m = m*m in O(n * m^2), equivalent to kmn @ knm.
        # B = A @ A.T + I
        #B = AAT / variance + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype) 
        B = AAT   + variance *torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        LB = jittering_cholesky(B)  # LB @ LB.T = B
        ## you need to divide by sqrt_var here otherwise it doesn't work
        AY = A @ Y /sqrt_var # m*1
        c = torch.linalg.solve_triangular(LB, AY, upper=False)  /sqrt_var# m * p
        #self.params = (L, A, AAT, LB, c)
        self.params = (L, LB, c)

        return self.params
    @torch.no_grad()
    def forward(self, X):

        L, LB, c = self.params
        tmp1 = torch.linalg.solve_triangular(LB.T, c, upper=True)
        tmp2 = torch.linalg.solve_triangular(L.T, tmp1, upper=True)
        kms = self.kernel( X,self.centers)
        return kms @ tmp2

class NKRRSolver(torch.nn.Module):
    
    """
    Cholesky decomposition of kmm
    L @ L.T = kmm 
    torch.triangular_solve L, K_mn
    A = L^{-1} K_mn  
    AAT = A @ A.T = L^{-1} K_mn @ K_nm L.T ^{-1}
    B = A @ A.T + I = L^{-1} K_mn @ K_nm L.T ^{-1} + I = LB @ LB.T
    AY = A @ Y / sqrt_var  = L^{-1} K_mn Y
    torch.triangular_solve LB AY
    C = LB^{-1} AY =  LB^{-1} L^{-1} K_mn Y       
    torch.triangular_solve LB.T C                                                                                           
    T1 = B_{-1} AY= LB.T^{-1} C 
       =  LB.T^{-1} LB^{-1} L^{-1} K_mn Y
       = B^{-1} L^{-1} K_mn Y 
       = (L^{-1} K_mn @ K_nm L.T ^{-1} + I)^{-1}  L^{-1} K_mn Y
       = (K_mn @ K_nm L.T ^{-1} + L )^{-1}  K_mn Y 
                                    
    alpha  = L.T^{-1} (K_mn @ K_nm L.T ^{-1} + L )^{-1}  K_mn Y
           = (K_mn @ K_nm  + L L.T )^{-1} K_mn Y  = (K_mn @ K_nm  + kmm )^{-1} K_mn Y 

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        kernel,
        centers_init: torch.Tensor,
        penalty: torch.Tensor,
        centers_transform  = None,
        pen_transform = None,
    ):
        super().__init__()
        self.kernel = kernel
        self.centers = centers_init
        self.penalty = penalty
        self.centers_transform = centers_transform
        self.pen_transform = pen_transform
        #self.x_train, self.y_train = None, None
        self.losses = None
        self.params = None
        self.dim = 32
        formula = "Exp(SqDist(x / g, y / g) * IntInv(-2)) * b"

        #formula = "Exp(-  SqDist(x,y)/g) * b"
        self.kernel.sigma.data =self.kernel.sigma.cuda().view(-1,1) if isinstance (self.kernel.sigma, torch.Tensor) else torch.tensor([self.kernel.sigma], device = 'cuda').view(-1,1)

        sigma_dim = self.kernel.sigma.shape[0]
        aliases = [
            "x = Vi(" + str(self.dim) + ")",  # First arg:  i-variable of size D
            "y = Vj(" + str(self.dim) + ")",  # Second arg: j-variable of size D
            "b = Vj(" + str(1) + ")",  # Third arg:  j-variable of size Dv
            f"g = Pm({sigma_dim})",
        ] 
        self.predict = Genred(formula, aliases, axis = 1)
    @torch.no_grad()
    def fit(self, X,Y):
        variance = self.penalty * X.shape[0]
        sqrt_var = math.sqrt(variance)
        kmn = self.kernel(self.centers, X)
        knm = kmn.T
        kmm = self.kernel(self.centers, self.centers)
        k = (kmn @ knm +  variance*kmm ) 
        k = k+ variance*torch.eye(k.shape[0], device=X.device, dtype=X.dtype) #/variance
        #k = self.kernel(self.centers, X)/variance + torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
        #k = LazyTensor(k, axis =1)
        #print(Y.shape)
        b = (kmn).squeeze()
        #alphas = k.solve(LazyTensor(b, axis =1), alpha=0)
        alphas = torch.linalg.solve(k,b )@Y  #/sqrt_var

    
        self.params = alphas

        return self.params
    def loss(self, X,Y):
        variance = self.penalty * X.shape[0]
        sqrt_var = math.sqrt(variance)
        kmn = self.kernel(self.centers, X)
        knm = kmn.T
        kmm = self.kernel(self.centers, self.centers)
        k = (kmn @ knm +  variance*kmm ) 
        k = k+ variance*torch.eye(k.shape[0], device=X.device, dtype=X.dtype) #/variance
        #k = self.kernel(self.centers, X)/variance + torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
        #k = LazyTensor(k, axis =1)
        #print(Y.shape)
        b = (kmn).squeeze()
        k_inv_kmn = torch.linalg.solve(k,b )
        alphas_y = k_inv_kmn@Y  #/sqrt_var
        knm_k_inv_kmn = knm@k_inv_kmn
        ndeff = knm_k_inv_kmn.trace()
        #datafit = torch.square(Y).sum() - Y.squeeze().T@knm_k_inv_kmn@Y.squeeze()
        datafit = (knm @ alphas_y- Y ).pow(2).mean() +  self.penalty  *alphas_y.T @kmm@alphas_y
  
        Kdiag = self.kernel(X, X,diag=True).sum()
        kmm_cond = kmm + 1e-4*torch.eye(kmm.shape[0], device=X.device, dtype=X.dtype)
        trace = Kdiag - (knm@torch.linalg.solve(kmm_cond,kmn ) ).trace()
        trace = trace * datafit / (variance)
        self.params = alphas_y
        #print(ndeff.item(),datafit.item(),trace.item())
        return datafit + ndeff +trace #+ trace
    @torch.no_grad()
    def forward(self, X):

        alphas = self.params
        sigma =self.kernel.sigma.cuda() if isinstance (self.kernel.sigma, torch.Tensor) else torch.tensor([self.kernel.sigma], device = 'cuda')

        
        # #print(kms)
        #kms = self.kernel( X,self.centers)
        #kms@alphas
        #self.predict(X,self.centers, alphas,sigma )
        return self.predict(X,self.centers, alphas,sigma )
class NKRRFalkonSolver(torch.nn.Module):
    
    """
    

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        kernel,
        penalty: torch.Tensor,
        n_nystrom = None,
    ):
        super().__init__()
        self.kernel = kernel
        self.kernel.sigma.data =self.kernel.sigma.cuda().view(-1,1) if isinstance (self.kernel.sigma, torch.Tensor) else torch.tensor([self.kernel.sigma], device = 'cuda').view(-1,1)
        print('shape', self.kernel.sigma.shape)
        self.penalty = penalty
        self.nystrom = n_nystrom
        #self.x_train, self.y_train = None, None
        self.losses = None
        self.params = None
    @torch.no_grad()
    def fit(self, X,Y):
        # self.train = X
        # variance = self.penalty * X.shape[0]
        # sqrt_var = math.sqrt(variance)
        # knn = self.kernel(X, X)
        # knn_conditioned = knn / variance + torch.eye(knn.shape[0], device=X.device, dtype=X.dtype) 
        # #self.params  = knn.solve(Y, alpha=variance)
        # #self.params  = knn_conditioned.solve(Y, alpha=1)
        # self.params = torch.linalg.solve(knn_conditioned,Y)  #/sqrt_var
        self.falkonsolver = Falkon(kernel=self.kernel, penalty=self.penalty, M=self.nystrom,options = FalkonOptions( use_cpu=not torch.cuda.is_available()) )
        self.falkonsolver.fit(X.cpu(),Y.cpu())
        return self.params
    @torch.no_grad()
    def forward(self, X):

        # alphas = self.params
    
        # kms = self.kernel(X,self.train)
        # #print(kms)
        # return kms@ alphas
        output = self.falkonsolver.predict(X)

        return output
class KRRFalkonSolver(torch.nn.Module):
    
    """
    

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        kernel,
        penalty: torch.Tensor,
    ):
        super().__init__()
        self.kernel = kernel
        self.penalty = penalty

        #self.x_train, self.y_train = None, None
        self.losses = None
        self.params = None
    @torch.no_grad()
    def fit(self, X,Y):
        self.train = X
        variance = self.penalty * X.shape[0]
        sqrt_var = math.sqrt(variance)
        knn = self.kernel(X, X)
        knn_conditioned = knn / variance + torch.eye(knn.shape[0], device=X.device, dtype=X.dtype) 
        #self.params  = knn.solve(Y, alpha=variance)
        #self.params  = knn_conditioned.solve(Y, alpha=1)
        self.params = torch.linalg.solve(knn_conditioned,Y)  #/sqrt_var
        
        return self.params
    @torch.no_grad()
    def forward(self, X):

        alphas = self.params
    
        kms = self.kernel(X,self.train)
        #print(kms)
        return kms@ alphas

class KRRSolver(torch.nn.Module):
    
    """
    

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        kernel,
        penalty: torch.Tensor,
    ):
        super().__init__()
        self.kernel = kernel
        self.penalty = penalty

        #self.x_train, self.y_train = None, None
        self.losses = None
        self.params = None
    @torch.no_grad()
    def fit(self, X,Y):
        self.train = X
        variance = self.penalty * X.shape[0]
        sqrt_var = math.sqrt(variance)
        knn = self.kernel(X, X)
        #knn_conditioned = knn / variance #+ LazyTensor (torch.eye(knn.shape[0], device=X.device, dtype=X.dtype) , axis =1)
        self.params  = knn.solve(Y, alpha=variance)
        #self.params  = knn_conditioned.solve(Y, alpha=1)
        #self.params = torch.linalg.solve(knn_conditioned)  #/sqrt_var

        return self.params
    @torch.no_grad()
    def forward(self, X):

        alphas = self.params
    
        kms = self.kernel(X,self.train)
        #print(kms)
        return kms@ alphas
class KRRGenredSolver(torch.nn.Module):
    
    """
    

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,

        penalty: torch.Tensor,
        sigma = 1,
        dim = 32
    ):
        super().__init__()
        self.penalty = penalty

        #self.x_train, self.y_train = None, None
        self.losses = None
        self.sigma =2*sigma.cuda()**2 if isinstance (sigma, torch.Tensor) else torch.tensor([2*sigma**2], device = 'cuda')
        self.dim = dim
        self.params = None
        formula = "Exp(-  SqDist(x,y)/g) * b"
        aliases = [
            "x = Vi(" + str(self.dim) + ")",  # First arg:  i-variable of size D
            "y = Vj(" + str(self.dim) + ")",  # Second arg: j-variable of size D
            "b = Vj(" + str(1) + ")",  # Third arg:  j-variable of size Dv
            "g = Pm(1)",
        ] 
        self.Kinv = KernelSolve(formula, aliases, "b", axis=1)
        self.predict = Genred(formula, aliases, axis = 1)
    @torch.no_grad()
    def fit(self, X,Y):
        self.train = X
        variance = self.penalty * X.shape[0]
        sqrt_var = math.sqrt(variance)

        #knn = self.kernel(X, X)
        #knn_conditioned = knn / variance #+ LazyTensor (torch.eye(knn.shape[0], device=X.device, dtype=X.dtype) , axis =1)
        self.params  = self.Kinv(X, X, Y, self.sigma,  alpha=variance)
        #self.params  = knn_conditioned.solve(Y, alpha=1)
        #self.params = torch.linalg.solve(knn_conditioned)  #/sqrt_var

        return self.params
    @torch.no_grad()
    def forward(self, X):

        alphas = self.params
        return self.predict(X,self.train, alphas,self.sigma )


class RFFSolver(torch.nn.Module):
    
    def __init__(
        self,
        penalty: torch.Tensor,
        rff_dim :int,
        sigma : float
    ):
        super().__init__()
        self.penalty = penalty
        self.rff_dim = rff_dim
        self.sigma = sigma
        #self.x_train, self.y_train = None, None
        self.losses = None
        self.params = None
    @torch.no_grad()
    def fit(self, X,Y):
        rff_weights = generate_orthogonal_matrix(X.shape[-1], self.rff_dim, "cuda" ,sigma = self.sigma,use_offset = True,seed = None)
        self.train = create_random_fourier_features(X, rff_weights[0], rff_weights[1])
  
        variance = self.penalty * X.shape[0]
        sqrt_var = math.sqrt(variance)

        lmbda_cov =  self.penalty # 0.08
        # XT = self.train.T.contiguous()
        # X_i = LazyTensor(XT[:, None, :])  # (D, 1, N)
        # X_j = LazyTensor(XT[None, :, :])  
        # center_cov =  (X_i * X_j).sum(dim=-1) 
        # X_ii = LazyTensor(XT[:, None, :]) 
        # trace = (X_ii * X_ii).sum(dim=2) .sum()*lmbda_cov
        # print(trace)

        center_cov = self.train.T@self.train
        cov_inv = self.train.shape[1] * torch.linalg.inv((self.train.shape[0] - 1) * center_cov + lmbda_cov*center_cov.trace() * torch.eye(self.train.shape[1]).cuda() )    
        self.params =  cov_inv@ self.train.T @Y, rff_weights
        #center_cov = center_cov#/variance
        #XT_Y =  self.train.T @Y

        #self.params= self.train.shape[1] *  center_cov.solve(XT_Y ,alpha=1),rff_weights 
       ## self.params=  center_cov.solve(XT_Y ,alpha=self.penalty ),rff_weights 
        #knn = self.kernel(X, X)
        #knn_conditioned = knn / variance #+ LazyTensor (torch.eye(knn.shape[0], device=X.device, dtype=X.dtype) , axis =1)
        #self.params  = knn.solve(Y, alpha=variance)
        #self.params  = knn_conditioned.solve(Y, alpha=1)
        #self.params  = center_cov.solve(self.train.T @Y, alpha=1)
        #self.params = torch.linalg.solve(knn_conditioned)  #/sqrt_var

        return self.params
    @torch.no_grad()
    def forward(self, X):
        #print(X.shape)
        mu_rff, rff_weights = self.params
        x_rff = create_random_fourier_features(X, rff_weights[0], rff_weights[1])
        #test_feature_rff = feature_encoder.forward(test_feature) 
        #print(logits_text.shape, test_feature_rff.shape, mu_rff.shape)
        output =x_rff @ mu_rff  
        #print(output.shape)
        return output
    
class Selector:
    def __init__(self, rng, num_centers):
        self.selector = UniformSelector(rng, num_centers)
    def select( X, *args, **kwargs):
        selector = UniformSelector(np.random.default_rng(100), 1000)
        centers = selector.select(X,Y =None)
        return centers
def choose_centers_init(Xtr,  num_centers, seed) -> torch.Tensor:
    selector = UniformSelector(np.random.default_rng(seed), num_centers)
    centers = selector.select(Xtr, None)
    # noinspection PyTypeChecker
    return centers  






#  flk_opt = FalkonOptions( use_cpu=not torch.cuda.is_available() ,chol_force_ooc=True
#                                 # cg_full_gradient_every=10, cg_epsilon_32=1e-6,cg_tolerance=1e-7,
#                                 # cg_differential_convergence=True,))
#         )
        

            


    