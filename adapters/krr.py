
import math
import torch
import os
from pykeops.torch import KernelSolve,Genred
from falkon import Falkon, kernels,FalkonOptions

os.environ['CUDA_PATH']= "/usr/local/cuda-11.7"
def KernelSolver(args, X_nystrom=None):
    """
    Returns a solver object based on the provided solver type.

    Parameters:
    args (object): An object containing the solver type and other solver-specific parameters.
    X_nystrom (object, optional): An optional object used for some solver types. Defaults to None.

    Returns:
    object: A solver object of the specified type.
    """

    
    if args.solver == 'KRRG':
        return KRRGenredSolver(**{
            "sigma": args.sigma,
            "dim": 32,
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

    else:
        raise ValueError(f"Unknown solver type: {args.solver}")


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
        """
        Initializes a Gaussian Kernel object.

        Args:
            sigma (float, optional): The standard deviation of the Gaussian kernel. Defaults to 0.1.

        Returns:
            None
        """
        self.sigma = sigma
    def __call__(self,x,y):
        return gaussianconv_pytorch_compiled (x, y, sigma=self.sigma )


class NKRRSolver(torch.nn.Module):
   
    def __init__(
        self,
        kernel,
        centers_init: torch.Tensor,
        penalty: torch.Tensor,
        centers_transform  = None,
        pen_transform = None,
    ):
        """
        Initializes an torch based  Nystrom KRR Solver object.

        Args:
            kernel: The kernel to be used for the NKRR solver.
            centers_init (torch.Tensor): The initial centers for the NKRR solver.
            penalty (torch.Tensor): The penalty term for the NKRR solver.
            centers_transform (optional): The transformation to be applied to the centers. Defaults to None.
            pen_transform (optional): The transformation to be applied to the penalty term. Defaults to None.

        Returns:
            None
        """
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
        """
        Fits the Nystrom KRR solver to the given data by explicitly solving the nystrom system with torch.linalg.solve.

        Parameters:
        X (torch.Tensor): The input data.
        Y (torch.Tensor): The output data.

        Returns:
        torch.Tensor: The fitted parameters of the NKRR solver.
        """
        variance = self.penalty * X.shape[0]
        sqrt_var = math.sqrt(variance)
        kmn = self.kernel(self.centers, X)
        knm = kmn.T
        kmm = self.kernel(self.centers, self.centers)
        k = (kmn @ knm +  variance*kmm ) 
        k = k+ variance*torch.eye(k.shape[0], device=X.device, dtype=X.dtype) #/variance
        b = (kmn).squeeze()
        alphas = torch.linalg.solve(k,b )@Y  #/sqrt_var
        self.params = alphas

        return self.params
    @torch.no_grad()
    def forward(self, X):

        alphas = self.params
        sigma =self.kernel.sigma.cuda() if isinstance (self.kernel.sigma, torch.Tensor) else torch.tensor([self.kernel.sigma], device = 'cuda')
        return self.predict(X,self.centers, alphas,sigma )

class NKRRFalkonSolver(torch.nn.Module):
    

    def __init__(
        self,
        kernel,
        penalty: torch.Tensor,
        n_nystrom = None,
    ):
        """
        Initializes an instance of a Nystreom KRR solver  class.

        Args:
            kernel: The kernel to be used in the solver.
            penalty (torch.Tensor): The penalty parameter for the solver.
            n_nystrom (int, optional): The number of Nyström samples. Defaults to None.

        Returns:
            None
        """
        super().__init__()
        self.kernel = kernel
        self.kernel.sigma.data =self.kernel.sigma.cuda().view(-1,1) if isinstance (self.kernel.sigma, torch.Tensor) else torch.tensor([self.kernel.sigma], device = 'cuda').view(-1,1)
        print('Sigma shape', self.kernel.sigma.shape)
        self.penalty = penalty
        self.nystrom = n_nystrom
        #self.x_train, self.y_train = None, None
        self.losses = None
        self.params = None
    @torch.no_grad()
    def fit(self, X,Y):
        """
        Fits the Nystrom KRR solver using Falkon Preconditionned CG to the given data.

        Args:
            X: The input data.
            Y: The target data.

        Returns:
            The model parameters.
        """
        self.falkonsolver = Falkon(kernel=self.kernel, penalty=self.penalty, M=self.nystrom,options = FalkonOptions( use_cpu=not torch.cuda.is_available()) )
        self.falkonsolver.fit(X.cpu(),Y.cpu())
        return self.params
    @torch.no_grad()
    def forward(self, X):
        output = self.falkonsolver.predict(X)

        return output

class KRRGenredSolver(torch.nn.Module):
    
    def __init__(
        self,

        penalty: torch.Tensor,
        sigma = 1,
        dim = 32
    ):
        """
        Initializes a KRR Solver .

        Args:
            penalty (torch.Tensor): The penalty value for the solver.
            sigma (int, optional): The sigma value for the solver. Defaults to 1.
            dim (int, optional): The dimensionality of the solver. Defaults to 32.

        Returns:
            None
        """
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
        """
        Fits the KRR Solver to the given data using Keops CG algorithm.

        Parameters:
            X (torch.Tensor): The input data.
            Y (torch.Tensor): The target data.

        Returns:
            torch.Tensor: The fitted parameters of the model.
        """
        self.train = X
        variance = self.penalty * X.shape[0]
        sqrt_var = math.sqrt(variance)
        self.params  = self.Kinv(X, X, Y, self.sigma,  alpha=variance)
        return self.params
    @torch.no_grad()
    def forward(self, X):

        alphas = self.params
        return self.predict(X,self.train, alphas,self.sigma )





