import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel,InducingPointKernel
from gpytorch.kernels.keops import RBFKernel
from gpytorch.kernels import ScaleKernel, RBFKernel
def get_field_volume(model,likelihood, split_feature_volume,resolution = 128 ):
    model.eval()
    likelihood.eval()
    logits_list = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for split_features in split_feature_volume:
            #print(split_features.shape)
            observed_pred = model(split_features )
            preds = model.likelihood(observed_pred).mean
            logits_list.append(preds.squeeze(0).detach())

    logits = torch.cat(logits_list, dim=0)
    field_volume = logits.squeeze().view( (resolution,) * 3)
    model.train()
    return field_volume

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=32))
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y,x_nystrom, likelihood, sigma = 1):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = RBFKernel(ard_num_dims=32) 
        self.base_covar_module.raw_lengthscale.data = sigma 
        #self.base_covar_module.raw_lengthscale.requires_grad = False
       # self.base_covar_module = LinearKernel()
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=x_nystrom.data.clone().contiguous(), likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    