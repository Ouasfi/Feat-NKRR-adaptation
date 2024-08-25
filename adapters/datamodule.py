
import contextlib
import numpy as np
import torch
from scipy.spatial import cKDTree #as KDTree
import torch.nn.functional as F
import os
class DataModule:
    def __init__(self, field,feat_ctx, feat_fn ):
        """
        Initializes a DataModule instance.

        Args:
            field: The field to be used in the data module.
            feat_ctx: The feature context to be used in the data module.
            feat_fn: The feature function to be used in the data module.

        Returns:
            None
        """
        #super().__init__()
        self.field = field
        self.feat_ctx = feat_ctx
        self.get_features = feat_fn
    def get( self, n_local_queries = 1, **kwargs):
        """
        Retrieves data for training based on the provided query type.

        Args:
            n_local_queries (int, optional): The number of local queries to sample. Defaults to 1.
            **kwargs: Additional keyword arguments to control the query type and other parameters.

        Returns:
            tuple: A tuple containing the sampled data, training features, and training outputs.
        """
        query_type = kwargs.get('query_type', 'gaussian')
        if query_type == 'gaussian':
            data = self.sample_gaussian(self.field.latents,n_local_queries)
        elif query_type == 'levels':
            data = self.sample_levels(self.field,n_local_queries)
        N_input =  data.get('pc', data['pos']).size(2)
        features, outputs, f_dim = self.get_features(self.field, feat_ctx = self.feat_ctx, **kwargs )
        X_train = features.transpose(1,2).reshape(-1,f_dim)
        Y_train = outputs
        Y_train[:, :N_input] = 1e-6
        if  query_type == 'levels':
            Y_train[:, N_input:] = self.field.latents['sdfs']
        Y_train = Y_train.view(-1,1)

        return data, X_train.detach().cpu().contiguous(), Y_train.cpu().detach().contiguous()
    @staticmethod
    def sample_gaussian( data,n_local_queries, query_key ='pos_non_manifold' ):
        """
        Samples data points using a Gaussian distribution.

        Args:
            data (dict): The input data containing 'pc' or 'pos' keys.
            n_local_queries (int): The number of local queries to sample.
            query_key (str, optional): The key to use for storing the sampled points. Defaults to 'pos_non_manifold'.

        Returns:
            dict: The input data with the sampled points added under the specified query key.
        """
        scr = 183965288784846061718375689149290307792 #secrets.randbits(128)
        rng = np.random.default_rng( scr )    
        pos = data.get('pc', data['pos']).squeeze(0).transpose(0,1).cpu().numpy()
        N = pos.shape[0]
        ptree  = cKDTree(pos)
        sigmas = ptree.query(pos,51, workers = 20)[0][:,-1]
        points = np.expand_dims(pos,1)  + 1*np.expand_dims(sigmas,(1,-1)) * rng.normal(0.0, 1.0, size= (N, n_local_queries,3) )
        points = points.reshape(-1,3)
        data[query_key] = torch.from_numpy(points.astype(np.float32)).transpose(0,1).unsqueeze(0)
        return data
    @staticmethod
    def sample_levels( field,n_local_queries, query_key ='pos_non_manifold' ):
        """
        Samples data points using level sets.

        Args:
            field: The input field containing 'pc' or 'pos' keys.
            n_local_queries (int): The number of local queries to sample.
            query_key (str, optional): The key to use for storing the sampled points. Defaults to 'pos_non_manifold'.

        Returns:
            dict: The input data with the sampled points added under the specified query key.
        """
        scr = 183965288784846061718375689149290307792 #secrets.randbits(128)
        rng = np.random.default_rng( scr )    
        pos = field.latents.get('pc', field.latents['pos']).clone()
        N = pos.shape[2]
        pos.requires_grad = True#.squeeze(0).transpose(0,1).cpu().numpy()
        outputs = field(pos)
        grads = torch.autograd.grad(outputs, pos, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
        normals = F.normalize(grads, p=2, dim=1)

        pos_np  = pos.data.squeeze(0).transpose(0,1).cpu().numpy()
        ptree  = cKDTree(pos_np)
        sigmas = ptree.query(pos_np,51, workers = 20)[0][:,-1]
        scale= sigmas.mean()
        sdfs = scale/51* (2*torch.rand(n_local_queries,1,N)-1).to(pos.device)
        queries = pos.detach() + sdfs * normals.detach()
        queries = queries.transpose(1,2).view(-1,3)
        field.latents[query_key] =  queries.transpose(0,1).unsqueeze(0) #torch.from_numpy(points.astype(np.float32)).transpose(0,1).unsqueeze(0)
        sdfs = sdfs.squeeze(1).flatten()
        field.latents['sdfs'] = sdfs
        data = field.latents
        return data
    @staticmethod
    def subsample_queries(data, points_uniform,n_queries,query_key ='pos_non_manifold'):
        idx = np.random.randint(points_uniform.shape[0], size = n_queries )
        data['idx'] = idx
        points = points_uniform[idx,:]
        data[query_key] = torch.from_numpy(points.astype(np.float32)).transpose(0,1).unsqueeze(0)
        return data


def bilateral_filter(pointcloud, normals, sigma_spatial, sigma_normal):
    # Compute pairwise distances
    dists = torch.norm(pointcloud[:, None, :] - pointcloud[None, :, :], dim=2)
    
    # Compute the weights for each point
    weights = torch.exp(-((normals[None, :, :] - normals[:, None, :])**2).sum(dim=2) / (2 * sigma_normal**2)) * torch.exp(-(dists**2) / (2 * sigma_spatial**2))
    
    # Compute the average normal for each point
    smoothed_normals = torch.sum(normals[None, :, :] * weights[:, :, None], dim=1) / torch.sum(weights, dim=1)[:, None]
    
    return smoothed_normals
@contextlib.contextmanager
def POCO_Feature_Context(m,normalize = False):
    """
    Context manager for extracting features from a POCO model.

    Parameters:
        m (nn.Module): The POCO model to extract features from.
        normalize (bool, optional): Whether to normalize the features. Defaults to False.

    Yields:
        list: A list of extracted features.
    """
    
    # Extracts features from a POCO model and yields them.
    # The features are extracted from the decoder's fc8 layer.

    f_i = []  # List to store the extracted features.

    def get_f_i(m):
        """
        Defines a hook function that appends the input feature to f_i.

        Parameters:
            m (nn.Module): The module from which the input is coming from.
            input (torch.Tensor): The input tensor.
            output (torch.Tensor): The output tensor.

        Returns:
            hook (function): The hook function.
        """
        def hook(m, input, output):
            """
            Appends the input feature to f_i.

            The input feature is normalized if the normalize parameter is True.

            Parameters:
                m (nn.Module): The module from which the input is coming from.
                input (torch.Tensor): The input tensor.
                output (torch.Tensor): The output tensor.
            """
            feature = input if not normalize else (normalize_features(input[0]), )  # Normalize the input feature if normalize is True.
            f_i.append(feature)  # Append the feature to f_i.
        return hook

    # Register the hook function to the decoder's fc8 layer.
    m.decoder.fc8.register_forward_hook(get_f_i(m))

    # Yield the f_i list, which contains the extracted features.
    yield f_i

### Convonet
@contextlib.contextmanager
def CONVONET_Feature_Context(m, normalize = False):
    """
    Context manager for extracting features from a CONVONET model.

    Parameters:
        m (nn.Module): The CONVONET model to extract features from.
        normalize (bool, optional): Whether to normalize the features. Defaults to False.

    Yields:
        list: A list of extracted features.

    Example:
        with CONVONET_Feature_Context(model, normalize=True) as features:
            # Perform some operations on the model
            # ...
            # Access the extracted features
            print(features)
    """
    
    f_i = []
    def get_f_i(m):
        def hook(m, input, output):
            feature = input[0].transpose(1, 2) if not normalize else normalize_features(input[0].transpose(1, 2))
            f_i.append((feature,))
        return hook
    m.decoder.fc_c[0].register_forward_hook(get_f_i(m))

    yield f_i
def compute_features(field,feat_ctx, query_key = 'pos_non_manifold',  **kwargs ):
    """
    Computes features from a given field using a feature context.

    Parameters:
        field: The field from which to compute features.
        feat_ctx: The feature context to use for computing features.
        query_key (str, optional): The key to use for querying the field. Defaults to 'pos_non_manifold'.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: A tuple containing the computed features, the outputs of the field, and the size of the features.
    """
    data = field.latents
    with torch.no_grad():
        inputs = data.get('pc', data['pos'])
        with feat_ctx(field.model) as f_i:
            queries = torch.cat((inputs[0][None].expand(1,-1,-1), data[query_key] .cuda()), dim = 2).cuda()
            #print(queries.shape)
            outputs = field( queries)
    
    features = f_i[0][0] 
    #features  = normalize_features(features)#.contiguous()
    return features, outputs, features.size(1)
def compute_features_convonet(field,feat_ctx, query_key = 'pos_non_manifold',  **kwargs ):
    """
    Computes features from a given field using a feature context, specifically designed for convonet models.

    Parameters:
        field: The field from which to compute features.
        feat_ctx: The feature context to use for computing features.
        query_key (str, optional): The key to use for querying the field. Defaults to 'pos_non_manifold'.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: A tuple containing the computed features, the outputs of the field, and the size of the features.
    """
    data = field.latents
    with torch.no_grad():
        inputs = data.get('pc', data['pos'])
        with feat_ctx(field.model) as f_i:
            queries = torch.cat((inputs[0][None].expand(1,-1,-1), data[query_key] .cuda()), dim = 2).cuda()
            print(queries.shape)
            outputs = field( queries.transpose(1,2))
    
    features = f_i[0][0] 

    return features, outputs, features.size(1)

def normalize_features(features):
    """
    Normalizes a given set of features by subtracting the mean along the second dimension.

    Parameters:
        features (torch.Tensor): The input features to be normalized.

    Returns:
        torch.Tensor: The normalized features.
    """
    #features = F.normalize( (features-features.mean(1).unsqueeze(1)) , dim = 1)
    features = (features-features.mean(1).unsqueeze(1))
    return features
