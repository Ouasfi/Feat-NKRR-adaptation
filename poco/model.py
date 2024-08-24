import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
from scipy.spatial import cKDTree as KDTree
import   torch_cluster 
from torch_geometric.data import Data
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.transforms import RandomRotate
import math
from math import ceil
import logging
import numpy as np
from knn_cuda import KNN
## functional
def batch_gather(input, dim, index):

    index_shape = list(index.shape)
    input_shape = list(input.shape)

    views = [input.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(input.shape))
    ]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)

    output = torch.gather(input, dim, index)

    # compute final shape
    output_shape = input_shape[0:dim] + index_shape[1:] + input_shape[dim+1:]

    return output.reshape(output_shape)
def max_pool(input: torch.Tensor, indices: list) -> torch.Tensor:
    features = batch_gather(input, dim=2, index=indices).contiguous()
    features = features.max(dim=3)[0]
    return features
def interpolate(x, neighbors_indices, method="mean"):

    mask = (neighbors_indices > -1)
    neighbors_indices[~mask] = 0

    x = batch_gather(x, 2, neighbors_indices)
    if neighbors_indices.shape[-1] > 1:
        if method=="mean":
            return x.mean(-1)
        elif method=="max":
            return x.mean(-1)[0]
    else:
        return x.squeeze(-1)
    
def knn(points, support_points, K, neighbors_indices=None, dists = False):

    if neighbors_indices is not None:
        return neighbors_indices

    if K > points.shape[2]:
        #print(K,points.shape[2])
        K = points.shape[2]
    if len(points.shape)==4:
        points = points.squeeze(0)
    if len(support_points.shape)==4:
        support_points = support_points.squeeze(0)
    pts = points.cpu().detach().transpose(1,2).numpy().copy()
    s_pts = support_points.cpu().detach().transpose(1,2).numpy().copy()
    n = pts.shape[1]
    indices = []
    distances = []
    #print(pts.shape)
    for i in range(pts.shape[0]):
        #print(pts[i].shape, s_pts[i].shape)
        tree = KDTree(pts[i],)
        if dists:
            d, indices_ = tree.query(s_pts[i], k=K,workers=20)
            distances.append(torch.tensor(d, dtype=torch.float))
        else:
            _, indices_ = tree.query(s_pts[i], k=K, workers=10)
        indices.append(torch.tensor(indices_, dtype=torch.long))
    indices = torch.stack(indices, dim=0)

    if K==1:
        indices = indices.unsqueeze(2)
    if dists:
        distances = torch.stack(distances, dim=0).squeeze()
        return distances.to(points.device), indices.to(points.device)
    #print(indices.shape)
    return indices.to(points.device)


def compute_knn_fast(support , queries, k=5, *args, **kwargs):

    knn = KNN(k=k, transpose_mode=True)
    return  knn(support.transpose(1,2), queries.transpose(1,2))[1]



## very slow 
def compute_knn(support , queries, k=5, *args, **kwargs):
    #print(support.shape, queries.shape)
    num_batches_q, dim_q , num_points_q = queries.shape
    # Flatten the data into (num_batches * num_points, dim) shape
    flattened_queries = queries.transpose(1,2).flatten(0,1)
    batch_q = torch.arange(num_batches_q).repeat_interleave(num_points_q)

    num_batches_s, dim_s , num_points_s = support.shape
    # Flatten the data into (num_batches * num_points, dim) shape
    flattened_support = support.transpose(1,2).flatten(0,1)
    batch_s = torch.arange(num_batches_s).repeat_interleave(num_points_s)
    # Compute KNN graph for the flattened data

    knn_indices = torch_cluster.knn(flattened_support, flattened_queries, k, batch_s, batch_q)[1]
    #indices = knn(queries, support, k)
    #print(indices)
    # Reshape the result to match the original batch format
    #assert (knn_indices.view(-1,num_points_q, k)[0,:,:]-torch.arange(num_points_q).cuda().unsqueeze(-1)).sum().item()==0 , 'Bad KNN order'
    #assert (knn_indices.view(-1,num_points_q, k)[1,:,:]-indices).sum().item()==0 , 'Bad KNN order'
    
    return knn_indices.view(num_batches_q, num_points_q, k)


def sampling_quantized(pos, ratio=None, n_support=None, support_points=None, support_points_ids=None):


    if support_points is not None:
        return support_points, support_points_ids

    assert((ratio is None) != (n_support is None))

    if ratio is not None:
        support_point_number = max(1,int(pos.shape[2] * ratio))
    else:
        support_point_number = n_support

    if support_point_number == pos.shape[2]:
        support_points_ids = torch.arange(pos.shape[2], dtype=torch.long, device=pos.device)
        support_points_ids = support_points_ids.unsqueeze(0).expand(pos.shape[0], pos.shape[2])
        return pos, support_points_ids
    elif support_point_number>0 and support_point_number<pos.shape[2]:

        # voxel_size
        maxi, _ = torch.max(pos, dim=2)
        mini, _ = torch.min(pos, dim=2)
        vox_size = (maxi-mini).norm(2, dim=1)/ math.sqrt(support_point_number)

        Rx = RandomRotate(180, axis=0)
        Ry = RandomRotate(180, axis=1)
        Rz = RandomRotate(180, axis=2)

        support_points_ids = []
        for i in range(pos.shape[0]):
            pts = pos[i].clone().transpose(0,1)
            ids = torch.arange(pts.shape[0],device=pos.device)
            sampled_count = 0
            sampled = []
            vox = vox_size[i]
            while(True):
                data = Data(pos=pts)
                data = Rz(Ry(Rx(data)))

                c = voxel_grid(data.pos, batch=torch.zeros(data.pos.shape[0]), size=vox)
                _, perm = consecutive_cluster(c)

                if sampled_count + perm.shape[0] < support_point_number:
                    sampled.append(ids[perm])
                    sampled_count += perm.shape[0]

                    tmp = torch.ones_like(ids)
                    tmp[perm] = 0
                    tmp = (tmp > 0)
                    pts = pts[tmp]
                    ids = ids[tmp]
                    vox = vox / 2
                    # pts = pts[perm]
                    # ids = ids[perm]
                else:
                    n_to_select = support_point_number - sampled_count
                    perm = perm[torch.randperm(perm.shape[0])[:n_to_select]]
                    #print(ids, perm)
                    sampled.append(ids[perm])
                    break
            sampled = torch.cat(sampled)
            support_points_ids.append(sampled)

        support_points_ids = torch.stack(support_points_ids, dim=0)


        support_points_ids = support_points_ids.to(pos.device)

        support_points = batch_gather(pos, dim=2, index=support_points_ids)
        return support_points, support_points_ids
    else:
        raise ValueError(f"Search Quantized - ratio value error {ratio} should be in ]0,1]")


class Convolution_FKAConv(torch.nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=16, bias=False, dim=3, kernel_separation=False, adaptive_normalization=True,**kwargs):
        super().__init__()

        # parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.dim = dim
        self.adaptive_normalization = adaptive_normalization

        # convolution kernel
        if kernel_separation:
            # equivalent to two kernels K1 * K2
            dm = int(ceil(self.out_channels / self.in_channels))
            self.cv = nn.Sequential(
                nn.Conv2d(in_channels, dm*in_channels, (1, kernel_size), bias=bias, groups=self.in_channels),
                nn.Conv2d(in_channels*dm, out_channels, (1, 1), bias=bias)
            )
        else:
            self.cv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), bias=bias)

        # normalization radius
        if self.adaptive_normalization:
            self.norm_radius_momentum = 0.1
            self.norm_radius = nn.Parameter(torch.Tensor(1,), requires_grad=False)
            self.alpha = nn.Parameter(torch.Tensor(1,), requires_grad=True)
            self.beta = nn.Parameter(torch.Tensor(1,), requires_grad=True)
            torch.nn.init.ones_(self.norm_radius.data)
            torch.nn.init.ones_(self.alpha.data)
            torch.nn.init.ones_(self.beta.data)

        # features to kernel weights
        self.fc1 = nn.Conv2d(self.dim, self.kernel_size, 1, bias=False)
        self.fc2 = nn.Conv2d(2 * self.kernel_size, self.kernel_size, 1, bias=False)
        self.fc3 = nn.Conv2d(2 * self.kernel_size, self.kernel_size, 1, bias=False)
        self.bn1 = nn.InstanceNorm2d(self.kernel_size, affine=True)
        self.bn2 = nn.InstanceNorm2d(self.kernel_size, affine=True)



    def fixed_normalization(self, pts, radius=None):
        maxi = torch.sqrt((pts.detach() ** 2).sum(1).max(2)[0])
        maxi = maxi + (maxi == 0)
        return pts / maxi.view(maxi.size(0), 1, maxi.size(1), 1)



    def forward(self, x, pos, support_points, neighbors_indices):

        if x is None:
            return None

        pos = batch_gather(pos, dim=2, index=neighbors_indices).contiguous()
        x = batch_gather(x, dim=2, index=neighbors_indices).contiguous()

        # center the neighborhoods (local coordinates)
        pts = pos - support_points.unsqueeze(3)


        # normalize points
        if self.adaptive_normalization:


            # compute distances from points to their support point
            distances = torch.sqrt((pts.detach() ** 2).sum(1))

            # update the normalization radius
            if self.training:
                mean_radius = distances.max(2)[0].mean()
                self.norm_radius.data = (
                    self.norm_radius.data * (1 - self.norm_radius_momentum)
                    + mean_radius * self.norm_radius_momentum
                )

            # normalize
            pts = pts / self.norm_radius

            # estimate distance weights
            distance_weight = torch.sigmoid(-self.alpha * distances + self.beta)
            distance_weight_s = distance_weight.sum(2, keepdim=True)
            distance_weight_s = distance_weight_s + (distance_weight_s == 0) + 1e-6
            distance_weight = (
                distance_weight / distance_weight_s * distances.shape[2]
            ).unsqueeze(1)

            # feature weighting matrix estimation
            if pts.shape[3] == 1:
                mat = F.relu(self.fc1(pts))
            else:
                mat = F.relu(self.bn1(self.fc1(pts)))
            mp1 = torch.max(mat * distance_weight, dim=3, keepdim=True)[0].expand(
                (-1, -1, -1, mat.shape[3])
            )
            mat = torch.cat([mat, mp1], dim=1)
            if pts.shape[3] == 1:
                mat = F.relu(self.fc2(mat))
            else:
                mat = F.relu(self.bn2(self.fc2(mat)))
            mp2 = torch.max(mat * distance_weight, dim=3, keepdim=True)[0].expand(
                (-1, -1, -1, mat.shape[3])
            )
            mat = torch.cat([mat, mp2], dim=1)
            mat = F.relu(self.fc3(mat)) * distance_weight
            # mat = torch.sigmoid(self.fc3(mat)) * distance_weight
        else:
            pts = self.fixed_normalization(pts)

            # feature weighting matrix estimation
            if pts.shape[3] == 1:
                mat = F.relu(self.fc1(pts))
            else:
                mat = F.relu(self.bn1(self.fc1(pts)))
            mp1 = torch.max(mat, dim=3, keepdim=True)[0].expand(
                (-1, -1, -1, mat.shape[3])
            )
            mat = torch.cat([mat, mp1], dim=1)
            if pts.shape[3] == 1:
                mat = F.relu(self.fc2(mat))
            else:
                mat = F.relu(self.bn2(self.fc2(mat)))
            mp2 = torch.max(mat, dim=3, keepdim=True)[0].expand(
                (-1, -1, -1, mat.shape[3])
            )
            mat = torch.cat([mat, mp2], dim=1)
            mat = F.relu(self.fc3(mat))

        # compute features
        features = torch.matmul(
            x.transpose(1, 2), mat.permute(0, 2, 3, 1)
        ).transpose(1, 2)
        features = self.cv(features).squeeze(3)

        return features

Conv = Convolution_FKAConv

NormLayer = nn.BatchNorm1d
sampling = sampling_quantized
#knn = compute_knn_fast # compute_knn_fast #compute_knn
##conv_fkaconv

class Convolution_FKAConv(torch.nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=16, bias=False, dim=3, kernel_separation=False, adaptive_normalization=True,**kwargs):
        super().__init__()

        # parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.dim = dim
        self.adaptive_normalization = adaptive_normalization

        # convolution kernel
        if kernel_separation:
            # equivalent to two kernels K1 * K2
            dm = int(ceil(self.out_channels / self.in_channels))
            self.cv = nn.Sequential(
                nn.Conv2d(in_channels, dm*in_channels, (1, kernel_size), bias=bias, groups=self.in_channels),
                nn.Conv2d(in_channels*dm, out_channels, (1, 1), bias=bias)
            )
        else:
            self.cv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), bias=bias)

        # normalization radius
        if self.adaptive_normalization:
            self.norm_radius_momentum = 0.1
            self.norm_radius = nn.Parameter(torch.Tensor(1,), requires_grad=False)
            self.alpha = nn.Parameter(torch.Tensor(1,), requires_grad=True)
            self.beta = nn.Parameter(torch.Tensor(1,), requires_grad=True)
            torch.nn.init.ones_(self.norm_radius.data)
            torch.nn.init.ones_(self.alpha.data)
            torch.nn.init.ones_(self.beta.data)

        # features to kernel weights
        self.fc1 = nn.Conv2d(self.dim, self.kernel_size, 1, bias=False)
        self.fc2 = nn.Conv2d(2 * self.kernel_size, self.kernel_size, 1, bias=False)
        self.fc3 = nn.Conv2d(2 * self.kernel_size, self.kernel_size, 1, bias=False)
        self.bn1 = nn.InstanceNorm2d(self.kernel_size, affine=True)
        self.bn2 = nn.InstanceNorm2d(self.kernel_size, affine=True)



    def fixed_normalization(self, pts, radius=None):
        maxi = torch.sqrt((pts.detach() ** 2).sum(1).max(2)[0])
        maxi = maxi + (maxi == 0)
        return pts / maxi.view(maxi.size(0), 1, maxi.size(1), 1)



    def forward(self, x, pos, support_points, neighbors_indices):

        if x is None:
            return None

        pos = batch_gather(pos, dim=2, index=neighbors_indices).contiguous()
        x = batch_gather(x, dim=2, index=neighbors_indices).contiguous()

        # center the neighborhoods (local coordinates)
        pts = pos - support_points.unsqueeze(3)


        # normalize points
        if self.adaptive_normalization:


            # compute distances from points to their support point
            distances = torch.sqrt((pts.detach() ** 2).sum(1))

            # update the normalization radius
            if self.training:
                mean_radius = distances.max(2)[0].mean()
                self.norm_radius.data = (
                    self.norm_radius.data * (1 - self.norm_radius_momentum)
                    + mean_radius * self.norm_radius_momentum
                )

            # normalize
            pts = pts / self.norm_radius

            # estimate distance weights
            distance_weight = torch.sigmoid(-self.alpha * distances + self.beta)
            distance_weight_s = distance_weight.sum(2, keepdim=True)
            distance_weight_s = distance_weight_s + (distance_weight_s == 0) + 1e-6
            distance_weight = (
                distance_weight / distance_weight_s * distances.shape[2]
            ).unsqueeze(1)

            # feature weighting matrix estimation
            if pts.shape[3] == 1:
                mat = F.relu(self.fc1(pts))
            else:
                mat = F.relu(self.bn1(self.fc1(pts)))
            mp1 = torch.max(mat * distance_weight, dim=3, keepdim=True)[0].expand(
                (-1, -1, -1, mat.shape[3])
            )
            mat = torch.cat([mat, mp1], dim=1)
            if pts.shape[3] == 1:
                mat = F.relu(self.fc2(mat))
            else:
                mat = F.relu(self.bn2(self.fc2(mat)))
            mp2 = torch.max(mat * distance_weight, dim=3, keepdim=True)[0].expand(
                (-1, -1, -1, mat.shape[3])
            )
            mat = torch.cat([mat, mp2], dim=1)
            mat = F.relu(self.fc3(mat)) * distance_weight
            # mat = torch.sigmoid(self.fc3(mat)) * distance_weight
        else:
            pts = self.fixed_normalization(pts)

            # feature weighting matrix estimation
            if pts.shape[3] == 1:
                mat = F.relu(self.fc1(pts))
            else:
                mat = F.relu(self.bn1(self.fc1(pts)))
            mp1 = torch.max(mat, dim=3, keepdim=True)[0].expand(
                (-1, -1, -1, mat.shape[3])
            )
            mat = torch.cat([mat, mp1], dim=1)
            if pts.shape[3] == 1:
                mat = F.relu(self.fc2(mat))
            else:
                mat = F.relu(self.bn2(self.fc2(mat)))
            mp2 = torch.max(mat, dim=3, keepdim=True)[0].expand(
                (-1, -1, -1, mat.shape[3])
            )
            mat = torch.cat([mat, mp2], dim=1)
            mat = F.relu(self.fc3(mat))

        # compute features
        features = torch.matmul(
            x.transpose(1, 2), mat.permute(0, 2, 3, 1)
        ).transpose(1, 2)
        features = self.cv(features).squeeze(3)

        return features


class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,adaptive_normalization=True):
        super().__init__()

        self.cv0 = nn.Conv1d(in_channels, in_channels//2, 1)
        self.bn0 = NormLayer(in_channels//2)
        self.cv1 = Conv(in_channels//2, in_channels//2, kernel_size, adaptive_normalization=adaptive_normalization)
        self.bn1 = NormLayer(in_channels//2)
        self.cv2 = nn.Conv1d(in_channels//2, out_channels, 1)
        self.bn2 = NormLayer(out_channels)
        self.activation = nn.ReLU(inplace=True)

        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.bn_shortcut = NormLayer(out_channels) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, pos, support_points, neighbors_indices):

        x_short = x
        x = self.activation(self.bn0(self.cv0(x)))
        x = self.activation(self.bn1(self.cv1(x, pos, support_points, neighbors_indices)))
        x = self.bn2(self.cv2(x))

        x_short = self.bn_shortcut(self.shortcut(x_short))
        if x_short.shape[2] != x.shape[2]:
            x_short = max_pool(x_short, neighbors_indices)

        x = self.activation(x + x_short)

        return x


class FKA_Conv_Encoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels, segmentation=False, hidden=64, dropout=0.5, last_layer_additional_size=None, adaptive_normalization=True, fix_support_number=False, **kwargs):
        super().__init__()

        self.lcp_preprocess = True
        self.segmentation = segmentation
        self.adaptive_normalization = adaptive_normalization
        self.fix_support_point_number = fix_support_number

        self.cv0 = Conv(in_channels, hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.bn0 = NormLayer(hidden)

        
        self.resnetb01 = ResidualBlock(hidden, hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb10 = ResidualBlock(hidden, 2*hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb11 = ResidualBlock(2*hidden, 2*hidden, 16, adaptive_normalization=self.adaptive_normalization) 
        self.resnetb20 = ResidualBlock(2*hidden, 4*hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb21 = ResidualBlock(4*hidden, 4*hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb30 = ResidualBlock(4*hidden, 8*hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb31 = ResidualBlock(8*hidden, 8*hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb40 = ResidualBlock(8*hidden, 16*hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb41 = ResidualBlock(16*hidden, 16*hidden, 16, adaptive_normalization=self.adaptive_normalization)
        if self.segmentation:

            self.cv5 = nn.Conv1d(32*hidden, 16 * hidden, 1)
            self.bn5 = NormLayer(16*hidden)
            self.cv3d = nn.Conv1d(24*hidden, 8 * hidden, 1)
            self.bn3d = NormLayer(8 * hidden)
            self.cv2d = nn.Conv1d(12 * hidden, 4 * hidden, 1)
            self.bn2d = NormLayer(4 * hidden)
            self.cv1d = nn.Conv1d(6 * hidden, 2 * hidden, 1)
            self.bn1d = NormLayer(2 * hidden)
            self.cv0d = nn.Conv1d(3 * hidden, hidden, 1)
            self.bn0d = NormLayer(hidden)

            if last_layer_additional_size is not None:
                self.fcout = nn.Conv1d(hidden+last_layer_additional_size, out_channels, 1)
            else:
                self.fcout = nn.Conv1d(hidden, out_channels, 1)
        else:
            self.fcout = nn.Conv1d(16*hidden, out_channels, 1)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    @staticmethod
    def compute_support_ids( data, segmentation=True):

        pos = data["pos"].clone()

        add_batch_dimension = False
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(0)
            add_batch_dimension = True
        
        # compute the support points
        # if self.fix_support_point_number:
        #     support1, _ = sampling(pos, n_support=512)
        #     support2, _ = sampling(support1, n_support=128)
        #     support3, _ = sampling(support2, n_support=32)
        #     support4, _ = sampling(support3, n_support=8)
        # else:
        support1, _ = sampling(pos, 0.25)
        support2, _ = sampling(support1, 0.25)
        support3, _ = sampling(support2, 0.25)
        support4, _ = sampling(support3, 0.25)


        # compute the ids
        ids00 = knn(pos, pos, 16)
        ids01 = knn(pos, support1, 16)
        ids11 = knn(support1, support1, 16)
        ids12 = knn(support1, support2, 16)
        ids22 = knn(support2, support2, 16)
        ids23 = knn(support2, support3, 16)
        ids33 = knn(support3, support3, 16)
        ids34 = knn(support3, support4, 16)
        ids44 = knn(support4, support4, 16)
        if segmentation:
            ids43 = knn(support4, support3, 1)
            ids32 = knn(support3, support2, 1)
            ids21 = knn(support2, support1, 1)
            ids10 = knn(support1, pos, 1)

        ret_data = {}
        if add_batch_dimension:
            support1 = support1.squeeze(0)
            support2 = support2.squeeze(0)
            support3 = support3.squeeze(0)
            support4 = support4.squeeze(0)
            ids00 = ids00.squeeze(0)
            ids01 = ids01.squeeze(0)
            ids11 = ids11.squeeze(0)
            ids12 = ids12.squeeze(0)
            ids22 = ids22.squeeze(0)
            ids23 = ids23.squeeze(0)
            ids33 = ids33.squeeze(0)
            ids34 = ids34.squeeze(0)
            ids44 = ids44.squeeze(0)

        ret_data["support1"] = support1
        ret_data["support2"] = support2
        ret_data["support3"] = support3
        ret_data["support4"] = support4

        ret_data["ids00"] = ids00
        ret_data["ids01"] = ids01
        ret_data["ids11"] = ids11
        ret_data["ids12"] = ids12
        ret_data["ids22"] = ids22
        ret_data["ids23"] = ids23
        ret_data["ids33"] = ids33
        ret_data["ids34"] = ids34
        ret_data["ids44"] = ids44

        if segmentation:
            if add_batch_dimension:
                ids43 = ids43.squeeze(0)
                ids32 = ids32.squeeze(0)
                ids21 = ids21.squeeze(0)
                ids10 = ids10.squeeze(0)
            
            ret_data["ids43"] = ids43
            ret_data["ids32"] = ids32
            ret_data["ids21"] = ids21
            ret_data["ids10"] = ids10
        
        
        return ret_data

    def forward(self, data,  compute_support_ids=True, cat_in_last_layer=None):



        if  compute_support_ids:

            spatial_data = FKA_Conv_Encoder.compute_support_ids(data, segmentation= self.segmentation)
            for key, value in spatial_data.items():
                data[key] = value
            # data = {**data, **spatial_data}

        x = data["x"]
        pos = data["pos"]

        x0 = self.activation(self.bn0(self.cv0(x, pos, pos, data["ids00"])))
        x0 = self.resnetb01(x0, pos, pos, data["ids00"])
        x1 = self.resnetb10(x0, pos, data["support1"], data["ids01"])
        x1 = self.resnetb11(x1, data["support1"], data["support1"], data["ids11"])
        x2 = self.resnetb20(x1, data["support1"], data["support2"], data["ids12"])
        x2 = self.resnetb21(x2, data["support2"], data["support2"], data["ids22"])
        x3 = self.resnetb30(x2, data["support2"], data["support3"], data["ids23"])
        x3 = self.resnetb31(x3, data["support3"], data["support3"], data["ids33"])
        x4 = self.resnetb40(x3, data["support3"], data["support4"], data["ids34"])
        x4 = self.resnetb41(x4, data["support4"], data["support4"], data["ids44"])

        if self.segmentation:
            
            x5 = x4.max(dim=2, keepdim=True)[0].expand_as(x4)
            x4d = self.activation(self.bn5(self.cv5(torch.cat([x4, x5], dim=1))))
            x4d = x4

            x3d = interpolate(x4d, data["ids43"])
            x3d = self.activation(self.bn3d(self.cv3d(torch.cat([x3d, x3], dim=1))))

            x2d = interpolate(x3d, data["ids32"])
            x2d = self.activation(self.bn2d(self.cv2d(torch.cat([x2d, x2], dim=1))))
            
            x1d = interpolate(x2d, data["ids21"])
            x1d = self.activation(self.bn1d(self.cv1d(torch.cat([x1d, x1], dim=1))))
            
            xout = interpolate(x1d, data["ids10"])
            xout = self.activation(self.bn0d(self.cv0d(torch.cat([xout, x0], dim=1))))
            xout = self.dropout(xout)
            if cat_in_last_layer is not None:
                xout = torch.cat([xout, cat_in_last_layer.expand(-1,-1,xout.shape[2])], dim=1)
            xout = self.fcout(xout)

        else:

            xout = x4
            xout = self.dropout(xout)
            xout = self.fcout(xout)
            xout = xout.mean(dim=2)

        return xout


class Attention_KHeads_Decoder(torch.nn.Module):

    def __init__(self, latent_size, out_channels, K=16, dists = False, **kwargs):
        super().__init__()
        self.dists = dists
        logging.info(f"Attention_KHeads_Decoder - Simple - K={K}")
        # self.projection_layer = FKAConv(latent_size, latent_size, 16, sampling=None, neighborhood_search=knn, neighborhood_size=16, ratio=1)
        self.fc1 = torch.nn.Conv2d(latent_size+3, latent_size, 1)
        self.fc2 = torch.nn.Conv2d(latent_size, latent_size, 1)
        self.fc3 = torch.nn.Conv2d(latent_size, latent_size, 1)

        self.fc8 = torch.nn.Conv1d(latent_size, out_channels, 1)
        self.activation = torch.nn.ReLU()

        self.fc_query = torch.nn.Conv2d(latent_size, 64, 1)
        self.fc_value = torch.nn.Conv2d(latent_size, latent_size,1)

        self.k = K

    @staticmethod
    def compute_nearest_neighbors( data,k,  forward_key= 'pos_non_manifold'):

        pos = data["pos"]
        pos_non_manifold = data[forward_key]

        add_batch_dimension_pos = False
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(0)
            add_batch_dimension_pos = True

        add_batch_dimension_non_manifold = False
        if len(pos_non_manifold.shape) == 2:
            pos_non_manifold = pos_non_manifold.unsqueeze(0)
            add_batch_dimension_non_manifold = True

        if pos.shape[1] != 3:
            pos = pos.transpose(1,2)

        if pos_non_manifold.shape[1] != 3:
            pos_non_manifold = pos_non_manifold.transpose(1,2)
        # if  self.dists:
        #     distances, indices = knn(pos, pos_non_manifold, self.k, dists = self.dists)
        # else: 
        indices = knn(pos, pos_non_manifold, k, dists = False)

        if add_batch_dimension_non_manifold or add_batch_dimension_pos:
            indices = indices.squeeze(0)

        ret_data = {}
        ret_data["nn_indices_"+forward_key] = indices
        # if self.dists:
        #     ret_data["proj_dists_"+forward_key] = distances
        return ret_data

    def forward(self, data,  compute_query_nn=True, last_layer=True, 
                            forward_key= 'pos_non_manifold',return_last_features=False):
        # if spatial_only:
        #     return self.compute_nearest_neighbors(data,forward_key=forward_key)

        if compute_query_nn :
            nearset_neighobs_data = Attention_KHeads_Decoder.compute_nearest_neighbors(data,k = self.k, forward_key=forward_key)
            for key, value in nearset_neighobs_data.items():
                data[key] = value

        
        x = data["latents"]
        indices = data["nn_indices_"+forward_key]
        pos = data["pos"]
        pos_non_manifold = data[forward_key]


        if pos.shape[1] != 3:
            pos = pos.transpose(1,2)

        if pos_non_manifold.shape[1] != 3:
            pos_non_manifold = pos_non_manifold.transpose(1,2)

        x = batch_gather(x, 2, indices)
        pos = batch_gather(pos, 2, indices)
        pos = pos_non_manifold.unsqueeze(3) - pos

        x = torch.cat([x,pos], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))

        query = self.fc_query(x)
        value = self.fc_value(x)

        attention = torch.nn.functional.softmax(query, dim=-1).mean(dim=1)
        x = torch.matmul(attention.unsqueeze(-2), value.permute(0,2,3,1)).squeeze(-2)
        x = x.transpose(1,2)
        
        if return_last_features:
            xout = self.fc8(x)
            return xout, x

        if last_layer:
            x = self.fc8(x)

        return x
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_latent_scale(net, data, latent_size = 16, n_iter = 1,n_manifold= 3000):
    pos = data["pos"][0].cpu().transpose(0,1).numpy()
    device = data["pos"].device
    # create the latent storage
    latent = torch.zeros((pos.shape[0], latent_size), dtype=torch.float)
    counts = torch.zeros((pos.shape[0],), dtype=torch.float)


    iteration = 0
    for current_value in range(n_iter):
        while counts.min() < current_value+1:
            # print("iter", iteration, current_value)
            valid_ids = torch.tensor(np.argwhere(counts.cpu().numpy()==current_value)[:,0]).long()
            
            if pos.shape[0] >= n_manifold:

                ids = torch.randperm(valid_ids.shape[0])[:n_manifold]
                ids = valid_ids[ids]
                
                if ids.shape[0] < n_manifold:
                    ids = torch.cat([ids, torch.randperm(pos.shape[0])[:n_manifold - ids.shape[0]]], dim=0)
                assert(ids.shape[0] == n_manifold)
            else:
                ids = torch.arange(pos.shape[0])


            data_partial = {
                "pos": data["pos"][0].transpose(1,0)[ids].transpose(1,0).unsqueeze(0),
                "x": data["x"][0].transpose(1,0)[ids].transpose(1,0).unsqueeze(0)
            }

            partial_latent = net.get_latent(data_partial, with_correction=False)["latents"]
            latent[ids] += partial_latent[0].cpu().numpy().transpose(1,0)
            counts[ids] += 1

            iteration += 1

    latent = latent / counts.unsqueeze(1)
    latent = latent.transpose(1,0).unsqueeze(0).to(device)
    data["latents"] = latent
    #print(latent.shape)
    return data

class POCONetwork(torch.nn.Module):

    def __init__(self, in_channels, latent_size, out_channels, backbone, decoder,dists = False , kernel = lambda x : x , n_offsets= None, **kwargs):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = eval(backbone)(in_channels, latent_size, segmentation=True, dropout=0,)
        self.decoder = eval(decoder["name"])(latent_size, out_channels, decoder["k"])
        self.lcp_preprocess = True
        self.encoder_spatial = False
        logging.info(f"Network -- encoder -- {count_parameters(self.encoder)} parameters")
        logging.info(f"Network -- decoder -- {count_parameters(self.decoder)} parameters")
    @staticmethod
    def compute_support_ids_nn(data,k , forward_key = 'pos_non_manifold',compute_support_ids= True, compute_query_nn = True):
        if compute_query_nn:
            data["nn_indices_"+forward_key] = Attention_KHeads_Decoder.compute_nearest_neighbors(data,k,
                                                                forward_key=forward_key)["nn_indices_"+forward_key]
        if compute_support_ids:
            net_data = FKA_Conv_Encoder.compute_support_ids(data, segmentation= True)
            for key, value in net_data.items():
                data[key] = value
        
        return data
    def forward(self, data, compute_support_ids=True, compute_query_nn=True, forward_key = 'pos_non_manifold', large_scale = False, **kwargs):

         
        data = POCONetwork.compute_support_ids_nn(data,self.decoder.k, forward_key, compute_support_ids, compute_query_nn )

        if large_scale:
            latents = get_latent_scale(self, data, **kwargs)["latents"] 
        else:
            latents = self.encoder(data, compute_support_ids=False)
        data["latents"] = latents
        ret_data = self.decoder(data, compute_query_nn=False, forward_key=forward_key)

        return ret_data



    def get_latent(self, data, compute_support_ids=True):

        latents = self.encoder(data, compute_support_ids=compute_support_ids)
        data["latents"] = latents
        return data

    def decode(self, data, queries,  *args):
        data['pos_non_manifold'] = queries
        output = self.decoder(data)
        return self.to_sdf(output)#["outputs"]
    @staticmethod
    def to_sdf(outputs):
        x = outputs.softmax(dim =1)
        return (x[:,1]-x[:,0])
if __name__ =='__main__':
    model = POCONetwork(in_channels= 3, latent_size=32, out_channels=32,
                                    backbone = 'FKA_Conv_Encoder',
                                    decoder = {"name": "Attention_KHeads_Decoder", "k": 16}).cuda()
    data = {}
    data['pos']= torch.rand(1, 3, 3000).cuda()
    data['x']= torch.ones_like(data['pos'])
    data['pos_non_manifold']= torch.rand(1,3, 2000).cuda()
    #POCO_Network.compute_support_ids_nn(data, forward_key = 'pos_non_manifold', k = 16,compute_support_ids= True, compute_query_nn=True )
    model( data, compute_support_ids=True, compute_query_nn=True, forward_key = 'pos_non_manifold', large_scale = False)
