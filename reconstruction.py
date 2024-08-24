import mcubes
import numpy as np
import torch
import open3d as o3d

import functools
from poco import model as poco

# Define the decorator with size argument
from diso import DiffMC
from diso import DiffDMC
from diso._C import CUMCFloat as mcf
from diso  import _C 
import torch.nn.functional as F ;
from mise import MISE
def CUMC (field_volume,threshold):
    diffdmc = DiffMC(dtype=torch.float32)
    vertices, triangles = diffdmc (
                              -field_volume, isovalue=threshold, normalize = False)
    vertices += 1 # Retrived in run mc 
    return vertices.data.cpu().numpy(), triangles.data.cpu().numpy()

class Field:
    
    def __init__(self, model, inputs,  
                 encode_method = "get_latent",
                 output_transform = lambda x :x):
        super().__init__()
        self.model = model
        self.output_transform = output_transform
        self.latents =  getattr(self.model, encode_method) (inputs)
    def __call__(self, points):
        outputs = self.model.decode(self.latents, points)
        return self.output_transform (outputs)
class Reconstructor:
    def __init__(self, field):
        self.field = field
    @torch.no_grad()
    def __call__(self,threshold, resolution = 128, bounds = (-0.5, 0.5) , batch_points =  50000, mc_device = 'cpu'):
         grid_points_split = self.get_mc_points(resolution, bounds , batch_points )
         field_volume = self.compute_field_volume(grid_points_split, resolution,mc_device = mc_device)
         mesh = self.run_mc (field_volume, threshold, resolution ,  bounds,mc_device = mc_device )
         return mesh
    @staticmethod
    def  run_mc (field_volume, threshold, resolution = 128, bounds = (-0.5, 0.5) ,mc_device = 'cpu'):
            min, max = bounds
            threshold = np.log(threshold) - np.log(1. - threshold)
            mc_fn =  CUMC if mc_device =='cuda' else mcubes.marching_cubes
            vertices, triangles = mc_fn (field_volume, threshold)

                # remove translation due to padding
            #vertices -= 0.5
            vertices -= 1
                #rescale to original scale
            step = (max - min) / (resolution - 1)
            vertices = np.multiply(vertices, step)
            vertices += [min, min, min]
  
                # make mesh with open 3d
            pred_mesh = o3d.geometry.TriangleMesh()
            pred_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            pred_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            return pred_mesh
    @staticmethod
    def create_grid_points(minimun, maximum, res):
        x = np.linspace(minimun, maximum, res)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        X = X.reshape((np.prod(X.shape),))
        Y = Y.reshape((np.prod(Y.shape),))
        Z = Z.reshape((np.prod(Z.shape),))

        points_list = np.column_stack((X, Y, Z))
        del X, Y, Z, x
        return points_list
    @staticmethod
    def  get_mc_points(resolution = 128, bounds = (-0.5 , 0.5), batch_points =  50000, device = "cuda"):
            min,max = bounds
            a = max + min
            b = max - min

            grid_points = Reconstructor.create_grid_points(min, max, resolution)
            grid_coords = grid_points
            #grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()
            #)
            #grid_coords = 2 * grid_points - a
            #grid_coords = grid_coords / b
            grid_coords = torch.from_numpy(grid_coords).to(device, dtype=torch.float32)
            grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(device)
            grid_points_split =    torch.split(grid_coords, batch_points, dim=1)
            return grid_points_split
    @torch.no_grad()
    def compute_field_volume(self,grid_points_split, resolution = 128, mc_device = 'cpu'):
        logits_list = []
        self.field.model.eval()
        for points in grid_points_split:
            with torch.no_grad():
                out = self.field(points)
                logits_list.append(out.squeeze(0).detach().to(mc_device))

        logits = torch.cat(logits_list, dim=0)
        if mc_device =='cuda':
                return torch.reshape(logits, (resolution,) * 3) 
            # mesh from  logit 
        field_volume = np.reshape(logits.numpy(), (resolution,) * 3) 
        field_volume = np.pad(field_volume, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        return field_volume 
    @staticmethod
    def compute_feature_volume(field,grid_points_split, resolution = 128, mc_device = 'cpu'):
        logits_list = []
        field.model.eval()
        for points in grid_points_split:
            with torch.no_grad():
                with field.model.feat_ctx(field.model) as f_i:
                    outputs = field( points)
                features = f_i[0][0] 
        #features = datamodule.normalize_features(features)
                points_features = features.transpose(1,2).reshape(-1,features.size(1))#.contiguous()
                logits_list.append(points_features.squeeze(0).detach().to(mc_device))

        logits = torch.cat(logits_list, dim=0)
        #if mc_device =='cuda':
        return logits
   
    



class MiseReconstructor (Reconstructor):
    def __init__(self, field):
        super().__init__(field)
    def __call__(self,threshold, upsampling_steps = 1 , resolution_0 = 128, bounds = (-0.5, 0.5) , batch_points =  50000, mc_device = 'cpu'):
         #grid_points_split = self.get_mc_points(resolution, bounds , batch_points )
         field_volume,resolution = self.compute_field_volume(resolution0=resolution_0, 
                                                    upsampling_steps = upsampling_steps,threshold = threshold,  mc_device = mc_device)
         if mc_device== 'cuda':
            field_volume = torch.from_numpy(field_volume).to(mc_device)
         mesh = self.run_mc (field_volume , threshold, resolution ,  bounds,mc_device = mc_device )
         return mesh
    @torch.no_grad()
    def compute_field_volume(self,resolution0=128, upsampling_steps = 1,threshold = 0.5,  mc_device = 'cpu'):
        threshold = np.log(threshold) - np.log(1. - threshold)
        mesh_extractor = MISE(
                        resolution0, upsampling_steps, threshold)
        box_size = 1.
        points = mesh_extractor.query()
        while points.shape[0] != 0:
            # Query points
            pointsf = points / mesh_extractor.resolution
            # Normalize to bounding box
            pointsf = box_size * (pointsf - 0.5)
            pointsf = torch.FloatTensor(pointsf).to('cuda')
            # Evaluate model and update
            print(pointsf.shape)
            values = -self.field(pointsf.unsqueeze(0)).cpu().numpy().squeeze()
            values = values.astype(np.float64)
            mesh_extractor.update(points, values)
            points = mesh_extractor.query()

        field_volume = mesh_extractor.to_dense()
        #field_volume = np.pad(field_volume, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)

        return field_volume , mesh_extractor.resolution

                