from runner import ReconstructionPipeline 
import opts
from adapters.utils import set_all_seeds
import pandas as pd 
from tabulate import tabulate
from reconstruction import Field,Reconstructor
import torch
import os
os.environ['DISPLAY'] = ':99.0' 
os.environ['PYVISTA_OFF_SCREEN'] = 'true' 
# parse command line arguments
parser = opts.krr_opts()
args = parser.parse_args()

# set the root directory for the dataset

# read the shape names from the split file
#split_file = open(f'{args.root}/{args.classe}/{args.split_file}', 'r').readlines()
#args.shape = [shape.strip() for shape in split_file][args.id]

print(args.shape)
# set the seed for random number generation
set_all_seeds(0)

# set the resolution for the mesh reconstruction
resolution = 128

# create the reconstruction pipeline
pipeline = ReconstructionPipeline(args)

# run the reconstruction pipeline with the backbone model
mesh, df = pipeline.run(resolution)

# prepare the dataset for KRR
X_train, Y_train, X_nystrom = pipeline.prepare_krr_dataset(n_local_queries=3)

# fit the KRR solver
krr_solver = pipeline.fit_krr_solver(X_train, Y_train, X_nystrom)

# reconstruct the mesh using KRR
mesh_kernel, feature_volume = pipeline.reconstruct_krr(krr_solver, resolution, bounds=(-0.5, 0.5))

# evaluate the reconstruction using KRR
df_krr = pipeline.evaluate_reconstruction(mesh_kernel)

# run the NKRR adaptation
cd1_gt, nkrr_adapter = pipeline.run_nkrr_adaptation(feature_volume, X_nystrom, X_train, Y_train, epochs=50)

final_resolution = 128
if final_resolution != resolution: #if the resolution used for validation (to select the best epoch) is different from the final resolution. When compararing to the backbone model, the resolution should be the same.
    print("computing grid points")
    grid_points = Reconstructor.get_mc_points(final_resolution, bounds=(-0.5, 0.5), batch_points=100000)
    print("Computing feature volume")
    feature_volume = Reconstructor.compute_feature_volume(pipeline.field, grid_points, resolution=final_resolution, mc_device='cuda')
    print("feature volume shape", feature_volume.shape)
    split_feature_volume = torch.split(feature_volume, 128*128*128, dim=0) 
##evaluate the NKRR adaptation
else:
    split_feature_volume = [feature_volume]
mesh_kernel, df_fkrr = pipeline.evaluate_nkrr_adaptater(nkrr_adapter, split_feature_volume, final_resolution)
# concatenate the results and set the index
df = pd.concat([df, df_krr, df_fkrr], ignore_index=True)
df.index = [f'{args.backbone}', f'{args.backbone}_krr', f'{args.backbone}_fnkrr']
print(tabulate(df, headers='keys', tablefmt='fancy_grid'))

