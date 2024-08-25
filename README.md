# Feat.NKRR-adaptation
[NeurIPS'23] "Robustifying Generalizable Implicit Shape Networks with a Tunable Non-Parametric Model"

![Teaser Image](teaser.jpg)


**Robustifying Generalizable Implicit Shape Networks with a Tunable Non-Parametric Model**
=====================================================================================

**Overview**
------------

This repository provides an implementation of the NeurIPS 2023 paper "Robustifying Generalizable Implicit Shape Networks with a Tunable Non-Parametric Model". The paper proposes a novel approach to improve the robustness and generalizability of implicit shape networks using a tunable non-parametric model.

**Background**
-------------

Implicit shape networks have shown great promise in representing complex shapes and scenes. However, they often suffer from limited robustness and generalizability, particularly when faced with noisy or incomplete data. To address this limitation, our approach combines the strengths of implicit shape networks with the flexibility of non-parametric models.

**Methodology**
--------------

Our approach consists of the following key components:

1. **Implicit Shape Network**: We use a variant of the implicit shape network architecture, which represents shapes as continuous functions that map 3D coordinates to occupancy values.
2. **Tunable Non-Parametric Model**: We introduce a tunable non-parametric model that adapts to the complexity of the input data. This model is based on a Gaussian process kernel, which provides a flexible and robust way to represent complex shapes.
3. **Hybrid Model**: We combine the implicit shape network and the tunable non-parametric model to form a hybrid model. This model leverages the strengths of both components to improve robustness and generalizability.

**Implementation**
-----------------

Our implementation is based on PyTorch and provides the following features:

* **Implicit Shape Network**: We provide a PyTorch implementation of the implicit shape network architecture, which can be used as a standalone model or as a component of the hybrid model.
* **Tunable Non-Parametric Model**: We provide a PyTorch implementation of the tunable non-parametric model, which can be used as a standalone model or as a component of the hybrid model. Specifically, our implementation leverages the [Falkon](https://github.com/falkonml/falkon)  for for efficient kernel-based learning. We also show an alternative method with Sparse Gaussian Process Regression using the  [GPyTorch](https://gpytorch.ai/) library.
* **Training and Evaluation**: We provide scripts for training and evaluating the hybrid model on various datasets.

**Datasets**
------------



We provide support for the following datasets:

* **ShapeNet**: A large-scale dataset of 3D shapes, which is commonly used for evaluating shape representation and reconstruction methods.
* **ScanNet**: A dataset of 3D scans, which is commonly used for evaluating shape reconstruction and scene understanding methods.
* **FAUST**: A dataset of 3D human body scans, which is commonly used for evaluating shape registration and reconstruction methods..

**Usage**
---------
The `example.py`  script that sets up a reconstruction pipeline using the `ReconstructionPipeline` class from the `runner` module. Here's an example of how to use this file:

#### Command Line Arguments

The script uses the `opts` module to parse command line arguments. You can pass the following arguments:

* `--root`: the root directory of the dataset
* `--classe`: the class of shapes to reconstruct
* `--split_file`: the file containing the list of shapes to reconstruct (e.g., train/val/test)
* `--id`: the ID of the shape to reconstruct
* `--shape`: the name of the shape to reconstruct (optional)

#### Running the Script

To run the script, save the `example.py` file and execute it from the command line:
```bash
python example.py --root /path/to/dataset --classe my_class --split_file my_split_file.txt --id 0 -p 1e-4 --n_nystrom 1000 --solver KRRG  --optim_nystrom --backbone poco --n_points 10000 
```
Replace the `--root`, `--classe`, `--split_file`, and `--id` arguments with your own values.

For parallel processing you can use (don't forget to save the results):
```bash
parallel -j n_processes  CUDA_VISIBLE_DEVICES=1 python example.py   -p 1e-4 --n_nystrom 1000 --solver KRRG  --optim_nystrom --backbone poco -id {} --silent --save --classe my_class --split_file my_split_file.txt  --n_points 10000 :::  $(seq 0 $(wc -l < my_split_file.txt))

```

#### Output

The script will print the name of the shape being reconstructed, set the seed for random number generation, and set the resolution for the mesh reconstruction and and print the  Chamfer Distance between the reconstructed mesh and ground-truth pointcloud.

#### Customization

You can customize the script by modifying the `example.py` file. For example, you can change the resolution of the mesh reconstruction by modifying the `resolution` variable.

**Example Use Cases**
--------------------

Here are some example use cases for this repository:

* **Shape Reconstruction**: Use this model to reconstruct 3D shapes from noisy or incomplete data.
* **Scene Understanding**: Use this model to understand complex scenes, such as indoor or outdoor environments.


**License**
----------

This repository is licensed under the MIT License. See the `LICENSE` file for details.

**Citing this Work**
-------------------

If you use this repository in your research, please cite the following paper:

* Amine Ouasfi, Adnane Boukhayma. (2023). Robustifying Generalizable Implicit Shape Networks with a Tunable Non-Parametric Model. In Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS 2023).



**Acknowledgments**
------------------

This repository is based on the following open-source projects:

* [Convolutional Occupancy Networks (ConvONet)](https://github.com/autonomousvision/convolutional_occupancy_networks)
* [POCO: Point Convolution for Surface Reconstruction](https://github.com/valeoai/POCO)

We would like to thank the authors of these projects for making their code available, and for providing a foundation for our own research.

[teaser.jpg]: teaser.png