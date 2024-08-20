# Compact 3D Gaussian Splatting for Static and Dynamic Radiance Fields
### Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park
### [[Project Page](https://maincold2.github.io/c3dgs/)] [[Extended Paper](https://arxiv.org/abs/2408.03822)]

This is an extended version of [C3DGS (CVPR24)](https://github.com/maincold2/Compact-3DGS) for dynamic scenes.

Our code is based on [STG@d9833e6](https://github.com/oppo-us-research/SpacetimeGaussians/tree/d9833e6c8406e7f8f6b0a437762d6d8e0758defb).

## Setup

### Installation
```shell
git clone https://github.com/maincold2/Dynamic_C3DGS.git
cd Dynamic_C3DGS
bash script/setup.sh
```

### Dataset Preprocessing
We used [Neural 3D](https://github.com/facebookresearch/Neural_3D_Video.git) and [Technicolor](https://www.interdigital.com/data_sets/light-field-dataset) datasets.

```python
conda activate colmapenv
# Neural 3D
python script/pre_n3d.py --videopath <location>/<scene>
# Technicolor
python script/pre_technicolor.py --videopath <location>/<scene>
```

## Running

### Training
```python
conda activate dynamic_c3dgs
python train.py --quiet --eval --config config/n3d_ours/<scene>.json --model_path <path to save model> --source_path <location>/<scene>/colmap_0
```
#### --comp 
Applying post-processings for compression.
#### --store_npz 
Storing npz file reflecting the actual storage.

#### For example, we can apply post-processings and save the compressed results in .npz format by the following command.
```
python train.py --quiet --eval --config configs/n3d_ours/cook_spinach.json --model_path log/ours_cook_spinach --source_path <location>/cook_spinach/colmap_0 --comp --store_npz
```
<details>
<summary><span style="font-weight: bold;">More hyper-parameters in the config file</span></summary>

  Command line arguments can also set these.
  #### lambda_mask
  Weight of masking loss to control the number of Gaussians, 0.0005 by default
  #### mask_lr
  Learning rate of the masking parameter, 0.01 by default
  #### net_lr 
  Learning rate for the neural field, 0.001 by default
  #### net_lr_step
  Step schedule for training the neural field
  #### max_hashmap
  Maximum hashmap size (log) of the neural field
  #### rvq_size_geo
  Codebook size in each R-VQ stage for geometric attributes
  #### rvq_num_geo
  The number of R-VQ stages for geometric attributes
  #### rvq_size_temp
  Codebook size in each R-VQ stage for temporal attributes
  #### rvq_num_temp
  The number of R-VQ stages for temporal attributes
  #### mask_prune_iter
  Pruning inteval after densification, 1000 by default
  #### rvq_iter
  The iteration at which R-VQ is implemented
</details>
<br>

### Evaluation

```python
# Neural 3D
python test.py --quiet --eval --skip_train --valloader colmapvalid --configpath config/n3d_ours/<scene>.json --model_path <path to model>
# Technicolor
python test.py --quiet --eval --skip_train --valloader technicolorvalid --configpath config/techni_ours/<scene>.json --model_path <path to model>
```

#### To run the original STG, use stg's config files (e.g., configs/n3d_stg/\<scene\>.json).

## Real-time Viewer
Without --comp and --store_npz options, our code saves the models in the original STG format, which can be used for the STG's viewer.

## Acknowledgements
A great thanks to the authors of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [STG](https://github.com/oppo-us-research/SpacetimeGaussians) for their amazing work. For more details, please check out their repos.

## BibTeX
```
@article{Lee_2024_C3DGS,
  title={Compact 3D Gaussian Splatting for Static and Dynamic Radiance Fields},
  author={Lee, Joo Chan and Rho, Daniel and Sun, Xiangyu and Ko, Jong Hwan and Park, Eunbyung},
  journal={arXiv preprint arXiv:2408.03822},
  year={2024}
}
@InProceedings{Lee_2024_CVPR,
  author    = {Lee, Joo Chan and Rho, Daniel and Sun, Xiangyu and Ko, Jong Hwan and Park, Eunbyung},
  title     = {Compact 3D Gaussian Representation for Radiance Field},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024},
  pages     = {21719-21728}
}
```