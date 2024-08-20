#!/bin/bash


conda create -n dynamic_c3dgs python=3.7.13
conda activate dynamic_c3dgs

# seems that we sometimes got stuck in environment.yml, so we install the packages one by one
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge


# Install for Gaussian Rasterization (Ch9) - Ours-Full
pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9


# install simpleknn
pip install thirdparty/gaussian_splatting/submodules/simple-knn

# install opencv-python-headless, to work with colmap on server
pip install opencv-python
# Install MMCV for CUDA KNN, used for init point sampling, reduce number of points when sfm points are too many
cd thirdparty
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -e .
cd ../../

# other packages
pip install natsort
pip install scipy
pip install kornia
# install colmap for preprocess, work with python3.8
conda create -n colmapenv python=3.8
conda activate colmapenv
pip install opencv-python-headless
pip install tqdm
pip install natsort
pip install Pillow
pip install dahuffman==0.4.1
pip install vector-quantize-pytorch==1.8.1
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install plyfile
pip install scikit-image
# just some files need torch be installed.
conda install pytorch==1.12.1 -c pytorch -c conda-forge
conda config --set channel_priority false
conda install colmap -c conda-forge

conda activate dynamic_c3dgs