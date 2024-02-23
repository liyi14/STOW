## Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

#### Building on another system
To build on a system that does not have a GPU device but provide the drivers:
```bash
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
```

### Example conda environment setup and dependencies intallation
```bash
conda create --name stow python=3.8 -y
conda activate stow
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

#Ensure you have the right version of setuptools installed: 
pip install setuptools==59.5.0

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
# pip install git+https://github.com/cocodataset/panopticapi.git
# pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
git clone git@github.com:liyi14/STOW.git
cd STOW
pip install -r requirements.txt

cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

### Tested System
This has been tested and implemented on a systems with these attributes:

- Ubuntu 20.04.6
- CUDA Version 11.7
- NVIDIA GeForce RTX 2080 Ti

