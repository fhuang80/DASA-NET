DASA-Net: A Novel Deep-Learning-Based Method for Point Cloud Tree Species Recognition
========

1. Introduction 
--------
The accompanied materials are provided along with the paper “DASA-Net: A Novel Deep-Learning-Based Method for Point Cloud Tree Species Recognition”. The materials mainly include the source code of DASA-Net model, the point cloud dataset mentioned in the paper, and the descriptions and instructions of the code operations. Need to note:
(1)	The model code includes modules for testing with comparative models, e.g., PointNeXt, ASSANet, DeepGCN, and PointNet++, and the corresponding evaluation metrics, e.g., OA, mAcc.
(2)	Due to the large volume of point cloud datasets (24 species, 6,312 samples), the raw data cannot be uploaded to GitHub. Please prepare and download it yourself.

2. Data Preprocessing
----------
Process raw single-tree point cloud data (in .laz format) through the following pipeline:
(1)	Execute sampleto2048.py to downsample point clouds to 2048 points and output with .txt format.
(2)	Run changelabel.py to unify species labels to the range of 0-23.
(3)	Use txttoh5.py to convert labeled .txt files into HDF5 datasets for model training.

3. Requirements of Operating Environment
----------
open3d: The open3d environment can be used to run all local python scripts in the handover file, and the environment installation can refer to relevant materials online. The corresponding relationship between the following files and their environment is described as follows ( this code is all running on Linux).
(1) The OpenPoints environment is used for deep learning of tree species recognition.(Configure it yourself)
(2) Torch_scatter is a wheel that may be used in the configuration environment.
(3) DASA-NET is a code file.

4. Model Training
----------
Activate the environment
conda activate openpoints
cd DASA-NET /examples/classification

Single card training
CUDA_VISIBLE_DEVICES=1 python main.py \
--config cfgs DASA-NET /cfgs/modelnet40ply2048/pointnext-FPPS.yaml
Config options: assanet-l.yaml, pointnext-s.yaml, deepgcn.yaml

After running, the log results and model files will have the following path:
DASA-NET /examples/classification/log/modelnet40ply2048
The project provides up-to-date training weights for everyone to use.

Module Configuration
Enable/disable modules in DASA-NET\openpoints\models\backbone\pointnextFPPS:
Modules	Location	Operation
FRP	Lines 242-279	Comment lines 243-279 → Enable line 242
HeadDiffConv	layers/diffConv2	Import and call (enabled by default)
AFDS 	Each Set Abstraction layer starts	Replace the original downsampling method

5. Prediction Results 
----------
Test results on 24 tree species (6,312 point clouds):
Model	OA	mAcc
DASA-Net	96.62%	96.58%
PointNeXt	95.79%	95.88%
ASSANet	95.25%	95.21%



