
# DecoyDB: A Dataset for Graph Contrastive Learning in Protein-Ligand Binding Affinity Prediction

🔧[Code](https://github.com/spatialdatasciencegroup/Decoy_DB), 📂[Dataset](https://huggingface.co/datasets/YupuZ/DecoyDB)

This repository provides benchmark implementations for evaluating **DecoyDB**, a large-scale dataset designed to facilitate **graph contrastive learning** in **protein-ligand binding affinity prediction**. It includes models across three categories: CNN-based methods, sequence-based DTA methods, and GNN-based methods (including contrastive pretraining frameworks).

```
.
├── README                  # This file
├── environment.yml         # Conda enviroment
├── CNN
│   ├── Onion
│   └── Pafnucy
├── DTA
│   ├── DeepDTA
│   └── GraphDTA
└── GNN
    ├── ConBAP
    ├── EGNN
    ├── Frad
    ├── GIGN        
    └── SchNet
```

---

## Overview

- **CNN/**  
  3D‐convolutional networks (e.g. OnionNet, Pafnucy) that voxelize the protein–ligand complex.

- **DTA/**  
  Sequence‐based architectures (DeepDTA, GraphDTA) treating protein and ligand as 1D strings or graphs.

- **GNN/**  
  Graph Neural Networks operating on atomic graphs of protein-ligand complexes. Includes both base models (EGNN, SchNet, GIGN, TorchMD-Net) and pretraining methods (ConBAP, Frad).

## Dataset
All data used in this paper are publicly available and can be accessed here:

Pdbbind: http://www.pdbbind.org.cn/download.php

DecoyDB: https://huggingface.co/datasets/YupuZ/DecoyDB/

## Environment

To ensure reproducibility, all dependencies are specified in the provided environment.yml file.

```bash
conda env create -f environment.yml
```
This will create a new Conda environment with all necessary packages installed.

## Usage:

We use GIGN (a GNN-based model) as a running example to illustrate the full pipeline. Before running any scripts, make sure to update the paths in the corresponding Python files.

### 1. Prepare pre-train data and fine-tune data
#### 1.1 Convert PDBQT Files in DecoyDB to PDB 
```bash
python ./GNN/GIGN/processqt.py
```
This script converts *.pdbqt decoy files into individual .pdb files where each file contains exactly one decoy structure.

#### 1.2 Convert PDB Files to Graph Representations
``` bash
python preprocessing.py
```
This generates intermediate graph representations suitable for spatial learning.
#### 1.3 Convert Graphs to PyTorch Geometric Format
``` bash
python predataset.py
```
This step serializes the graphs into PyTorch Geometric-compatible format.
#### 1.4 Generate the labeled dataset for fine-tuning (e.g., from PDBbind)
```bash
python preprocess_GIGN.py
```
This script processes labeled complexes (with known binding affinities) into graph structures. 
#### 1.5 Convert Fine-Tuning Graphs to PyTorch Geometric Format
```bash
python dataset_GIGN.py
```
This step converts the processed labeled graphs into a PyTorch Geometric-compatible dataset that can be used for supervised fine-tuning.
### 2. Pretrain GIGN with our customized Contrastive Learning

```bash
python pretrain_GIGN.py
```

Models and logs will be saved in `GNN/GIGN/models/`.

### 3. Fine-tune on Binding Affinity Data

```bash
python train_GIGN.py  1 -1000
```

The first argument 1 indicates that the model will load the pretrained weights (the model path need to be changed in the code).

The second argument -1000 means the last 1000 samples of the training set will be used as the validation set.

## Acknowledgements
This project builds upon several publicly available codebases. We sincerely thank the original authors and contributors of the following works:

Pafnucy: https://github.com/oddt/pafnucy

OnionNet: https://github.com/zhenglifang/OnionNet

DeepDTA: https://github.com/hkmztrk/DeepDTA

GraphDTA: https://github.com/thinng/GraphDTA

SchNet, EGNN, GIGN: https://github.com/guaguabujianle/GIGN

ConBAP: https://github.com/ld139/ConBAP

Frad: https://github.com/fengshikun/FradNMI

We thank these authors for making their code publicly available, which provided the foundation for many of the experiments in this repository.