# SOAD

This repo contains the offical PyTroch code for **Self-supervised Denoising and Bulk Motion Artifact Removal of 3D Optical Coherence Tomography Angiography of Awake Brain** @ MICCAI 2024

## Overview
<img title="Overview" alt="Overview" src="figures/pipeline.png">

## Result
<img title="Result" alt="Result" src="figures/result.png">

## Instructions

Checkpoint, example data and ROI labels are available at [data link](https://drive.google.com/drive/folders/12xubKEdMbBcUJ0Gf_Rz5fGhvo-SRAXKE?usp=sharing) 

Besides training `octa_train.py` and testing `octa_test.py` scripts, `cnr_msr_normal.py` and `cnr_msr_corrupted.py` are used to calculate CNR and MSR based on the ROI labels and visualize the scores as well as the ROI bounding boxes.

## Acknowledgments

This repo mainly refers to [UDVD](https://github.com/sreyas-mohan/udvd) and [Magic-VNet](https://github.com/Hsuxu/Magic-VNet) for the training scripts and network architecture.