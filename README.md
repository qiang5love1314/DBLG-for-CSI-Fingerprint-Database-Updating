# DBLG for CSI Fingerprint Database Updating

## Project Title & Description

This repository contains the implementation of the algorithms proposed in our paper, "DBLG: An Innovative Deep-Broad Learning and GAN Framework for CSI Fingerprint Database Refinement." It integrates a Deep-Broad Learning System (Deep-BLS) with a Generative Adversarial Network (GAN) to enable fine-grained updates of CSI fingerprint databases.

## Overview

With the rapid advancement of Integrated Sensing and Communication (ISAC) in 6G, CSI-based fingerprinting has emerged as a key technology for indoor localization. However, during the offline phase, fingerprint databases constructed via crowdsourcing are often affected by noise interference, incomplete spatial coverage, and the high computational cost of Gaussian regression models. In practical applications, accurately updating the fingerprint database during the offline stage is essential, as its quality directly determines the final localization performance.

## Our Contribution

In this repository, we present DBLG — a novel fingerprint database update framework that integrates a Deep-Broad Learning System (Deep-BLS) with Generative Adversarial Networks (GAN) to enable fine-grained CSI fingerprint refinement.

The DBLG framework includes three key stages:

1. **Initial Database Construction**  
   A Deep-BLS model is used to build a stable global CSI fingerprint database from the raw data.

2. **GAN-based Refinement**  
   Utilize GAN to extract features from the raw data, predict and update the global CSI fingerprint database.

3. **Final Database Generation**  
   The system incorporates confidence coefficients to construct a high-precision fingerprint database.

## Experimental Results

We evaluate DBLG in two real-world indoor environments. Compared to several existing methods, DBLG achieves up to 46.78% improvement in localization accuracy while maintaining efficiency and scalability in large-scale deployment.

## Installation & Usage
### Dependencies
Before using this code, you need to install the following dependencies:
```python
pip install -r requirements.txt
```
### Obtaining the Dataset
To use the dataset, look at [CSI-dataset-for-indoor-localization](https://github.com/qiang5love1314/CSI-dataset-for-indoor-localization). In this paper, we used the CSI data of the lab and meeting room.

## Citation

[Paper link](https://ieeexplore.ieee.org/document/11036427)

If you use this work for your research, you may want to cite:

```latex
@INPROCEEDINGS{zhang2024DBLG,
  author={Zhang, Mingbo and Lu, Lingyun and Zhu, Xiaoqiang and Li, Lingkun and Gao, Ruipeng},
  booktitle={2024 20th International Conference on Mobility, Sensing and Networking (MSN)}, 
  title={DBLG: An Innovative Deep-Broad Learning and GAN Framework for CSI Fingerprint Database Refinement}, 
  year={2024},
  pages={114-121},
  doi={10.1109/MSN63567.2024.00026}}
```
