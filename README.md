# latentOT

Latent optimal transport (LOT) for simultaneous aligning and clustering 



# Overview

Latent optimal transport is a low-rank distributional alignment technique. It is suitable for data admitting clustering structure, where alignment is based on a clustering level to make transport more robust to outliers and noise.

Users could customize their cost matrix to fit their clustering strategies.

The algorithm requires two numbers of anchors to be pre-specified. The numbers naturally correspond to the numbers of clusters for the source and target.



# Dependencies

sklearn, numpy, matplotlib, POT: Python Optimal Transport Rémi Flamary and Nicolas Courty, POT Python Optimal Transport library,

Website: https://pythonot.github.io/, 2017



# Usage

The code contains a Python implementation on LOT.



lOT.py contains the module for LOT.



lot_mnist_demo.ipynb provides a simple demo on distributional alignments for pre- and post-dropout of MNIST digits. 


