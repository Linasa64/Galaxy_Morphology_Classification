# Galaxy Morphology Classification

## Table of Contents
- [Abstract](#abstract)
- [Environment](#environment)
- [Dataset](#dataset)

## Abstract
This work explores the development of a computationally efficient Convolutional Neural Network (CNN) for classifying galaxies based on their morphology. Driven by the need for a resource-conscious and scalable solution, we prioritize a lightweight CNN architecture designed to operate effectively on affordable hardware with minimal environmental impact.

To enhance flexibility and accommodate varying classification needs, we implement a hierarchical approach. The model progressively refines classifications, mimicking the Hubble Sequence by dividing galaxies into increasingly specific categories in a stage-wise manner. This hierarchical structure allows astronomers to choose the desired level of granularity, balancing classification accuracy with computational cost.

We evaluate our model on the EFIGI dataset, aiming to demonstrate
its ability to achieve high accuracy while minimizing resource demands. This research aims to provide astronomers with a practical and sustainable tool for efficiently classifying galaxies in large-scale astronomical surveys.

## Environment
0. Python 3.8.10
1. Create a new virtualPython environment : ``python3 -m venv envgalaxies``
2. Activate it : ``source envgalaxies/bin/activate``
3. Install the packages, with the given `requirements.txt` file : ``pip install -r requirements.txt``
This requirements were built downloading the following packages : 
- numpy
- matplotlib
- pandas
- scikit-learn
- astropy
- keras
- tensorflow
- torch
- opencv-python
- tqdm

## Dataset
The dataset used is available [here](https://www.astromatic.net/projects/efigi/). 
It is called the EFIGI dataset, and is described in [this](https://arxiv.org/pdf/1103.5734) paper.
