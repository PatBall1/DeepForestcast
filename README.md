# DeepFore[st]cast

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![DOI](https://zenodo.org/badge/431184218.svg)](https://zenodo.org/badge/latestdoi/431184218)

Deep convolutional neural networks to forecast tropical deforestation.

## Citation

Please cite:

Ball, J. G. C., Petrova, K., Coomes, D. A., & Flaxman, S. (2022). Using deep convolutional neural networks to forecast spatial patterns of Amazonian deforestation. *Methods in Ecology and Evolution*, 13, 2622– 2634. [https://doi.org/10.1111/2041-210X.13953](https://doi.org/10.1111/2041-210X.13953)


## Requirements
- Python 3.8+
- scikit-learn
- torch 1.9.0
- torchaudio 0.9.0
- torchvision 0.10.0


## Introduction

Tropical forests are subject to diverse deforestation pressures but their conservation is essential to achieve global climate goals. Predicting the location of deforestation is challenging due to the complexity of the natural and human systems involved but accurate and timely forecasts could enable effective planning and on-the-ground enforcement practices to curb deforestation rates. New computer vision technologies based on deep learning can be applied to the increasing volume of Earth observation data to generate novel insights and make predictions with unprecedented accuracy.

Here, we demonstrate the ability of deep convolutional neural networks to learn spatiotemporal patterns of deforestation from a limited set of freely available global data layers, including multispectral satellite imagery, the Hansen maps of historic deforestation (2001-2020) and the ALOS JAXA digital surface model, to forecast future deforestation (2021). We designed four original deep learning model architectures, based on 2D Convolutional Neural Networks (2DCNN), 3D Convolutional Neural Networks (3DCNN), and Convolutional Long Short-Term Memory (ConvLSTM) Recurrent Neural Networks (RNN) to produce spatial maps that indicate the risk to each forested pixel (~30 m) in the landscape of becoming deforested within the next year. They were trained and tested on data from two ~80,000 km2 tropical forest regions in the Southern Peruvian Amazon.

## Getting started

[Preprint available](https://www.biorxiv.org/content/10.1101/2021.12.14.472442v1.full)

## Forecast

Deep CNNs can predict how deforestation frontiers are likely to evolve.

Some examples from the Southern Peruvian Amazon:

a) Agricultural expansion  
b) Illegal gold mine  
c) New forest road  
d) Remote landslide*

*unlikely for forest loss to continue

![Example forecast](/report/figures/ForecastExamples.png)

## Networks

Schematic of network

![Network](/report/figures/schematicOfNetwork.png#gh-light-mode-only)
![Network](/report/figures/schematicOfnetworkDARK.png#gh-dark-mode-only)

[2D CNN architecture](/src/models/2DCNN.py)

![2DCNN](/report/figures/2DCNNmodel.PNG)

[3D CNN architecture](/src/models/ConvRNN.py)

![3DCNN](/report/figures/3DConvModel.PNG)

[ConvLSTM architecture](/src/models/ConvRNN.py)

![ConvLSTM](/report/figures/LSTMmodels.PNG)



## Project Organization
```
├── LICENSE
├── Makefile           <- Makefile with commands like `make init` or `make lint-requirements`
├── README.md          <- The top-level README for developers using this project.
|
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
|   |                     the creator's initials, and a short `-` delimited description, e.g.
|   |                     `1.0_jqp_initial-data-exploration`.
│   ├── exploratory    <- Notebooks for initial exploration.
│   └── reports        <- Polished notebooks for presentations or intermediate results.
│
├── report             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── sections       <- LaTeX sections. The report folder can be linked to your overleaf
|                         report with github submodules.
│
├── requirements       <- Directory containing the requirement files.
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data_loading   <- Scripts to download or generate data
│   │
│   ├── preprocessing  <- Scripts to turn raw data into clean data and features for modeling
|   |
│   ├── models         <- Scripts to train models and then use trained models to make
│   │                     predictions
│   │
│   └── tests          <- Scripts for unit tests of your functions
│
└── setup.cfg          <- setup configuration file for linting rules
```

## Code formatting
To automatically format your code, make sure you have `black` installed (`pip install black`) and call
```black . ``` 
from within the project directory.

---

Project template created by the [Cambridge AI4ER Cookiecutter](https://github.com/ai4er-cdt/ai4er-cookiecutter).
