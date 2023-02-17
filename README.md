# Noise Source Estimation
- This repository contains source code that corresponds to the paper: "Estimating the Noise Sources of a Camera System from an Image and Metadata" (ICCV 2023 submission with ID 1770).
- We propose three versions for noise source estimation: without metadata (*w/o-Meta*), with minimal metadata (*Min-Meta*) and with the full set of metadata (*Full-Meta*).

## General Information
All python scripts that are intended to be executable have a commented out example code at the end. Before you run any scripts, please uncomment and customize the respective code blocks first. 

## Installation Requirements
To install all necessary python packages that we used for testing, run:
```
pip install -r requirements.txt
```

## Testing
We added the pre-trained models with benchmarking data to this repository.
To test one of our proposed models, customize and run:
```
python noise/estimation/NoiseSourceEstimation/estimator.py
```

## Training
Our models can be re-trained on own datasets.

- To noise own images, customize and run:
```
python noise/simulation/generateHighLevelNoise.py
```

- To create a training dataset in the expected .h5 format, customize and run:
```
python noise/estimation/NoiseSourceEstimation/prepareDataset.py
```

- To start the training process, customize and run:
```
python noise/estimation/NoiseSourceEstimation/estimator.py
```

## Benchmarking
We compare our noise estimators against four noise estimators from the litarature (referenced at the bottom of this readme file): 
- B+F, PCA, $\text{DRNE}_\text{cust.}$, and PGE-Net in the case of total noise estimation.
- PGE-Net for additional photon shot noise estimation.

A subset of benchmarking data can be found in the data directory.
We set the Udacity dataset as default.

### B+F
Customize and run:
```
python noise/estimation/B+F/classicNoiseEstimation_B+F.py
```

### $\text{DRNE}_\text{cust.}$
Use the model type "baseline" and run:
```
python noise/estimation/NoiseSourceEstimation/estimator.py
```

### PCA
Customize and run:
```
python noise/estimation/PCA/classicNoiseEstimation_PCA.py
```

### PGE-Net
Customize and run:
```
python noise/estimation/PGE-Net/evaluate_pge.py
```

## Citations
B+F:
```bibtex
@article{shin2005block,
  title={Block-based noise estimation using {A}daptive {G}aussian {F}iltering},
  author={Shin, Dong-Hyuk and Park, Rae-Hong and Yang, Seungjoon and Jung, Jae-Han},
  journal={{IEEE} Trans. Consumer Electronics},
  volume={51},
  number={1},
  pages={218--226},
  year={2005}
}
```

$\text{DRNE}_\text{cust.}$ is inspired by:
```bibtex
@article{tan2019pixelwise,
  title={Pixelwise estimation of signal-dependent image noise using deep residual learning},
  author={Tan, Hanlin and Xiao, Huaxin and Lai, Shiming and Liu, Yu and Zhang, Maojun},
  journal={Computational intelligence and neuroscience},
  volume={2019},
  year={2019},
  publisher={Hindawi}
}

@misc{tan2019pixelwiseGitHub,
  author = {Hanlin Tan},
  title = {Pixel-wise-Estimation-of-Signal-Dependent-Image-Noise},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TomHeaven/Pixel-wise-Estimation-of-Signal-Dependent-Image-Noise-using-Deep-Residual-Learning}},
  commit = {7f2a573}
}
```

PCA:
```bibtex
@InProceedings{Chen15iccv,
  author={Chen, Guangyong and Zhu, Fengyuan and Heng, Pheng Ann},
  booktitle=iccv, 
  title={An Efficient Statistical Method for Image Noise Level Estimation}, 
  year={2015},
  pages={477-485},
  doi={10.1109/ICCV.2015.62}
}

@misc{chen2015efficientGitHub,
  author = {Zongsheng Yue},
  title = {Noise Level Estimation for Signal Image},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zsyOAOA/noise_est_ICCV2015}},
  commit = {a53b4dd}
}
```

PGE-Net:
```bibtex
@inproceedings{byun2021fbi,
  title={Fbi-denoiser: Fast blind image denoiser for poisson-gaussian noise},
  author={Byun, Jaeseok and Cha, Sungmin and Moon, Taesup},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5768--5777},
  year={2021}
}
```
