# EchoNet-RV: A Deep Learning Model for the Automated Echocardiographic Assessment of Right Ventricular Function

EchoNet-RV is a deep learning model that enables frame-by-frame segmentation of the right ventricle (RV) and prediction of RV fractional area change (RVFAC) from apical four-chamber (A4C) echocardiographic videos. In the final version of EchoNet-RV, the segmentation module and the RVFAC inference module operate independently; therefore, RVFAC prediction is entirely segmentation-free.

> [**Artificial Intelligence-Enabled Echocardiographic Assessment of Right Ventricular Function**](https://pubmed.ncbi.nlm.nih.gov/41646670/)<br/>
  Márton Tokodi, Bryan He, Andrea Ferencz, Ádám Szijártó, Kai Shiida, Máté Tolvaj, Alexandra Fábián, Marcell Illyés, Milos Vukadinovic, Andreas Østvik, Vegard Holmstrøm, Bjørnar Grenne, Béla Merkely, Susan Cheng, Yasufumi Nagata, Masaaki Takeuchi, Chung-Lieh Hung, Attila Kovács, David Ouyang<br/>
  <b>(under review)</b> (2026)

## Clinical Significance


## Structure of EchoNet-RV


## Datasets Used for Model Development and Evaluation


## Performance of the Deep Learning Model


Installation
------------

If you want to run EchoNet-RV on a CUDA-enabled GPU, ensure that the version of the CUDA Toolkit installed on your computer supports the version of PyTorch specified in requirements.txt

Clone this repository to the desired destination using the following commend:

    git clone https://github.com/echonet/rv.git

Create a Python virtual environment (version 3.10) dedicated to this project and activate it

The code and its dependencies can be installed by navigating to the cloned directory and running

    pip install .
    pip install -r requirements.txt

Usage
-----

### Preprocess DICOM files

    python -m echonet preprocess --data_dir /path/to/dicom/files --output /path/to/output/videos --crop_size 112 112
  
  This creates a directory named `/path/to/output/videos`, which will contain the preprocessed videos in `.avi` format.

### Model training and evaluation

#### Semantic segmentation of the right ventricle

    python -m echonet segmentation --save_video

This creates a directory named `output/segmentation/deeplabv3_resnet50_random/`, which will contain
  - log.csv: training and validation losses
  - best.pt: checkpoint of weights for the model with the lowest validation loss
  - size.csv: estimated size of left ventricle for each frame and indicator for beginning of beat
  - videos: directory containing videos with segmentation overlay

#### Prediction of RVFAC

  python -m echonet video

This creates a directory named `output/video/r2plus1d_18_32_2_pretrained/`, which will contain
  - log.csv: training and validation losses
  - best.pt: checkpoint of weights for the model with the lowest validation loss
  - test_predictions.csv: RVFAC prediction for the sampled clips

### Inference

NOTE: Inference scripts wil download the mode weight automatically.

#### Semantic segmentation of the right ventricle

    python -m echonet segmentation_inference --data_dir /path/to/input/videos --output /path/to/output/videos

#### Prediction of RVFAC

    python -m echonet video_inference --data_dir /path/to/input/videos --output /path/to/output/videos
