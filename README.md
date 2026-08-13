# EchoNet-RV: A Deep Learning Model for the Automated Echocardiographic Assessment of Right Ventricular Function

EchoNet-RV is a deep learning model that enables frame-by-frame segmentation of the right ventricle (RV) and prediction of RV fractional area change (RVFAC) from apical four-chamber (A4C) echocardiographic videos. In the final version of EchoNet-RV, the segmentation module and the RVFAC inference module operate independently; therefore, RVFAC prediction is entirely segmentation-free.

> [**Artificial Intelligence-Enabled Echocardiographic Assessment of Right Ventricular Function**](https://pubmed.ncbi.nlm.nih.gov/41646670/)<br/>
  Márton Tokodi, Bryan He, Andrea Ferencz, Ádám Szijártó, Kai Shiida, Máté Tolvaj, Alexandra Fábián, Marcell Illyés, Milos Vukadinovic, Andreas Østvik, Vegard Holmstrøm, Bjørnar Grenne, Béla Merkely, Susan Cheng, Yasufumi Nagata, Masaaki Takeuchi, Chung-Lieh Hung, Attila Kovács, David Ouyang.<br/><b>(under review)</b> (2026)

## Clinical Significance


## Architecture of EchoNet-RV

EchoNet-RV comprises two key modules:
1) <b>Semantic segmentation module:</b> A DeepLabV3 model with a ResNet-50 backbone that performs frame-level semantic segmentation of the RV cavity.
2) <b>RVFAC regression module:</b> A spatiotemporal convolutional neural network based on the R(2+1)D-18 architecture that directly estimates RVFAC from each video without relying on RV segmentation.

Given that substantial beat-to-beat variation in the end-diastolic and end-systolic RV areas (and thus in RVFAC) may occur in conditions such as atrial fibrillation and premature atrial or ventricular contractions, test-time augmentation was applied to improve the robustness of the final predictions. Briefly, five potentially overlapping 32-frame clips were randomly sampled from each video, and the RVFAC predictions generated for these clips by the RVFAC regression module were averaged to obtain the final video-level prediction.

## Datasets Used for Model Development and Evaluation

<b>Stanford dataset:</b> The Stanford dataset, used for model development and internal validation, comprised a total of 8,489 A4C videos from 8,314 studies of 5,386 patients who underwent transthoracic echocardiography between 2016 and 2018 as part of clinical care at Stanford Health Care (California, USA). Videos were randomly split into three sets of 5,892, 1,277, and 1,320 videos for training, validation, and testing, respectively. Splitting was performed at the patient level to avoid including videos from the same patient in more than one of the three sets.

<b>Semmelweis dataset:</b> The Semmelweis dataset, used for external validation and for assessing the prognostic value of the predicted RVFAC values, consisted of a total of 3,107 A4C videos from 982 studies of 872 patients who underwent transthoracic echocardiography between 2013 and 2021 at the Heart and Vascular of Semmelweis University (Budapest, Hungary).
 
<b>MacKay Memorial Hospital dataset:</b> The MacKay Memorial Hospital (MMH) dataset, also used for external validation, included a total of 1,077 A4C videos from 1,077 studies of 1,077 patients who underwent transthoracic echocardiography between 2009 and 2022 at MacKay Memorial Hospital (Taipei, Taiwan).

<b>University of Occupational and Environmental Health dataset:</b> The third external test set comprised 1,315 A4C videos from 341 studies of 341 patients who underwent transthoracic echocardiography between January 2014 and December 2020 at the University Hospital of the University of Occupational and Environmental Health (UOEH; Kitakyushu, Japan).


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
