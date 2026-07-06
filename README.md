Installation
------------

First, clone this repository and enter the directory by running:

    git clone https://github.com/echonet/rv.git
    cd rv


The code and its dependencies can be installed by navigating to the cloned directory and running

    pip install --user .

Usage
-----

### Preprocess the dicom files into videos

    python -m echonet preprocess --data_dir /path/to/dicom/files --output /path/to/output/videos --crop_size 112 112
  
  This creates a directory named `/path/to/output/videos`, which will contain the preprocessed videos in `.avi` format.

### Training and evaluating a model

#### Frame-by-frame Semantic Segmentation of the Left Ventricle

    python -m echonet segmentation --save_video

This creates a directory named `output/segmentation/deeplabv3_resnet50_random/`, which will contain
  - log.csv: training and validation losses
  - best.pt: checkpoint of weights for the model with the lowest validation loss
  - size.csv: estimated size of left ventricle for each frame and indicator for beginning of beat
  - videos: directory containing videos with segmentation overlay

#### Prediction of RVFAC from Subsampled Clips

  python -m echonet video

This creates a directory named `output/video/r2plus1d_18_32_2_pretrained/`, which will contain
  - log.csv: training and validation losses
  - best.pt: checkpoint of weights for the model with the lowest validation loss
  - test_predictions.csv: ejection fraction prediction for subsampled clips

### Inference

#### Segmentation of the Left Ventricle

    python -m echonet segmentation_inference --data_dir /path/to/input/videos --output /path/to/output/videos

#### Prediction of RVFAC from Subsampled Clips

    python -m echonet video_inference --data_dir /path/to/input/videos --output /path/to/output/videos