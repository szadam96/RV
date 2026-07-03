"""Functions for training and running segmentation."""

import math
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm

import echonet
from echonet.utils.segmentation import _video_collate_fn

import wget

@click.command("segmentation_inference")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", type=click.Path(file_okay=False))
def run(
    data_dir,
    output,
):
    # Download model weights
    DestinationForWeights = "weights"

    if os.path.exists(DestinationForWeights):
        print("The weights are at", DestinationForWeights)
    else:
        print("Creating folder at ", DestinationForWeights, " to store weights")
        os.mkdir(DestinationForWeights)
        
    segmentationWeightsURL = 'https://github.com/echonet/RV/releases/download/v1/segmentation.pt'


    if not os.path.exists(os.path.join(DestinationForWeights, os.path.basename(segmentationWeightsURL))):
        print("Downloading Segmentation Weights, ", segmentationWeightsURL," to ",os.path.join(DestinationForWeights,os.path.basename(segmentationWeightsURL)))
        filename = wget.download(segmentationWeightsURL, out = DestinationForWeights)
    else:
        print("Segmentation Weights already present")


    # Seed RNGs
    np.random.seed(0)
    torch.manual_seed(0)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "segmentation")
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = torchvision.models.segmentation.__dict__['deeplabv3_resnet50'](pretrained=True, aux_loss=False)

    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)  # change number of outputs to 1
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    weights = os.path.join(DestinationForWeights, 'deeplabv3_resnet50_random.pt')

    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    state_dict = {key: state_dict[key] for key in state_dict if key[:22] != "module.aux_classifier."}
    model.load_state_dict(state_dict)
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    # Saving videos with segmentations
    dataset = echonet.datasets.Echo(external_test_location=data_dir, split="EXTERNAL_TEST",
                                    target_type=["Filename"],  # Need filename for saving, and human-selected frames to annotate
                                    mean=mean, std=std,  # Normalization
                                    length=None, max_length=None, period=1  # Take all frames
                                    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=0, shuffle=False, pin_memory=False, collate_fn=_video_collate_fn)

    # Save videos with segmentation
    if not all(os.path.isfile(os.path.join(output, "videos", f)) for f in dataloader.dataset.fnames):
        # Only run if missing videos

        model.eval()

        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output, "size"), exist_ok=True)
        echonet.utils.latexify()

        with torch.no_grad():
            with open(os.path.join(output, "size.csv"), "w") as g:
                g.write("Filename,Frame,Size,ComputerSmall\n")
                for (x, filenames, length) in tqdm.tqdm(dataloader):
                    # Run segmentation model on blocks of frames one-by-one
                    # The whole concatenated video may be too long to run together
                    y = np.concatenate([model(x[i:(i + 1), :, :, :].to(device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], 1)])
                    filenames = [list(f) for f in filenames][0]

                    start = 0
                    x = x.numpy()
                    for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                        # Extract one video and segmentation predictions
                        video = x[start:(start + offset), ...]
                        logit = y[start:(start + offset), 0, :, :]

                        # Un-normalize video
                        video *= std.reshape(1, 3, 1, 1)
                        video += mean.reshape(1, 3, 1, 1)

                        # Get frames, channels, height, and width
                        f, c, h, w = video.shape  # pylint: disable=W0612
                        assert c == 3

                        # Put two copies of the video side by side
                        video = np.concatenate((video, video), 3)

                        # If a pixel is in the segmentation, saturate blue channel
                        # Leave alone otherwise
                        video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])  # pylint: disable=E1111

                        # Add blank canvas under pair of videos
                        video = np.concatenate((video, np.zeros_like(video)), 2)

                        # Compute size of segmentation per frame
                        size = (logit > 0).sum((1, 2))

                        # Identify systole frames with peak detection
                        trim_min = sorted(size)[round(len(size) ** 0.05)]
                        trim_max = sorted(size)[round(len(size) ** 0.95)]
                        trim_range = trim_max - trim_min
                        systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])

                        # Write sizes and frames to file
                        for (frame, s) in enumerate(size):
                            g.write("{},{},{},{}\n".format(filename, frame, s, 1 if frame in systole else 0))

                        # Move to next video
                        start += offset