"""Functions for running inference with the RV segmentation module."""

import os
import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm
import wget

import echonet
from echonet.utils.segmentation import _video_collate_fn

import warnings
warnings.filterwarnings("ignore")


@click.command("segmentation_inference")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", type=click.Path(file_okay=False))
def run(data_dir, output):
    # Download model weights
    DestinationForWeights = "weights"
    segmentationWeightsURL = "https://github.com/echonet/RV/releases/download/v1/segmentation.pt"

    if not os.path.exists(os.path.join(DestinationForWeights, os.path.basename(segmentationWeightsURL))):
        os.makedirs(DestinationForWeights, exist_ok=True)
        print("Downloading weights for the segmentation module (", segmentationWeightsURL, ") to ",
              os.path.join(DestinationForWeights, os.path.basename(segmentationWeightsURL)), ".", sep="")
        wget.download(segmentationWeightsURL, out=DestinationForWeights)
    else:
        print("Weights for the RV segmentation module have already been downloaded!")

    np.random.seed(0)
    torch.manual_seed(0)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "segmentation")
    os.makedirs(output, exist_ok=True)

    # Initialize the RVFAC regression module
    model = torchvision.models.segmentation.__dict__["deeplabv3_resnet50"](pretrained=False, aux_loss=False)
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 3,
                                           kernel_size=model.classifier[-1].kernel_size)

    if torch.cuda.is_available():
        print("CUDA is available: using GPU acceleration.")
        device = torch.device("cuda")
        model = torch.nn.DataParallel(model)
        model.to(device)
        checkpoint = torch.load(os.path.join(DestinationForWeights, os.path.basename(segmentationWeightsURL)))
        state_dict = checkpoint["state_dict"]
        state_dict = {key: state_dict[key] for key in state_dict if key[:22] != "module.aux_classifier."}
        model.load_state_dict(state_dict)
    else:
        print("CUDA is not available: using CPU.")
        device = torch.device("cpu")
        checkpoint = torch.load(os.path.join(DestinationForWeights, os.path.basename(segmentationWeightsURL)),
                                map_location="cpu")

        state_dict = checkpoint["state_dict"]
        state_dict = {key: state_dict[key] for key in state_dict if key[:22] != "module.aux_classifier."}
        state_dict_cpu = {key.replace("module.", ""): state_dict[key] for key in state_dict}
        model.load_state_dict(state_dict_cpu)

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    dataset = echonet.datasets.Echo(external_test_location=data_dir, split="EXTERNAL_TEST",
                                    target_type=["Filename"],
                                    mean=mean, std=std,
                                    length=None, max_length=None, period=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=0, shuffle=False,
                                             pin_memory=False, collate_fn=_video_collate_fn)

    # Generate and save segmentation output videos
    if not all(os.path.isfile(os.path.join(output, "videos", f)) for f in dataloader.dataset.fnames):
        model.eval()

        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output, "size"), exist_ok=True)
        echonet.utils.latexify()

        with torch.no_grad():
            with open(os.path.join(output, "size.csv"), "w") as g:
                g.write("Filename,Frame,Size,ComputerSmall\n")
                for (x, filenames, length) in tqdm.tqdm(dataloader):
                    # Run the segmentation module on individual frames
                    # The concatenated batch of video frames may be too large to process at once
                    y = np.concatenate([model(x[i:(i + 1), :, :, :].to(device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], 1)])
                    filenames = [list(f) for f in filenames]

                    start = 0
                    x = x.numpy()
                    for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                        # Extract one video and its segmentation outputs
                        filename = ''.join(filename)
                        video = x[start:(start + offset), ...]
                        logit = y[start:(start + offset), 0, :, :]

                        # Denormalize the video
                        video *= std.reshape(1, 3, 1, 1)
                        video += mean.reshape(1, 3, 1, 1)

                        # Get frames, channels, height, and width
                        f, c, h, w = video.shape  # pylint: disable=W0612
                        assert c == 3

                        # Place two copies of the video side by side
                        video = np.concatenate((video, video), 3)

                        # Saturate the blue channel for pixels within the segmentation mask
                        video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])  # pylint: disable=E1111

                        # Add blank canvas below the pair of videos
                        video = np.concatenate((video, np.zeros_like(video)), 2)

                        # Compute the size of the segmented area for each frame
                        size = (logit > 0).sum((1, 2))

                        # Identify end-systolic frames with peak detection
                        trim_min = sorted(size)[round(len(size) ** 0.05)]
                        trim_max = sorted(size)[round(len(size) ** 0.95)]
                        trim_range = trim_max - trim_min
                        systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])

                        # Write frame-level segmentation sizes and end-systolic labels to file
                        for (frame, s) in enumerate(size):
                            g.write("{},{},{},{}\n".format(filename, frame, s, 1 if frame in systole else 0))
                        
                        # Plot segmented area over time
                        fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                        plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                        ylim = plt.ylim()
                        for s in systole:
                            plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                        plt.ylim(ylim)
                        plt.title(filename)
                        plt.xlabel("Seconds")
                        plt.ylabel("Size (pixels)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output, "size", filename.split('.')[0] + ".pdf"))
                        plt.close(fig)

                        # Normalize sizes to [0, 1]
                        size -= size.min()
                        size = size / size.max()
                        size = 1 - size

                        # Iterate over the frames in the video
                        for (f, s) in enumerate(size):

                            # Plot the normalized segmented area for each frame
                            video[:, :, int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))] = 255.

                            if f in systole:
                                # Mark computer-selected end-systolic frames with a vertical line
                                video[:, :, 115:224, int(round(f / len(size) * 200 + 10))] = 255.

                            # Get pixel coordinates for a circle centered at the current data point
                            r, c = skimage.draw.disk((int(round(115 + 100 * s)),
                                                      int(round(f / len(size) * 200 + 10))), 4.1)

                            # Highlight the current frame's data point with a circle
                            video[f, :, r, c] = 255.

                        # Rearrange dimensions and save
                        video = video.transpose(1, 0, 2, 3)
                        video = video.astype(np.uint8)
                        echonet.utils.savevideo(os.path.join(output, "videos", filename), video, 50)
                        
                        # Advance to the next video in the concatenated batch
                        start += offset
