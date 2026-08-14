"""Functions for running inference with the RVFAC regression module."""

import os
import os.path

import click
import echonet
import numpy as np
import torch
import torchvision
import wget

import warnings
warnings.filterwarnings("ignore")


@click.command("rvfac_inference")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", type=click.Path(file_okay=False))
def run(data_dir, output):
    os.makedirs(output, exist_ok=True)

    # Download model weights
    DestinationForWeights = "weights"
    rvfacWeightsURL = "https://github.com/echonet/RV/releases/download/v1/video.pt"

    if not os.path.exists(os.path.join(DestinationForWeights, os.path.basename(rvfacWeightsURL))):
        os.makedirs(DestinationForWeights, exist_ok=True)
        print("Downloading weights for the RVFAC regression module (", rvfacWeightsURL, ") to ",
              os.path.join(DestinationForWeights, os.path.basename(rvfacWeightsURL)), ".", sep="")
        wget.download(rvfacWeightsURL, out=DestinationForWeights)
    else:
        print("Weights for the RVFAC regression module have already been downloaded!")
            
    # Initialize and run the RVFAC regression module
    model = torchvision.models.video.r2plus1d_18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    if torch.cuda.is_available():
        print("CUDA is available: using GPU acceleration.")
        device = torch.device("cuda")
        model = torch.nn.DataParallel(model)
        model.to(device)
        checkpoint = torch.load(os.path.join(DestinationForWeights, os.path.basename(rvfacWeightsURL)))
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("CUDA is not available: using CPU.")
        device = torch.device("cpu")
        checkpoint = torch.load(os.path.join(DestinationForWeights, os.path.basename(rvfacWeightsURL)),
                                map_location="cpu")
        state_dict_cpu = {k[7:]: v for (k, v) in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict_cpu)

    output_csv = os.path.join(output, "rvfac_prediction_results.csv")
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    # Sample five 32-frame clips per video for test-time augmentation
    kwargs = {"target_type": "EF",
              "mean": mean,
              "std": std,
              "length": 32,
              "period": 2,
              "clips": 5,
              }

    ds = echonet.datasets.Echo(split="external_test", external_test_location=data_dir, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0, shuffle=True,
                                                  pin_memory=(device.type == "cuda"))

    # Run inference and average predictions across sampled clips to obtain video-level RVFAC
    loss, yhat, y = echonet.utils.video.run_epoch(model, test_dataloader, False, None, device, save_all=False)

    # Save predicted RVFAC values to CSV
    with open(output_csv, "w") as g:
        g.write("FileName,PredictedRVFAC\n")
        for (filename, pred) in zip(ds.fnames, yhat):
            if isinstance(pred, np.float32) or isinstance(pred, np.float64):
                g.write("{},{:.4f}\n".format(filename, pred))
            else:
                for (i, p) in enumerate(pred):
                    g.write("{},{},{:.4f}\n".format(filename, i, p))
