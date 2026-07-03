import os, os.path
import torch
import torchvision
import echonet
import click
import wget 
import numpy as np

@click.command("video_inference")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", type=click.Path(file_okay=False))
def run(data_dir, output):
    os.makedirs(output, exist_ok = True)
    #DestinationForWeights = "/Users/davidouyang/Dropbox/Echo Research/CodeBase/EchoNetDynamic-Weights"
    DestinationForWeights = "weights"

    # Download model weights

    if os.path.exists(DestinationForWeights):
        print("The weights are at", DestinationForWeights)
    else:
        print("Creating folder at ", DestinationForWeights, " to store weights")
        os.mkdir(DestinationForWeights)
        
    rvfacWeightsURL = "https://github.com/echonet/RV/releases/download/v1/video.pt"

        
    if not os.path.exists(os.path.join(DestinationForWeights, os.path.basename(rvfacWeightsURL))):
        print("Downloading RV FAC Weights, ", rvfacWeightsURL," to ",os.path.join(DestinationForWeights,os.path.basename(rvfacWeightsURL)))
        filename = wget.download(rvfacWeightsURL, out = DestinationForWeights)
    else:
        print("RV FAC Weights already present")
            
    # Initialize and Run EF model
    model = torchvision.models.video.r2plus1d_18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)



    print("loading weights from ", os.path.join(DestinationForWeights, "r2plus1d_18_32_2_pretrained"))

    if torch.cuda.is_available():
        print("cuda is available, original weights")
        #torch.cuda.empty_cache()
        device = torch.device("cuda")
        model = torch.nn.DataParallel(model)
        model.to(device)
        checkpoint = torch.load(os.path.join(DestinationForWeights, os.path.basename(rvfacWeightsURL)))
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("cuda is not available, cpu weights")
        device = torch.device("cpu")
        checkpoint = torch.load(os.path.join(DestinationForWeights, os.path.basename(rvfacWeightsURL)), map_location = "cpu")
        state_dict_cpu = {k[7:]: v for (k, v) in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict_cpu)

    output_csv = os.path.join(output, "output.csv")
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    kwargs = {"target_type": "EF",
            "mean": mean,
            "std": std,
            "length": 32,
            "period": 2,
            "clips": 5,
            }

    ds = echonet.datasets.Echo(split = "external_test", external_test_location = data_dir, **kwargs)
    print(ds.split, ds.fnames)


    ds = echonet.datasets.Echo(split = "external_test", external_test_location = data_dir, **kwargs)

    test_dataloader = torch.utils.data.DataLoader(ds, batch_size = 1, num_workers = 0, shuffle = True, pin_memory=(device.type == "cuda"))
    loss, yhat, y = echonet.utils.video.run_epoch(model, test_dataloader, False, None, device, save_all=False)

    with open(output_csv, "w") as g:
        for (filename, pred) in zip(ds.fnames, yhat):
            if isinstance(pred, np.float32) or isinstance(pred, np.float64):
                g.write("{},{:.4f}\n".format(filename, pred))
            else:
                for (i,p) in enumerate(pred):
                    g.write("{},{},{:.4f}\n".format(filename, i, p))