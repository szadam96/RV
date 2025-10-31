"""Functions for training and running segmentation."""

import random
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
import PIL
import sklearn
import pickle

import echonet


@click.command("segmentation")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.segmentation.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.segmentation.__dict__[name]))),
    default="deeplabv3_resnet50")
@click.option("--pretrained/--random", default=False)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--full/--last", default=True)
@click.option("--run_test/--skip_test", default=False)
@click.option("--save_video/--skip_video", default=False)
@click.option("--num_epochs", type=int, default=50)
@click.option("--lr", type=float, default=1e-6)
@click.option("--weight_decay", type=float, default=0)
@click.option("--lr_step_period", type=int, default=None)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=8)
@click.option("--batch_size", type=int, default=32)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(
    data_dir=None,
    output=None,

    model_name="deeplabv3_resnet50",
    pretrained=True,
    weights=None,
    full=True,

    run_test=False,
    save_video=False,
    num_epochs=50,
    lr=1e-5,
    weight_decay=1e-5,
    lr_step_period=None,
    num_train_patients=None,
    num_workers=8,
    batch_size=32,
    device=None,
    seed=0,
):
    """Trains/tests segmentation model.

    Args:
        data_dir (str, optional): Directory containing dataset. Defaults to
            `echonet.config.DATA_DIR`.
        output (str, optional): Directory to place outputs. Defaults to
            output/segmentation/<model_name>_<pretrained/random>/.
        model_name (str, optional): Name of segmentation model. One of ``deeplabv3_resnet50'',
            ``deeplabv3_resnet101'', ``fcn_resnet50'', or ``fcn_resnet101''
            (options are torchvision.models.segmentation.<model_name>)
            Defaults to ``deeplabv3_resnet50''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to False.
        weights (str, optional): Path to checkpoint containing weights to
            initialize model. Defaults to None.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
        save_video (bool, optional): Whether to save videos with segmentations.
            Defaults to False.
        num_epochs (int, optional): Number of epochs during training
            Defaults to 50.
        lr (float, optional): Learning rate for SGD
            Defaults to 1e-5.
        weight_decay (float, optional): Weight decay for SGD
            Defaults to 0.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            Defaults to math.inf (never decay learning rate).
        num_train_patients (int or None, optional): Number of training patients
            for ablations. Defaults to all patients.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        device (str or None, optional): Name of device to run on. Options from
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            Defaults to ``cuda'' if available, and ``cpu'' otherwise.
        batch_size (int, optional): Number of samples to load per batch
            Defaults to 32.
        seed (int, optional): Seed for random number generator. Defaults to 0.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "segmentation", "{}_{}".format(model_name, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute mean and std
    if weights is None:
        try:
            with open(os.path.join(output, "mean_std_bias.pkl"), "rb") as f:
                (mean, std, bias) = pickle.load(f)
        except:
            nx = 0  # number of elements taken (should be equal to samples by end of for loop)
            s1 = 0.  # sum of elements along channels (ends up as np.array of dimension (channels,))
            s2 = 0.  # sum of squares of elements along channels (ends up as np.array of dimension (channels,))
            ny = 0
            sy = 0.
            dataset = echonet.datasets.Echo(root=data_dir, split="train", target_type=["Trace"], segmentation=True)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            for (x, t) in tqdm.tqdm(dataloader, desc="Getting dataset statistics"):
                x = x.transpose(0, 1).contiguous().view(3, -1)
                nx += x.shape[1]
                s1 += torch.sum(x, dim=1).numpy()
                s2 += torch.sum(x ** 2, dim=1).numpy()
                ny += t.numel()
                sy += t.sum().item()
            mean = s1 / nx  # type: np.ndarray
            std = np.sqrt(s2 / nx - mean ** 2)  # type: np.ndarray

            mean = mean.astype(np.float32)
            std = std.astype(np.float32)

            bias = sy / ny

            with open(os.path.join(output, "mean_std_bias.pkl"), "wb") as f:
                pickle.dump((mean, std, bias), f)

    # Set up model
    model = torchvision.models.segmentation.__dict__[model_name](pretrained=pretrained, aux_loss=False)

    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 3, kernel_size=model.classifier[-1].kernel_size)  # change number of outputs to 1
    if weights is None:
        model.classifier[-1].weight.data[:] = 0
        model.classifier[-1].bias.data[:] = 0
        model.classifier[-1].bias.data[0] = math.log(bias / (1 - bias))

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights)
        state_dict = checkpoint["state_dict"]
        state_dict = {key: state_dict[key] for key in state_dict if key[:22] != "module.aux_classifier."}
        model.load_state_dict(state_dict)
        mean = checkpoint["mean"]
        std = checkpoint["std"]

    tasks = ["Filename", "Frame", "Trace", "Phase"]
    kwargs = {
        "target_type": tasks,
        "mean": mean,
        "std": std
    }


    # Set up optimizer
    if full:
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        # TODO make robust (breaks without dataparallel)
        optim = torch.optim.SGD(model.module.classifier[-1].parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs, pad=12, segmentation=True)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs, segmentation=True)

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint["state_dict"])
            optim.load_state_dict(checkpoint["opt_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_dict"])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ["train", "val"]:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))

                loss, inter, union, total, total_trace, filename, inter_list, union_list, loss_trace_list, phase = echonet.utils.segmentation.run_epoch(model, dataloader, phase == "train", optim, device)
                overall_dice = 2 * inter.sum() / (union.sum() + inter.sum())
                f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                    epoch,
                    phase,
                    loss,
                    overall_dice,
                    time.time() - start_time,
                    inter.size,
                    sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                    sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                    batch_size,
                    total,
                    total_trace,
                ))
                f.flush()
                os.makedirs(os.path.join(output, "predictions"), exist_ok=True)
                with open(os.path.join(output, "predictions", "epoch_{:03d}_{}.pkl".format(epoch, phase)), "wb") as p:
                    pickle.dump((filename, inter_list, union_list, loss_trace_list), p)
            scheduler.step()

            # Save checkpoint
            save = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_loss": bestLoss,
                "mean": mean,
                "std": std,
                "loss": total,
                "opt_dict": optim.state_dict(),
                "scheduler_dict": scheduler.state_dict(),
            }
            os.makedirs(os.path.join(output, "models"), exist_ok=True)
            torch.save(save, os.path.join(output, "models", "epoch_{:03d}.pt".format(epoch)))
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if total < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = total

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint["state_dict"])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if run_test:
            # Run on validation and test
            for split in ["test"]:
                dataset = echonet.datasets.Echo(root=data_dir, split=split, **kwargs, segmentation=True)
                dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                loss, inter, union, total, total_trace, filename, inter_list, union_list, loss_trace_list, phase = echonet.utils.segmentation.run_epoch(model, dataloader, False, None, device)

                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(inter, union, echonet.utils.dice_similarity_coefficient)))
                f.flush()

                with open(os.path.join(output, f"{split}_predictions.csv"), "w") as g:
                    g.write("Filename,Intersection,Union,Phase\n")
                    for x in zip(filename, inter_list, union_list, phase):
                        g.write("{},{:d},{:d},{}\n".format(*x))
                
    # Saving videos with segmentations
    dataset = echonet.datasets.Echo(root=data_dir, split="test",
                                    # target_type=["Filename", "LargeIndex", "SmallIndex"],  # Need filename for saving, and human-selected frames to annotate
                                    target_type=["Filename"],
                                    mean=mean, std=std,  # Normalization
                                    length=None, max_length=None, period=1  # Take all frames
                                    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=False, collate_fn=_video_collate_fn)

    # Save videos with segmentation
    if save_video:

        model.eval()

        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output, "size"), exist_ok=True)
        echonet.utils.latexify()

        with torch.no_grad():
            with open(os.path.join(output, "size.csv"), "w") as g:
                g.write("Filename,Frame,Size\n")
                for (x, (filenames), length) in tqdm.tqdm(dataloader):
                    # Run segmentation model on blocks of frames one-by-one
                    # The whole concatenated video may be too long to run together
                    y = np.concatenate([model(x[i:(i + batch_size), :, :, :].to(device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], batch_size)])

                    start = 0
                    x = x.numpy()
                    for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                        # Extract one video and segmentation predictions
                        video = x[start:(start + offset), ...]
                        logit = y[start:(start + offset), 0, :, :]


                        # Un-normalize video
                        video *= std.reshape(1, 3, 1, 1)
                        video += mean.reshape(1, 3, 1, 1)

                        os.makedirs("plot/trace", exist_ok=True)
                        copy = video.copy()
                        copy[:, 2, :, :] = np.maximum(255. * (logit > 0), video[:, 2, :, :])  # pylint: disable=E1111
                        for i in range(len(video)):
                            PIL.Image.fromarray(video[i, ...].clip(0, 255).astype(np.uint8).transpose((1, 2, 0))).save(f"plot/trace/{filename.replace('.avi', '')}_raw_{i:03d}.jpg")
                            PIL.Image.fromarray(copy[i, ...].clip(0, 255).astype(np.uint8).transpose((1, 2, 0))).save(f"plot/trace/{filename.replace('.avi', '')}_trace_{i:03d}.jpg")

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
                            g.write("{},{},{}\n".format(filename, frame, s))
                        g.flush()

                        # Plot sizes
                        fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                        plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                        ylim = plt.ylim()
                        for s in systole:
                            plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                        plt.ylim(ylim)
                        plt.title(os.path.splitext(filename)[0])
                        plt.xlabel("Seconds")
                        plt.ylabel("Size (pixels)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output, "size", os.path.splitext(filename)[0] + ".pdf"))
                        plt.close(fig)

                        # Normalize size to [0, 1]
                        size -= size.min()
                        size = size / size.max()
                        size = 1 - size

                        # Iterate the frames in this video
                        for (f, s) in enumerate(size):

                            # On all frames, mark a pixel for the size of the frame
                            video[:, :, int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))] = 255.

                            if f in systole:
                                # If frame is computer-selected systole, mark with a line
                                video[:, :, 115:224, int(round(f / len(size) * 200 + 10))] = 255.

                            def dash(start, stop, on=10, off=10):
                                buf = []
                                x = start
                                while x < stop:
                                    buf.extend(range(x, x + on))
                                    x += on
                                    x += off
                                buf = np.array(buf)
                                buf = buf[buf < stop]
                                return buf
                            d = dash(115, 224)

                            # Get pixels for a circle centered on the pixel
                            r, c = skimage.draw.disk((int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))), 4.1)

                            # On the frame that's being shown, put a circle over the pixel
                            video[f, :, r, c] = 255.

                        # Rearrange dimensions and save
                        video = video.transpose(1, 0, 2, 3)
                        video = video.astype(np.uint8)
                        echonet.utils.savevideo(os.path.join(output, "videos", filename), video, 50)

                        # Move to next video
                        start += offset


def run_epoch(model, dataloader, train, optim, device):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """

    total = 0.
    total_trace = 0.
    total_apex = 0.
    total_base = 0.
    n = 0

    pos = 0
    neg = 0
    pos_pix = 0
    neg_pix = 0

    model.train(train)

    inter = 0
    union = 0
    inter_list = []
    union_list = []
    loss_trace_list = []
    loss_apex_list = {"d": [], "s": []}
    loss_base_list = {"d": [], "s": []}
    filename = []
    phase = []
    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:

            for (_, (fn, frame, trace, p)) in dataloader:
                filename.extend(fn)
                phase.extend(p)
                loss_trace = 0
                loss_apex = 0
                loss_base = 0

                assert not torch.isnan(frame).any()
                assert not torch.isnan(trace).any()

                # Count number of pixels in/out of human segmentation
                pos += (trace == 1).sum().item()
                neg += (trace == 0).sum().item()

                # Count number of pixels in/out of computer segmentation
                pos_pix += (trace == 1).sum(0).to("cpu").detach().numpy()
                neg_pix += (trace == 0).sum(0).to("cpu").detach().numpy()

                # Run prediction for frames and compute loss
                frame = frame.to(device)
                trace = trace.to(device)
                y = model(frame)["out"]
                lt = torch.nn.functional.binary_cross_entropy_with_logits(y[:, 0, :, :], trace, reduction="none").sum((1, 2))
                loss_trace_list.extend(lt.detach().cpu().numpy())
                loss_trace += lt.sum()

                # Compute pixel intersection and union between human and computer segmentations
                inter += np.logical_and(y[:, 0, :, :].detach().cpu().numpy() > 0., trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                union += np.logical_or(y[:, 0, :, :].detach().cpu().numpy() > 0., trace[:, :, :].detach().cpu().numpy() > 0.).sum()

                inter_list.extend(np.logical_and(y[:, 0, :, :].detach().cpu().numpy() > 0., trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                union_list.extend(np.logical_or(y[:, 0, :, :].detach().cpu().numpy() > 0., trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                # Take gradient step if training
                loss = (loss_trace + 10 * loss_apex + 10 * loss_base)
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # Accumulate losses and compute baselines
                total += loss.item()
                total_trace += loss_trace.item()
                n += trace.size(0) # TODO: if some get masked for being missing, this will be off
                p = (pos + 1) / (pos + neg + 2)
                p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)

                # Show info on process bar
                pbar.set_postfix_str("{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}".format(
                    total / n / 112 / 112,
                    loss_trace.item() / trace.size(0) / 112 / 112,
                    -p * math.log(p) - (1 - p) * math.log(1 - p),
                    (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(),
                    2 * inter / (union + inter),
                ))
                pbar.update()

    inter_list = np.array(inter_list)
    union_list = np.array(union_list)

    return (total / n / 112 / 112,
            inter_list,
            union_list,
            total / n,
            total_trace / n,
            filename,
            inter_list,
            union_list,
            loss_trace_list,
            phase,
    )


def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos.

    This function should be used in a dataloader for a dataset that returns
    a video as the first element, along with some (non-zero) tuple of
    targets. Then, the input x is a list of tuples:
      - x[i][0] is the i-th video in the batch
      - x[i][1] are the targets for the i-th video

    This function returns a 3-tuple:
      - The first element is the videos concatenated along the frames
        dimension. This is done so that videos of different lengths can be
        processed together (tensors cannot be "jagged", so we cannot have
        a dimension for video, and another for frames).
      - The second element is contains the targets with no modification.
      - The third element is a list of the lengths of the videos in frames.
    """
    video, target = zip(*x)  # Extract the videos and targets

    # ``video'' is a tuple of length ``batch_size''
    #   Each element has shape (channels=3, frames, height, width)
    #   height and width are expected to be the same across videos, but
    #   frames can be different.

    # ``target'' is also a tuple of length ``batch_size''
    # Each element is a tuple of the targets for the item.

    i = list(map(lambda t: t.shape[1], video))  # Extract lengths of videos in frames

    # This contatenates the videos along the the frames dimension (basically
    # playing the videos one after another). The frames dimension is then
    # moved to be first.
    # Resulting shape is (total frames, channels=3, height, width)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))

    # Swap dimensions (approximately a transpose)
    # Before: target[i][j] is the j-th target of element i
    # After:  target[i][j] is the i-th target of element j
    # target = zip(*target)

    return video, target, i
