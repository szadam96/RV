"""EchoNet-Dynamic Dataset."""

import math
import os
import collections
import pandas
import PIL

import numpy as np
import torch
import torchvision
import echonet


class Echo(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(self, root=None,
                 split="train", target_type="RVFAC",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 max_clips=100,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None,
                 segmentation=False):
        if root is None:
            root = echonet.config.DATA_DIR

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.max_clips = max_clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location
        self.segmentation = segmentation

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            with open(os.path.join(self.root, "filelist.csv")) as f:
                data = pandas.read_csv(f)
            data["Split"] = data["Split"].apply(lambda x: x.upper())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = data["Filename"].tolist()
            self.fnames = [fn if os.path.splitext(fn)[1] != "" else fn + ".avi" for fn in self.fnames]  # Assume avi if no suffix
            self.outcome = data.values.tolist()

            # Load traces
            if segmentation:
                self.frames = collections.defaultdict(list)
                self.segmentations = []

                fnames = set(self.fnames)

                self.fnames = []
                with open(os.path.join(self.root, "segmentations.csv")) as f:
                    header = f.readline().strip().split(",")
                    assert header == ["Filename", "Frame", "Path", "Phase"]
                    for line in f:
                        filename, frame, path, phase = line.strip().split(',')
                        if filename not in fnames:
                            continue
                        self.fnames.append(filename)
                        self.segmentations.append((filename, int(frame), path, phase))

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = echonet.utils.loadvideo(video).astype(np.float32)

        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)

            pad_i, pad_j = np.random.randint(0, 2 * self.pad + 1, 2)

            # Explicit pad and crop
            # Uses significantly more memory
            # temp = np.zeros((c, f, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            # temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
            # video = temp[:, :, pad_i:(pad_i + h), pad_j:(pad_j + w)]
            # assert video.shape == (c, f, h, w)

            pad_i -= self.pad
            pad_j -= self.pad

            # Shift video that is still inside of crop
            video[:, :, max(0, -pad_i):min(h, h - pad_i), max(0, -pad_j):min(w, w - pad_j)] = video[:, :, max(0, pad_i):min(h, h + pad_i), max(0, pad_j):min(w, w + pad_j)]

            # Zero out areas that are not inside original window
            video[:, :, :max(0, -pad_i), :] = 0
            video[:, :, min(h, h - pad_i):, :] = 0
            video[:, :, :, :max(0, -pad_j)] = 0
            video[:, :, :, min(w, w - pad_j):] = 0

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
            if start.size > self.max_clips:
                # TODO: this messes up the clip number in test-time aug
                # Might need to have a clip index target
                start = np.random.choice(start, self.max_clips, replace=False)
                start.sort()
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Gather targets
        target = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "Index":
                target.append(self.segmentations[index][1])
            elif t == "Frame":
                frame = self.segmentations[index][1]

                # if frame is None or frame >= video.shape[1]:
                #     target.append(np.full((video.shape[0], video.shape[2], video.shape[3]), math.nan, video.dtype))
                # else:
                target.append(video[:, frame, :, :])
            elif t == "Phase":
                target.append(self.segmentations[index][3])
            elif t == "Trace":
                mask = np.array(PIL.Image.open(os.path.join(self.root, "Segmentations", self.segmentations[index][2])))
                r, c = mask.nonzero()

                if self.pad is not None:
                    # TODO: check that adjustment is the right way
                    r -= pad_i
                    c -= pad_j

                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)

                valid = (0 <= r) & (r < video.shape[2]) & (0 <= c) & (c < video.shape[3])
                r = r[valid]
                c = c[valid]
                mask[r, c] = 1

                target.append(mask)
            elif t in ["LargeApex", "LargeBase", "SmallApex", "SmallBase"]:
                if t == "LargeApex" or t == "LargeBase":
                    frame = self.frames[key][-1]
                else:
                    frame = self.frames[key][0]
                trace = self.trace[key][frame]
                if t == "LargeApex" or t == "SmallApex":
                    if hasattr(self, "apex"):
                        x, y = self.apex[key][frame]
                    else:
                        x, y = trace[0, 0], trace[0, 1]
                else:
                    if hasattr(self, "base"):
                        x, y = self.base[key][frame]
                    else:
                        try:
                            x, y = trace[0, 2], trace[0, 3]
                        except:
                            x, y = 0, 0
                x = round(x)
                y = round(y)

                if self.pad is not None:
                    x -= pad_j
                    y -= pad_i

                    # Apex/base falls slightly off image
                    # Move back onto image, as long as close
                    assert -self.pad <= x < video.shape[2] + self.pad
                    x = min(max(x, 0), video.shape[2] - 1)
                    assert -self.pad <= y < video.shape[3] + self.pad
                    y = min(max(y, 0), video.shape[3] - 1)

                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[y, x] = 1
                target.append(mask)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    # target.append(np.float32(self.outcome[index][self.header.index(t)]))  # TODO: is floating necessary
                    target.append(self.outcome[index][self.header.index(t)])

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Select clips from video
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        return torch.as_tensor(video), target

    def __len__(self):
        if not self.segmentation:
            return len(self.fnames)
        else:
            return len(self.segmentations)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)
