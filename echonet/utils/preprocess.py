import os
import pydicom
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import click


@click.command("preprocess")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", type=click.Path(file_okay=False))
@click.option("--crop_size", type=(int, int), default=(112, 112))
@click.option("--flip", type=bool, default=False)
def run(data_dir, output, crop_size, flip):
    """
    Preprocesses videos in the data_dir and saves them to the output directory.
    """
    dcm_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".dcm"):
                dcm_paths.append(Path(os.path.join(root, file)))

    output = Path(output)
    os.makedirs(output, exist_ok=True)

    for dcm_path in tqdm(dcm_paths, desc="Preprocessing DICOM files"):
        if not os.path.exists(os.path.join(output, dcm_path.stem + ".avi")):
            try:
                preprocess_video(dcm_path, output, crop_size, flip)
            except Exception as e:
                print(f"Error while preprocessing {dcm_path}: {e}")
        else:
            print(dcm_path, "has already been preprocessed.")


def masking(output):
    dimension = output.shape[0]
    
    # Mask pixels outside of scanning sector
    m1, m2 = np.meshgrid(np.arange(dimension), np.arange(dimension))

    image_mask = ((m1 + m2) > int(dimension / 2) + int(dimension / 10))
    image_mask *= ((m1 - m2) < int(dimension / 2) + int(dimension / 10))
    image_mask = np.reshape(image_mask, (dimension, dimension)).astype(np.int8)
    maskedImage = cv2.bitwise_and(output, output, mask=image_mask)

    return maskedImage


def preprocess_video(fileToProcess, destinationFolder, cropSize=(256, 256), flip=False):
    try:
        fileName = fileToProcess.stem

        # Load data from DICOM file
        dicom_dataset = pydicom.dcmread(fileToProcess, force=True)

        # Extract pixel array from the DICOM dataset
        pixel_array = dicom_dataset.pixel_array
        if len(pixel_array.shape) == 3:
            pixel_array = np.stack([pixel_array, pixel_array, pixel_array], axis=3)

        # Crop rows containing predominantly black pixels
        frame0 = pixel_array[0]
        mean = np.mean(frame0, axis=1)
        mean = np.mean(mean, axis=1)
        try:
            yCrop = np.where(mean < 1)[0][0]
        except:
            yCrop = 0
        pixel_array = pixel_array[:, yCrop:, :, :]

        # Crop frames if height is not equal to width
        bias = int(np.abs(pixel_array.shape[2] - pixel_array.shape[1]) / 2)
        if bias > 0:
            if pixel_array.shape[1] < pixel_array.shape[2]:
                pixel_array = pixel_array[:, :, bias:-bias, :]
            else:
                pixel_array = pixel_array[:, bias:-bias, :, :]
        frames, height, width, channels = pixel_array.shape

        # Extract frame rate from DICOM tags
        if hasattr(dicom_dataset, "RecommendedDisplayFrameRate"):
            fps = int(dicom_dataset.RecommendedDisplayFrameRate)
        elif hasattr(dicom_dataset, "FrameTime"):
            fps = int(round(1000 / float(dicom_dataset.FrameTime)))
        else:
            fps = 50
            print("Couldn't find frame rate. Using the default value of " + str(fps) + " frames per second.")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_filename = os.path.join(destinationFolder, fileName + '.avi')
        out = cv2.VideoWriter(video_filename, fourcc, fps, cropSize)

        # Iterate through the frames of the DICOM file
        for i in range(frames):
            outputA = pixel_array[i, :, :, 0]

            # Resize frame
            smallOutput = outputA[int(height / 10):(height - int(height / 10)),
                                  int(height / 10):(height - int(height / 10))]
            output = cv2.resize(smallOutput, cropSize, interpolation=cv2.INTER_CUBIC)

            # Mask image
            final_output = masking(output)

            # Flip horizontally (if needed)
            if flip:
                final_output = np.flip(final_output, axis=1)

            # Create and save the final output frame
            final_output = cv2.merge([final_output, final_output, final_output])
            out.write(final_output)

        out.release()
    except Exception as e:
        raise e
